"""Request-scoped user resolution and storage-key helpers for multi-user isolation."""

from __future__ import annotations

import contextvars
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import jwt
from jwt import InvalidTokenError

if TYPE_CHECKING:
    from fastapi import Request
else:  # pragma: no cover - type-only fallback for lightweight tooling environments.
    Request = Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestUser:
    """Resolved user identity for the current request."""

    raw_user_id: str
    storage_key: str
    email: Optional[str] = None
    name: Optional[str] = None
    authenticated: bool = False
    source: str = "anonymous"


ANONYMOUS_USER = RequestUser(
    raw_user_id="single_user",
    storage_key="single_user",
    authenticated=False,
    source="anonymous",
)

_current_user_ctx: contextvars.ContextVar[RequestUser] = contextvars.ContextVar(
    "agentic_request_user",
    default=ANONYMOUS_USER,
)


def normalize_user_storage_key(raw_user_id: Optional[str]) -> str:
    """Create a stable filesystem-safe user key."""
    if not raw_user_id:
        return "single_user"

    normalized = raw_user_id.strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "_", normalized)
    normalized = normalized.strip("._-")

    if not normalized:
        return "single_user"

    return normalized[:128]


def _extract_bearer_token(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None

    if not header_value.lower().startswith("bearer "):
        return None

    candidate = header_value[7:].strip()
    if not candidate:
        return None

    if candidate.lower() in {"cookie-session", "session", "1"}:
        return None

    return candidate


def _jwt_secret() -> Optional[str]:
    secret = os.getenv("AGENTIC_BRIDGE_SECRET") or os.getenv("JWT_SECRET")
    if not secret:
        return None
    return secret


def _decode_token(token: str) -> Optional[Dict[str, Any]]:
    secret = _jwt_secret()
    if not secret:
        return None

    try:
        return jwt.decode(token, secret, algorithms=["HS512", "HS256"], options={"verify_aud": False})
    except InvalidTokenError:
        return None


def _user_from_claims(claims: Dict[str, Any], source: str) -> Optional[RequestUser]:
    raw_user_id = (
        claims.get("uid")
        or claims.get("user_id")
        or claims.get("sub")
        or claims.get("email")
    )

    if not raw_user_id:
        return None

    raw_user_id_str = str(raw_user_id)
    storage_key = normalize_user_storage_key(raw_user_id_str)

    return RequestUser(
        raw_user_id=raw_user_id_str,
        storage_key=storage_key,
        email=str(claims.get("email")) if claims.get("email") is not None else None,
        name=str(claims.get("name")) if claims.get("name") is not None else None,
        authenticated=True,
        source=source,
    )


def resolve_request_user(request: Request) -> RequestUser:
    """Resolve the current user from bridge headers or bearer auth."""
    bridge_token = request.headers.get("X-Agentic-Bridge-Token") or request.query_params.get("bridge_token")
    if bridge_token:
        claims = _decode_token(bridge_token)
        if claims:
            user = _user_from_claims(claims, source="bridge_token")
            if user:
                return user

    bearer_token = _extract_bearer_token(request.headers.get("Authorization"))
    if bearer_token:
        claims = _decode_token(bearer_token)
        if claims:
            user = _user_from_claims(claims, source="authorization")
            if user:
                return user

    hinted_user_id = request.headers.get("X-Agentic-User-Id") or request.query_params.get("user_id")
    if hinted_user_id:
        normalized = normalize_user_storage_key(hinted_user_id)
        return RequestUser(
            raw_user_id=str(hinted_user_id),
            storage_key=normalized,
            authenticated=False,
            source="hint",
        )

    return ANONYMOUS_USER


def set_request_user(user: RequestUser) -> contextvars.Token[RequestUser]:
    """Set request user into context."""
    return _current_user_ctx.set(user)


def reset_request_user(token: contextvars.Token[RequestUser]) -> None:
    """Reset request user context."""
    _current_user_ctx.reset(token)


def get_current_user() -> RequestUser:
    """Return request user context, defaulting to single_user."""
    return _current_user_ctx.get()


def get_current_user_id() -> str:
    """Return filesystem-safe user key for data partitioning."""
    return get_current_user().storage_key
