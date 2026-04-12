"""Database-backed storage for AI-generated notifications and proactive alerts."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set

# Valid user_id pattern: alphanumeric, hyphens, underscores, dots, max 64 chars
VALID_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._-]{1,64}$')

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)


class _NotificationBase(DeclarativeBase):
    pass


class AINotificationRow(_NotificationBase):
    __tablename__ = "ai_notifications"
    __table_args__ = (
        UniqueConstraint("user_id", "notification_key", name="uq_ai_notifications_user_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    notification_key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    origin: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


@dataclass
class AINotificationRecord:
    id: int
    user_id: str
    notification_key: str
    kind: str
    origin: str
    severity: str
    status: str
    title: str
    summary: str
    details: Optional[str]
    score: Optional[float]
    payload: Dict[str, Any]
    first_seen_at: datetime
    last_seen_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class AINotificationStore:
    """Persists and tracks AI notifications with upsert semantics."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._available = False
        self._engine = None
        self._session_factory = None

        try:
            self._engine = create_engine(database_url, pool_pre_ping=True)
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, class_=Session)
            _NotificationBase.metadata.create_all(self._engine)
            self._available = True
        except Exception as exc:
            logger.error("Failed to initialize AI notification DB store: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._available and self._session_factory is not None

    def _new_session(self) -> Session:
        if not self._session_factory:
            raise RuntimeError("AI notification DB store is not initialized")
        return self._session_factory()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_user_id(user_id: Optional[str]) -> str:
        normalized = str(user_id or "single_user").strip()
        # Validate user_id to prevent injection attacks
        if normalized and not VALID_USER_ID_PATTERN.match(normalized):
            logger.warning(f"Invalid user_id format rejected: {repr(normalized[:100])}")
            raise ValueError(f"Invalid user_id format. Must match pattern: {VALID_USER_ID_PATTERN.pattern}")
        return normalized or "single_user"

    @staticmethod
    def _normalize_origin(origin: Optional[str]) -> str:
        normalized = str(origin or "ai_notifications_v1").strip().lower()
        return normalized or "ai_notifications_v1"

    @staticmethod
    def _normalize_severity(severity: Optional[str]) -> str:
        normalized = str(severity or "medium").strip().lower()
        if normalized not in {"low", "medium", "high", "critical"}:
            return "medium"
        return normalized

    @staticmethod
    def _normalize_status(status: Optional[str]) -> str:
        normalized = str(status or "active").strip().lower()
        if normalized not in {"active", "acknowledged", "resolved"}:
            return "active"
        return normalized

    @staticmethod
    def _serialize_payload(payload: Optional[Dict[str, Any]]) -> str:
        safe_payload = payload if isinstance(payload, dict) else {}
        try:
            return json.dumps(safe_payload, ensure_ascii=False)
        except Exception:
            return "{}"

    @staticmethod
    def _deserialize_payload(raw_payload: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_payload)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _to_datetime(value: Optional[datetime]) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def _row_to_record(self, row: AINotificationRow) -> AINotificationRecord:
        return AINotificationRecord(
            id=row.id,
            user_id=row.user_id,
            notification_key=row.notification_key,
            kind=row.kind,
            origin=row.origin,
            severity=row.severity,
            status=row.status,
            title=row.title,
            summary=row.summary,
            details=row.details,
            score=row.score,
            payload=self._deserialize_payload(row.payload_json),
            first_seen_at=self._to_datetime(row.first_seen_at),
            last_seen_at=self._to_datetime(row.last_seen_at),
            acknowledged_at=row.acknowledged_at,
            resolved_at=row.resolved_at,
            created_at=self._to_datetime(row.created_at),
            updated_at=self._to_datetime(row.updated_at),
        )

    def upsert_notification(
        self,
        *,
        user_id: str,
        notification_key: str,
        kind: str,
        severity: str,
        title: str,
        summary: str,
        details: Optional[str] = None,
        score: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        origin: str = "ai_notifications_v1",
        status: str = "active",
    ) -> Optional[AINotificationRecord]:
        if not self.is_available:
            return None

        normalized_user = self._normalize_user_id(user_id)
        normalized_key = str(notification_key or "").strip()
        if not normalized_key:
            return None

        normalized_kind = str(kind or "signal").strip().lower() or "signal"
        normalized_origin = self._normalize_origin(origin)
        normalized_severity = self._normalize_severity(severity)
        normalized_status = self._normalize_status(status)
        now = self._now()

        try:
            with self._new_session() as session:
                row = session.execute(
                    select(AINotificationRow).where(
                        AINotificationRow.user_id == normalized_user,
                        AINotificationRow.notification_key == normalized_key,
                    )
                ).scalar_one_or_none()

                if row:
                    # Preserve acknowledged state unless caller explicitly resolves.
                    if row.status == "acknowledged" and normalized_status == "active":
                        normalized_status = "acknowledged"

                    row.kind = normalized_kind
                    row.origin = normalized_origin
                    row.severity = normalized_severity
                    row.status = normalized_status
                    row.title = str(title or "").strip() or "AI Notification"
                    row.summary = str(summary or "").strip() or row.title
                    row.details = str(details).strip() if details is not None else None
                    row.score = float(score) if score is not None else None
                    row.payload_json = self._serialize_payload(payload)
                    row.last_seen_at = now
                    row.updated_at = now

                    if normalized_status == "acknowledged":
                        row.acknowledged_at = row.acknowledged_at or now
                        row.resolved_at = None
                    elif normalized_status == "resolved":
                        row.resolved_at = row.resolved_at or now
                    else:
                        row.resolved_at = None
                else:
                    acknowledged_at = now if normalized_status == "acknowledged" else None
                    resolved_at = now if normalized_status == "resolved" else None
                    row = AINotificationRow(
                        user_id=normalized_user,
                        notification_key=normalized_key,
                        kind=normalized_kind,
                        origin=normalized_origin,
                        severity=normalized_severity,
                        status=normalized_status,
                        title=str(title or "").strip() or "AI Notification",
                        summary=str(summary or "").strip() or "Generated AI insight",
                        details=str(details).strip() if details is not None else None,
                        score=float(score) if score is not None else None,
                        payload_json=self._serialize_payload(payload),
                        first_seen_at=now,
                        last_seen_at=now,
                        acknowledged_at=acknowledged_at,
                        resolved_at=resolved_at,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(row)

                session.commit()
                session.refresh(row)
                return self._row_to_record(row)
        except Exception as exc:
            logger.error(
                "Failed to upsert AI notification user=%s key=%s: %s",
                normalized_user,
                normalized_key,
                exc,
            )
            return None

    def mark_stale_notifications_resolved(
        self,
        *,
        user_id: str,
        active_keys: Sequence[str],
        origin: str = "ai_notifications_v1",
    ) -> int:
        if not self.is_available:
            return 0

        normalized_user = self._normalize_user_id(user_id)
        normalized_origin = self._normalize_origin(origin)
        active_key_set: Set[str] = {str(key or "").strip() for key in active_keys if str(key or "").strip()}
        now = self._now()
        resolved_count = 0

        try:
            with self._new_session() as session:
                rows = session.execute(
                    select(AINotificationRow).where(
                        AINotificationRow.user_id == normalized_user,
                        AINotificationRow.origin == normalized_origin,
                        AINotificationRow.status != "resolved",
                    )
                ).scalars().all()

                for row in rows:
                    if row.notification_key in active_key_set:
                        continue
                    row.status = "resolved"
                    row.resolved_at = now
                    row.updated_at = now
                    resolved_count += 1

                if resolved_count:
                    session.commit()
                else:
                    session.rollback()
        except Exception as exc:
            logger.error("Failed to resolve stale AI notifications for user=%s: %s", normalized_user, exc)
            return 0

        return resolved_count

    def list_notifications(
        self,
        *,
        user_id: str,
        limit: int = 40,
        include_resolved: bool = False,
    ) -> List[AINotificationRecord]:
        if not self.is_available:
            return []

        normalized_user = self._normalize_user_id(user_id)
        safe_limit = max(1, min(200, int(limit)))

        try:
            with self._new_session() as session:
                query = select(AINotificationRow).where(AINotificationRow.user_id == normalized_user)
                if not include_resolved:
                    query = query.where(AINotificationRow.status != "resolved")

                rows = session.execute(query).scalars().all()
                records = [self._row_to_record(row) for row in rows]

                severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
                status_rank = {"active": 0, "acknowledged": 1, "resolved": 2}

                records.sort(
                    key=lambda record: (
                        status_rank.get(record.status, 3),
                        severity_rank.get(record.severity, 9),
                        -self._to_datetime(record.last_seen_at).timestamp(),
                    )
                )
                return records[:safe_limit]
        except Exception as exc:
            logger.error("Failed to list AI notifications for user=%s: %s", normalized_user, exc)
            return []

    def set_acknowledged(
        self,
        *,
        user_id: str,
        notification_id: int,
        acknowledged: bool,
    ) -> Optional[AINotificationRecord]:
        if not self.is_available:
            return None

        normalized_user = self._normalize_user_id(user_id)
        now = self._now()

        try:
            with self._new_session() as session:
                row = session.execute(
                    select(AINotificationRow).where(
                        AINotificationRow.user_id == normalized_user,
                        AINotificationRow.id == int(notification_id),
                    )
                ).scalar_one_or_none()

                if not row:
                    return None

                if acknowledged:
                    row.status = "acknowledged"
                    row.acknowledged_at = now
                    row.resolved_at = None
                else:
                    row.status = "active"
                    row.resolved_at = None

                row.updated_at = now
                session.commit()
                session.refresh(row)
                return self._row_to_record(row)
        except Exception as exc:
            logger.error(
                "Failed to update AI notification acknowledgement user=%s id=%s: %s",
                normalized_user,
                notification_id,
                exc,
            )
            return None


def _normalize_database_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgres://"):
        return "postgresql://" + normalized[len("postgres://") :]
    return normalized


_ai_notification_store_lock = Lock()
_ai_notification_store_instance: Optional[AINotificationStore] = None
_ai_notification_store_initialized = False


def get_ai_notification_store() -> Optional[AINotificationStore]:
    """Get singleton AI notification store if a database URL is configured."""
    global _ai_notification_store_initialized, _ai_notification_store_instance

    with _ai_notification_store_lock:
        if _ai_notification_store_initialized:
            return _ai_notification_store_instance

        database_url = (
            os.getenv("AI_NOTIFICATIONS_DATABASE_URL")
            or os.getenv("CHECKUPS_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()

        if not database_url:
            _ai_notification_store_initialized = True
            return None

        store = AINotificationStore(_normalize_database_url(database_url))
        _ai_notification_store_instance = store if store.is_available else None
        _ai_notification_store_initialized = True

        if _ai_notification_store_instance:
            logger.info("AI notification store enabled via database backend")
        else:
            logger.warning("AI notification store unavailable; initialization failed")

        return _ai_notification_store_instance
