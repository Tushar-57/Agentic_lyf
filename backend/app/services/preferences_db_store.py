"""Database-backed storage for structured user preferences."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

# Valid user_id pattern: alphanumeric, hyphens, underscores, dots, max 64 chars
VALID_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._-]{1,64}$')

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)


class _PreferencesBase(DeclarativeBase):
    pass


class UserPreferenceRow(_PreferencesBase):
    __tablename__ = "knowledge_user_preferences"
    __table_args__ = (
        UniqueConstraint("user_id", "category", "preference_key", name="uq_pref_user_category_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    preference_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PreferenceDbStore:
    """Persists user preferences as individual preference records."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._available = False
        self._engine = None
        self._session_factory = None

        try:
            self._engine = create_engine(database_url, pool_pre_ping=True)
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, class_=Session)
            _PreferencesBase.metadata.create_all(self._engine)
            self._available = True
        except Exception as exc:
            logger.error("Failed to initialize preference DB store: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._available and self._session_factory is not None

    def _new_session(self) -> Session:
        if not self._session_factory:
            raise RuntimeError("Preference DB store is not initialized")
        return self._session_factory()

    @staticmethod
    def _normalize_user_id(user_id: Optional[str]) -> str:
        normalized = str(user_id or "single_user").strip()
        # Validate user_id to prevent injection attacks
        if normalized and not VALID_USER_ID_PATTERN.match(normalized):
            logger.warning(f"Invalid user_id format rejected: {repr(normalized[:100])}")
            raise ValueError(f"Invalid user_id format. Must match pattern: {VALID_USER_ID_PATTERN.pattern}")
        return normalized or "single_user"

    @staticmethod
    def _normalize_category(category: Any) -> str:
        normalized = str(category or "general").strip().lower()
        return normalized or "general"

    @staticmethod
    def _normalize_key(key: Any) -> str:
        normalized = str(key or "").strip()
        return normalized

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _serialize_json(payload: Any) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return json.dumps(str(payload), ensure_ascii=False)

    @staticmethod
    def _deserialize_json(raw_payload: str) -> Any:
        try:
            return json.loads(raw_payload)
        except Exception:
            return raw_payload

    def _flatten_preferences(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []

        for category, values in (preferences or {}).items():
            if category == "user_id" or not isinstance(values, dict):
                continue

            descriptions: Dict[str, str] = {}
            for key, value in values.items():
                if not isinstance(key, str) or not key.startswith("__") or not key.endswith("_description"):
                    continue
                pref_key = key[len("__") : -len("_description")]
                descriptions[pref_key] = str(value)

            for key, value in values.items():
                normalized_key = self._normalize_key(key)
                if not normalized_key or normalized_key.startswith("__"):
                    continue

                flattened.append(
                    {
                        "category": self._normalize_category(category),
                        "key": normalized_key,
                        "value": value,
                        "description": descriptions.get(normalized_key),
                    }
                )

        return flattened

    def upsert_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        if not self.is_available:
            return False

        normalized_user = self._normalize_user_id(user_id)
        flattened = self._flatten_preferences(preferences)
        now = self._now()

        try:
            with self._new_session() as session:
                session.execute(
                    delete(UserPreferenceRow).where(UserPreferenceRow.user_id == normalized_user)
                )

                for preference in flattened:
                    row = UserPreferenceRow(
                        user_id=normalized_user,
                        category=preference["category"],
                        preference_key=preference["key"],
                        value_json=self._serialize_json(preference["value"]),
                        description=preference.get("description"),
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(row)

                session.commit()
                return True
        except Exception as exc:
            logger.error("Failed to upsert user preferences for %s: %s", normalized_user, exc)
            return False

    def load_preferences(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        if not self.is_available:
            return {}

        normalized_user = self._normalize_user_id(user_id)

        try:
            with self._new_session() as session:
                rows = session.execute(
                    select(UserPreferenceRow).where(UserPreferenceRow.user_id == normalized_user)
                ).scalars().all()

                grouped: Dict[str, Dict[str, Any]] = {}
                for row in rows:
                    category = self._normalize_category(row.category)
                    category_bucket = grouped.setdefault(category, {})
                    category_bucket[row.preference_key] = self._deserialize_json(row.value_json)
                    if row.description:
                        category_bucket[f"__{row.preference_key}_description"] = row.description

                return grouped
        except Exception as exc:
            logger.error("Failed to load user preferences for %s: %s", normalized_user, exc)
            return {}

    def list_categories(self, user_id: str) -> List[str]:
        if not self.is_available:
            return []

        normalized_user = self._normalize_user_id(user_id)
        try:
            with self._new_session() as session:
                rows = session.execute(
                    select(UserPreferenceRow.category).where(UserPreferenceRow.user_id == normalized_user)
                ).all()
                categories = sorted({self._normalize_category(row[0]) for row in rows if row and row[0]})
                return categories
        except Exception as exc:
            logger.error("Failed to list preference categories for %s: %s", normalized_user, exc)
            return []

    def count_preferences(self, user_id: str) -> int:
        if not self.is_available:
            return 0

        normalized_user = self._normalize_user_id(user_id)
        try:
            with self._new_session() as session:
                rows = session.execute(
                    select(UserPreferenceRow.id).where(UserPreferenceRow.user_id == normalized_user)
                ).all()
                return len(rows)
        except Exception as exc:
            logger.error("Failed to count preferences for %s: %s", normalized_user, exc)
            return 0


def _normalize_database_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgres://"):
        return "postgresql://" + normalized[len("postgres://") :]
    return normalized


_preference_db_store_lock = Lock()
_preference_db_store_instance: Optional[PreferenceDbStore] = None
_preference_db_store_initialized = False


def get_preference_db_store() -> Optional[PreferenceDbStore]:
    """Get singleton preference DB store when a database URL is configured."""
    global _preference_db_store_initialized, _preference_db_store_instance

    with _preference_db_store_lock:
        if _preference_db_store_initialized:
            return _preference_db_store_instance

        database_url = (
            os.getenv("PREFERENCES_DATABASE_URL")
            or os.getenv("KNOWLEDGE_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()

        if not database_url:
            _preference_db_store_initialized = True
            return None

        store = PreferenceDbStore(_normalize_database_url(database_url))
        _preference_db_store_instance = store if store.is_available else None
        _preference_db_store_initialized = True

        if _preference_db_store_instance:
            logger.info("preference_db_enabled", "Preference DB store enabled via database backend")
        else:
            logger.warning("preference_db_unavailable", "Preference DB store unavailable; initialization failed")

        return _preference_db_store_instance
