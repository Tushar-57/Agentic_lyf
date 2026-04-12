"""Database-backed storage for knowledge entry payloads (excluding embeddings)."""

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

from ..models.knowledge import KnowledgeEntry, KnowledgeEntrySubType, KnowledgeEntryType

logger = logging.getLogger(__name__)


class _KnowledgeBase(DeclarativeBase):
    pass


class KnowledgeEntryRow(_KnowledgeBase):
    __tablename__ = "knowledge_entries"
    __table_args__ = (
        UniqueConstraint("user_id", "entry_id", name="uq_knowledge_entries_user_entry"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    entry_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    entry_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entry_sub_type: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    tags_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class KnowledgeDbStore:
    """Persists canonical knowledge entry data into SQL tables."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._available = False
        self._engine = None
        self._session_factory = None

        try:
            self._engine = create_engine(database_url, pool_pre_ping=True)
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, class_=Session)
            _KnowledgeBase.metadata.create_all(self._engine)
            self._available = True
        except Exception as exc:
            logger.error("Failed to initialize knowledge DB store: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._available and self._session_factory is not None

    def _new_session(self) -> Session:
        if not self._session_factory:
            raise RuntimeError("Knowledge DB store is not initialized")
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
    def _serialize_json(payload: Any, default: Any) -> str:
        data = payload if payload is not None else default
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return json.dumps(default, ensure_ascii=False)

    @staticmethod
    def _deserialize_json(raw_payload: str, default: Any) -> Any:
        try:
            parsed = json.loads(raw_payload)
            if isinstance(default, dict) and not isinstance(parsed, dict):
                return default
            if isinstance(default, list) and not isinstance(parsed, list):
                return default
            return parsed
        except Exception:
            return default

    @staticmethod
    def _normalize_timestamp(value: Optional[datetime]) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return value.astimezone(timezone.utc)
            return value.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _safe_entry_type(value: str) -> KnowledgeEntryType:
        try:
            return KnowledgeEntryType(value)
        except Exception:
            return KnowledgeEntryType.MEMORY

    @staticmethod
    def _safe_entry_sub_type(value: str) -> KnowledgeEntrySubType:
        try:
            return KnowledgeEntrySubType(value)
        except Exception:
            return KnowledgeEntrySubType.MISC_INTERACTION

    def _row_to_entry(self, row: KnowledgeEntryRow) -> KnowledgeEntry:
        metadata = self._deserialize_json(row.metadata_json, default={})
        tags = self._deserialize_json(row.tags_json, default=[])

        return KnowledgeEntry(
            entry_id=row.entry_id,
            user_id=row.user_id,
            entry_type=self._safe_entry_type(row.entry_type),
            entry_sub_type=self._safe_entry_sub_type(row.entry_sub_type),
            category=row.category,
            title=row.title,
            content=row.content,
            metadata=metadata if isinstance(metadata, dict) else {},
            tags=tags if isinstance(tags, list) else [],
            created_at=row.created_at if isinstance(row.created_at, datetime) else datetime.now(timezone.utc),
            updated_at=row.updated_at if isinstance(row.updated_at, datetime) else datetime.now(timezone.utc),
            embedding=None,
        )

    def upsert_entry(self, entry: KnowledgeEntry) -> bool:
        if not self.is_available:
            return False

        user_id = self._normalize_user_id(entry.user_id)
        created_at = self._normalize_timestamp(entry.created_at)
        updated_at = self._normalize_timestamp(entry.updated_at)

        try:
            with self._new_session() as session:
                row = session.execute(
                    select(KnowledgeEntryRow).where(
                        KnowledgeEntryRow.user_id == user_id,
                        KnowledgeEntryRow.entry_id == entry.entry_id,
                    )
                ).scalar_one_or_none()

                if row:
                    row.entry_type = str(getattr(entry.entry_type, "value", entry.entry_type))
                    row.entry_sub_type = str(getattr(entry.entry_sub_type, "value", entry.entry_sub_type))
                    row.category = str(entry.category)
                    row.title = str(entry.title)
                    row.content = str(entry.content)
                    row.metadata_json = self._serialize_json(entry.metadata, default={})
                    row.tags_json = self._serialize_json(entry.tags, default=[])
                    row.updated_at = updated_at
                else:
                    row = KnowledgeEntryRow(
                        user_id=user_id,
                        entry_id=str(entry.entry_id),
                        entry_type=str(getattr(entry.entry_type, "value", entry.entry_type)),
                        entry_sub_type=str(getattr(entry.entry_sub_type, "value", entry.entry_sub_type)),
                        category=str(entry.category),
                        title=str(entry.title),
                        content=str(entry.content),
                        metadata_json=self._serialize_json(entry.metadata, default={}),
                        tags_json=self._serialize_json(entry.tags, default=[]),
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                    session.add(row)

                session.commit()
                return True
        except Exception as exc:
            logger.error("Failed to upsert knowledge entry %s: %s", entry.entry_id, exc)
            return False

    def get_entry(self, user_id: str, entry_id: str) -> Optional[KnowledgeEntry]:
        if not self.is_available:
            return None

        normalized_user = self._normalize_user_id(user_id)
        try:
            with self._new_session() as session:
                row = session.execute(
                    select(KnowledgeEntryRow).where(
                        KnowledgeEntryRow.user_id == normalized_user,
                        KnowledgeEntryRow.entry_id == str(entry_id),
                    )
                ).scalar_one_or_none()

                if not row:
                    return None
                return self._row_to_entry(row)
        except Exception as exc:
            logger.error("Failed to fetch knowledge entry %s: %s", entry_id, exc)
            return None

    def list_entries(
        self,
        user_id: str,
        *,
        category: Optional[str] = None,
        entry_type: Optional[KnowledgeEntryType] = None,
    ) -> List[KnowledgeEntry]:
        if not self.is_available:
            return []

        normalized_user = self._normalize_user_id(user_id)

        try:
            with self._new_session() as session:
                query = select(KnowledgeEntryRow).where(KnowledgeEntryRow.user_id == normalized_user)

                if category:
                    query = query.where(KnowledgeEntryRow.category == category)

                if entry_type:
                    query = query.where(KnowledgeEntryRow.entry_type == str(getattr(entry_type, "value", entry_type)))

                query = query.order_by(KnowledgeEntryRow.updated_at.desc())
                rows = session.execute(query).scalars().all()
                return [self._row_to_entry(row) for row in rows]
        except Exception as exc:
            logger.error("Failed to list knowledge entries for user %s: %s", normalized_user, exc)
            return []

    def delete_entry(self, user_id: str, entry_id: str) -> bool:
        if not self.is_available:
            return False

        normalized_user = self._normalize_user_id(user_id)
        try:
            with self._new_session() as session:
                row = session.execute(
                    select(KnowledgeEntryRow).where(
                        KnowledgeEntryRow.user_id == normalized_user,
                        KnowledgeEntryRow.entry_id == str(entry_id),
                    )
                ).scalar_one_or_none()

                if not row:
                    return False

                session.delete(row)
                session.commit()
                return True
        except Exception as exc:
            logger.error("Failed to delete knowledge entry %s: %s", entry_id, exc)
            return False

    def delete_entries(self, user_id: str, entry_ids: List[str]) -> int:
        if not self.is_available:
            return 0

        normalized_user = self._normalize_user_id(user_id)
        normalized_ids = [str(entry_id) for entry_id in entry_ids if entry_id]
        if not normalized_ids:
            return 0

        try:
            with self._new_session() as session:
                result = session.execute(
                    delete(KnowledgeEntryRow).where(
                        KnowledgeEntryRow.user_id == normalized_user,
                        KnowledgeEntryRow.entry_id.in_(normalized_ids),
                    )
                )
                session.commit()
                return int(result.rowcount or 0)
        except Exception as exc:
            logger.error("Failed to bulk delete knowledge entries: %s", exc)
            return 0

    def clear_user_entries(self, user_id: str) -> int:
        if not self.is_available:
            return 0

        normalized_user = self._normalize_user_id(user_id)
        try:
            with self._new_session() as session:
                result = session.execute(
                    delete(KnowledgeEntryRow).where(KnowledgeEntryRow.user_id == normalized_user)
                )
                session.commit()
                return int(result.rowcount or 0)
        except Exception as exc:
            logger.error("Failed to clear knowledge entries for user %s: %s", normalized_user, exc)
            return 0

    def count_user_entries(self, user_id: str) -> int:
        if not self.is_available:
            return 0

        normalized_user = self._normalize_user_id(user_id)
        try:
            with self._new_session() as session:
                rows = session.execute(
                    select(KnowledgeEntryRow.id).where(KnowledgeEntryRow.user_id == normalized_user)
                ).all()
                return len(rows)
        except Exception as exc:
            logger.error("Failed to count knowledge entries for user %s: %s", normalized_user, exc)
            return 0


_knowledge_db_store_lock = Lock()
_knowledge_db_store_instance: Optional[KnowledgeDbStore] = None
_knowledge_db_store_initialized = False


def _normalize_database_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgres://"):
        return "postgresql://" + normalized[len("postgres://"):]
    return normalized


def get_knowledge_db_store() -> Optional[KnowledgeDbStore]:
    """Get singleton knowledge DB store when a database URL is configured."""
    global _knowledge_db_store_initialized, _knowledge_db_store_instance

    with _knowledge_db_store_lock:
        if _knowledge_db_store_initialized:
            return _knowledge_db_store_instance

        database_url = (
            os.getenv("KNOWLEDGE_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()

        if not database_url:
            _knowledge_db_store_initialized = True
            return None

        store = KnowledgeDbStore(_normalize_database_url(database_url))
        _knowledge_db_store_instance = store if store.is_available else None
        _knowledge_db_store_initialized = True

        if _knowledge_db_store_instance:
            logger.info("Knowledge DB store enabled via database backend")
        else:
            logger.warning("Knowledge DB store unavailable; initialization failed")

        return _knowledge_db_store_instance
