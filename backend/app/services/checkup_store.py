"""Database-backed storage for daily morning/evening checkup payloads."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from sqlalchemy import Date, DateTime, Integer, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

logger = logging.getLogger(__name__)


class _CheckupBase(DeclarativeBase):
    pass


class DailyCheckupRow(_CheckupBase):
    __tablename__ = "daily_checkups"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "checkup_type",
            "checkup_date",
            name="uq_daily_checkups_user_type_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    checkup_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    checkup_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


@dataclass
class DailyCheckupRecord:
    user_id: str
    checkup_type: str
    checkup_date: date
    payload: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DailyCheckupStore:
    """Persistent checkup store backed by SQLAlchemy."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._available = False
        self._engine = None
        self._session_factory = None

        try:
            self._engine = create_engine(database_url, pool_pre_ping=True)
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, class_=Session)
            _CheckupBase.metadata.create_all(self._engine)
            self._available = True
        except Exception as exc:
            logger.error("Failed to initialize daily checkup DB store: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._available and self._session_factory is not None

    def _new_session(self) -> Session:
        if not self._session_factory:
            raise RuntimeError("Daily checkup store is not initialized")
        return self._session_factory()

    @staticmethod
    def _normalize_checkup_type(checkup_type: str) -> str:
        normalized = str(checkup_type or "").strip().lower()
        if normalized not in {"morning", "evening"}:
            raise ValueError(f"Unsupported checkup type: {checkup_type}")
        return normalized

    @staticmethod
    def _normalize_checkup_date(value: Any) -> date:
        if isinstance(value, date):
            return value

        text = str(value or "").strip()
        if not text:
            raise ValueError("checkup_date is required")

        return date.fromisoformat(text[:10])

    @staticmethod
    def _normalize_timestamp(value: Optional[datetime]) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.now(timezone.utc)

    @staticmethod
    def _deserialize_payload(raw_payload: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_payload)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def upsert_checkup(
        self,
        *,
        user_id: str,
        checkup_type: str,
        checkup_date: Any,
        payload: Dict[str, Any],
    ) -> bool:
        """Insert/update a checkup payload for user+type+date."""
        if not self.is_available:
            return False

        normalized_type = self._normalize_checkup_type(checkup_type)
        normalized_date = self._normalize_checkup_date(checkup_date)
        normalized_user = str(user_id or "single_user").strip() or "single_user"

        payload_to_store = dict(payload or {})
        payload_to_store["checkup_type"] = normalized_type
        payload_to_store["date"] = normalized_date.isoformat()
        payload_to_store["checkup_date"] = normalized_date.isoformat()

        now = datetime.now(timezone.utc)
        serialized_payload = json.dumps(payload_to_store, ensure_ascii=False)

        try:
            with self._new_session() as session:
                row = session.execute(
                    select(DailyCheckupRow).where(
                        DailyCheckupRow.user_id == normalized_user,
                        DailyCheckupRow.checkup_type == normalized_type,
                        DailyCheckupRow.checkup_date == normalized_date,
                    )
                ).scalar_one_or_none()

                if row:
                    row.payload_json = serialized_payload
                    row.updated_at = now
                else:
                    row = DailyCheckupRow(
                        user_id=normalized_user,
                        checkup_type=normalized_type,
                        checkup_date=normalized_date,
                        payload_json=serialized_payload,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(row)

                session.commit()
                return True
        except Exception as exc:
            logger.error(
                "Failed to upsert daily checkup user=%s type=%s date=%s: %s",
                normalized_user,
                normalized_type,
                normalized_date,
                exc,
            )
            return False

    def list_checkups_for_user(self, user_id: str) -> List[DailyCheckupRecord]:
        """Return all checkup records for a user ordered by most recent."""
        if not self.is_available:
            return []

        normalized_user = str(user_id or "single_user").strip() or "single_user"
        records: List[DailyCheckupRecord] = []

        try:
            with self._new_session() as session:
                rows = session.execute(
                    select(DailyCheckupRow)
                    .where(DailyCheckupRow.user_id == normalized_user)
                    .order_by(DailyCheckupRow.checkup_date.desc(), DailyCheckupRow.updated_at.desc())
                ).scalars().all()

                for row in rows:
                    payload = self._deserialize_payload(row.payload_json)
                    payload.setdefault("checkup_type", row.checkup_type)
                    payload.setdefault("date", row.checkup_date.isoformat())
                    payload.setdefault("checkup_date", row.checkup_date.isoformat())

                    records.append(
                        DailyCheckupRecord(
                            user_id=row.user_id,
                            checkup_type=row.checkup_type,
                            checkup_date=row.checkup_date,
                            payload=payload,
                            created_at=self._normalize_timestamp(row.created_at),
                            updated_at=self._normalize_timestamp(row.updated_at),
                        )
                    )
        except Exception as exc:
            logger.error("Failed to list daily checkups for user=%s: %s", normalized_user, exc)

        return records

    def get_latest_checkups_for_user(self, user_id: str) -> Dict[str, DailyCheckupRecord]:
        """Return latest morning/evening records for the user."""
        latest: Dict[str, DailyCheckupRecord] = {}

        for record in self.list_checkups_for_user(user_id):
            if record.checkup_type in {"morning", "evening"} and record.checkup_type not in latest:
                latest[record.checkup_type] = record
            if "morning" in latest and "evening" in latest:
                break

        return latest


_checkup_store_lock = Lock()
_checkup_store_instance: Optional[DailyCheckupStore] = None
_checkup_store_initialized = False


def _normalize_database_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgres://"):
        return "postgresql://" + normalized[len("postgres://"):]
    return normalized


def get_daily_checkup_store() -> Optional[DailyCheckupStore]:
    """Get singleton checkup store if a DB URL is configured."""
    global _checkup_store_initialized, _checkup_store_instance

    with _checkup_store_lock:
        if _checkup_store_initialized:
            return _checkup_store_instance

        database_url = (
            os.getenv("CHECKUPS_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()

        if not database_url:
            _checkup_store_initialized = True
            return None

        store = DailyCheckupStore(_normalize_database_url(database_url))
        _checkup_store_instance = store if store.is_available else None
        _checkup_store_initialized = True

        if _checkup_store_instance:
            logger.info("checkup_store_enabled", "Daily checkup store enabled via database backend")
        else:
            logger.warning("checkup_store_unavailable", "Daily checkup store unavailable; database backend initialization failed")

        return _checkup_store_instance
