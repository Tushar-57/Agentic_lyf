"""
Knowledge base service providing CRUD operations and RAG functionality.
"""
import os
import hashlib
import json
import uuid
import logging
import re
import math
from time import monotonic
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set, Tuple

from ..models.knowledge import (
    KnowledgeEntry,
    KnowledgeEntrySubType, 
    KnowledgeEntryType, 
    KnowledgeQuery, 
    KnowledgeSearchResult,
    KnowledgeStats,
    UserPreferences
)
from ..llm.service import get_llm_service
from ..llm.base import EmbeddingRequest
from ..utils.logging import get_embedding_category_logger
from .vector_store import get_vector_store
from .knowledge_db_store import get_knowledge_db_store
from .preferences_db_store import get_preference_db_store

logger = logging.getLogger(__name__)
embedding_logger = get_embedding_category_logger("app.embedding.knowledge")


EMBEDDING_CACHE_KEY_FIELD = "_embedding_cache_key"
EMBEDDING_PROVIDER_COOLDOWN_SECONDS = 30.0


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, int(raw_value))
    except (TypeError, ValueError):
        return default


def _parse_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return default

    return min(maximum, max(minimum, parsed))


EMBEDDING_MAX_CHARS_PER_CHUNK = _parse_positive_int_env(
    "EMBEDDING_MAX_CHARS_PER_CHUNK",
    default=1200,
    minimum=400,
)
EMBEDDING_CHUNK_OVERLAP_CHARS = min(
    EMBEDDING_MAX_CHARS_PER_CHUNK // 2,
    _parse_positive_int_env("EMBEDDING_CHUNK_OVERLAP_CHARS", default=180, minimum=0),
)
EMBEDDING_MAX_CHUNKS_PER_ENTRY = _parse_positive_int_env(
    "EMBEDDING_MAX_CHUNKS_PER_ENTRY",
    default=8,
    minimum=1,
)
EMBEDDING_LOG_ENABLED = _parse_bool_env("EMBEDDING_LOG_ENABLED", default=True)
EMBEDDING_LOG_FULL_TEXT = _parse_bool_env("EMBEDDING_LOG_FULL_TEXT", default=True)
EMBEDDING_LOG_MAX_TEXT_CHARS = _parse_positive_int_env(
    "EMBEDDING_LOG_MAX_TEXT_CHARS",
    default=32000,
    minimum=200,
)

EMBEDDING_MICRO_ENTRY_MAX_CHARS = _parse_positive_int_env(
    "EMBEDDING_MICRO_ENTRY_MAX_CHARS",
    default=260,
    minimum=80,
)
EMBEDDING_SEMANTIC_SECTION_MAX_CHARS = _parse_positive_int_env(
    "EMBEDDING_SEMANTIC_SECTION_MAX_CHARS",
    default=420,
    minimum=180,
)

RAG_PRIMARY_SIMILARITY_THRESHOLD = _parse_float_env(
    "RAG_PRIMARY_SIMILARITY_THRESHOLD",
    default=0.58,
    minimum=0.0,
    maximum=1.0,
)
RAG_RELAXED_SIMILARITY_THRESHOLD = _parse_float_env(
    "RAG_RELAXED_SIMILARITY_THRESHOLD",
    default=0.32,
    minimum=0.0,
    maximum=1.0,
)
RAG_MIN_CONTEXT_RESULTS = _parse_positive_int_env(
    "RAG_MIN_CONTEXT_RESULTS",
    default=3,
    minimum=1,
)
RAG_RECENT_FALLBACK_LIMIT = _parse_positive_int_env(
    "RAG_RECENT_FALLBACK_LIMIT",
    default=4,
    minimum=1,
)
RAG_LEXICAL_FALLBACK_ENABLED = _parse_bool_env("RAG_LEXICAL_FALLBACK_ENABLED", default=True)

LEXICAL_FALLBACK_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "that", "this", "from", "what", "should", "would", "could",
    "there", "have", "been", "about", "your", "you", "our", "their", "into", "just", "more",
    "than", "then", "when", "where", "which", "while", "were", "will", "also", "right",
    "today", "now", "next", "need", "want", "help", "please", "show", "tell", "give", "make",
}

EMBEDDING_METADATA_ALLOWLIST: Dict[str, Set[str]] = {
    "default": {
        "agent_type",
        "role",
        "preferences",
        "priority",
        "milestones",
        "summary",
        "source",
    },
    "time_entry": {
        "agent_type",
        "source_action",
        "project_name",
        "description",
        "duration_minutes",
        "billable",
        "linked_goal",
        "focus_score",
        "energy_score",
    },
    "time_anchor": {
        "availability",
        "notifications",
        "integrations",
        "role",
    },
    "user_profile": {
        "role",
        "preferences",
        "mentor",
        "preferredTone",
        "onboarding_completed",
    },
    "planner": {
        "availability",
        "notifications",
        "integrations",
        "preference_profile",
    },
    "goals": {
        "priority",
        "category",
        "milestones",
        "smart_criteria",
        "endDate",
        "whyItMatters",
    },
    "insight": {
        "agent_type",
        "source",
        "checkup_type",
        "checkup_date",
        "summary",
    },
    "habit_snapshot": {
        "agent_type",
        "summary",
        "captured_at",
        "total_habits",
        "total_completion_events",
        "active_days",
        "current_run",
        "longest_run",
        "habit_highlights",
        "daily_completion_digest",
    },
    "system": {
        "last_updated",
    },
}

EMBEDDING_CONTEXT_ALLOWLIST: Dict[str, Set[str]] = {
    "default": {
        "source",
        "source_action",
        "summary",
        "description",
    },
    "time_entry": {
        "source",
        "source_action",
        "project_name",
        "description",
        "task_name",
        "duration_minutes",
        "duration_seconds",
        "billable",
        "linked_goal",
        "focus_score",
        "energy_score",
        "blockers",
        "context_notes",
        "ai_detail",
        "start_time",
        "end_time",
    },
    "time_anchor": {
        "check_in_time",
        "frequency",
        "timezone",
    },
    "planner": {
        "summary",
        "habits",
        "daily_completion_counts",
    },
    "insight": {
        "summary",
        "source",
        "source_action",
    },
    "habit_snapshot": {
        "source",
        "source_action",
        "captured_at",
        "total_habits",
        "total_completion_events",
        "active_days",
        "current_run",
        "longest_run",
        "habit_highlights",
        "daily_completion_digest",
    },
}

EMBEDDING_CHUNK_PROFILE: Dict[str, Dict[str, Any]] = {
    "default": {
        "max_chars": EMBEDDING_MAX_CHARS_PER_CHUNK,
        "overlap": EMBEDDING_CHUNK_OVERLAP_CHARS,
        "max_chunks": EMBEDDING_MAX_CHUNKS_PER_ENTRY,
        "semantic": True,
    },
    "time_anchor": {
        "max_chars": EMBEDDING_MICRO_ENTRY_MAX_CHARS,
        "overlap": 0,
        "max_chunks": 1,
        "semantic": True,
    },
    "time_entry": {
        "max_chars": 360,
        "overlap": 40,
        "max_chunks": 3,
        "semantic": True,
    },
    "user_profile": {
        "max_chars": 440,
        "overlap": 60,
        "max_chunks": 3,
        "semantic": True,
    },
    "planner": {
        "max_chars": 420,
        "overlap": 50,
        "max_chunks": 3,
        "semantic": True,
    },
    "goals": {
        "max_chars": 420,
        "overlap": 50,
        "max_chunks": 3,
        "semantic": True,
    },
    "insight": {
        "max_chars": 380,
        "overlap": 40,
        "max_chunks": 2,
        "semantic": True,
    },
    "habit_snapshot": {
        "max_chars": 360,
        "overlap": 30,
        "max_chunks": 2,
        "semantic": True,
    },
    "system": {
        "max_chars": 460,
        "overlap": 40,
        "max_chunks": 2,
        "semantic": True,
    },
}


class KnowledgeBaseService:
    """Service for managing knowledge base operations and RAG functionality."""
    
    def __init__(self, user_id: str = "single_user"):
        self.user_id = user_id
        self.vector_store = get_vector_store(user_id)
        self.knowledge_db_store = get_knowledge_db_store()
        self.preference_db_store = get_preference_db_store()
        self._user_preferences: Optional[UserPreferences] = None
        self._sync_event_index: Dict[str, str] = {}
        self._sync_event_index_loaded = False
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_loaded = False
        self._embedding_provider_cooldown_until = 0.0

        self._bootstrap_knowledge_db_store()

    def _attach_embedding_to_entry(self, entry: Optional[KnowledgeEntry]) -> Optional[KnowledgeEntry]:
        if not entry:
            return None

        embedding = self.vector_store.get_embedding(entry.entry_id)
        if self._embedding_has_signal(embedding):
            entry.embedding = embedding

        return entry

    def _persist_entry_to_db(self, entry: KnowledgeEntry) -> None:
        if not self.knowledge_db_store or not self.knowledge_db_store.is_available:
            return

        persisted = self.knowledge_db_store.upsert_entry(entry)
        if not persisted:
            logger.warning("Failed to persist knowledge entry to DB: %s", entry.entry_id)

    def _remove_entry_from_db(self, entry_id: str) -> None:
        if not self.knowledge_db_store or not self.knowledge_db_store.is_available:
            return

        self.knowledge_db_store.delete_entry(self.user_id, entry_id)

    def _remove_entries_from_db(self, entry_ids: List[str]) -> None:
        if not self.knowledge_db_store or not self.knowledge_db_store.is_available:
            return

        self.knowledge_db_store.delete_entries(self.user_id, entry_ids)

    def _clear_entries_from_db(self) -> None:
        if not self.knowledge_db_store or not self.knowledge_db_store.is_available:
            return

        self.knowledge_db_store.clear_user_entries(self.user_id)

    def _bootstrap_knowledge_db_store(self) -> None:
        """Reconcile SQL store from persisted vector metadata for this user."""
        if not self.knowledge_db_store or not self.knowledge_db_store.is_available:
            return

        vector_entries = [
            entry
            for entry in self.vector_store.get_all_entries()
            if getattr(entry, "user_id", self.user_id) == self.user_id
        ]

        if not vector_entries:
            return

        existing_count = self.knowledge_db_store.count_user_entries(self.user_id)
        if existing_count >= len(vector_entries):
            return

        persisted_count = 0
        for entry in vector_entries:
            if self.knowledge_db_store.upsert_entry(entry):
                persisted_count += 1

        logger.info(
            "Knowledge DB reconciliation persisted %d/%d entries for user %s (db_before=%d)",
            persisted_count,
            len(vector_entries),
            self.user_id,
            existing_count,
        )

    def _merge_preference_payload(
        self,
        base_payload: Dict[str, Any],
        override_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(base_payload)

        for section, value in (override_payload or {}).items():
            if section == "user_id":
                continue

            if (
                section in merged
                and isinstance(merged[section], dict)
                and isinstance(value, dict)
            ):
                merged[section].update(value)
            else:
                merged[section] = value

        merged["user_id"] = self.user_id
        return merged

    def _load_preferences_from_db_store(self) -> Optional[UserPreferences]:
        if not self.preference_db_store or not self.preference_db_store.is_available:
            return None

        stored_payload = self.preference_db_store.load_preferences(self.user_id)
        if not stored_payload:
            return None

        defaults = UserPreferences(user_id=self.user_id).model_dump()
        merged = self._merge_preference_payload(defaults, stored_payload)

        try:
            return UserPreferences(**merged)
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to hydrate user preferences from preference DB store: %s", exc)
            return None

    def _persist_preferences_to_db_store(self, preferences: UserPreferences) -> bool:
        if not self.preference_db_store or not self.preference_db_store.is_available:
            return False

        payload = preferences.model_dump()
        payload["user_id"] = self.user_id
        return self.preference_db_store.upsert_preferences(self.user_id, payload)

    def _truncate_for_log(self, value: Any, limit: int = 140) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[:limit - 3]}..."

    def _prepare_embedding_log_text(self, text: str) -> str:
        normalized = str(text or "")
        if EMBEDDING_LOG_FULL_TEXT:
            return normalized

        return self._truncate_for_log(normalized, limit=EMBEDDING_LOG_MAX_TEXT_CHARS)

    def _log_embedding_payload(
        self,
        *,
        action: str,
        embedding_key: str,
        embedding_text: str,
        chunks: List[str],
        entry_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        if not EMBEDDING_LOG_ENABLED:
            return

        prepared_text = self._prepare_embedding_log_text(embedding_text)
        embedding_logger.info(
            "EMBEDDING_INPUT action=%s user=%s entry_id=%s category=%s key=%s chars=%d chunks=%d text=%s",
            action,
            self.user_id,
            entry_id or "pending",
            category or "unknown",
            embedding_key,
            len(embedding_text),
            len(chunks),
            prepared_text,
        )

        for index, chunk in enumerate(chunks, start=1):
            prepared_chunk = self._prepare_embedding_log_text(chunk)
            chunk_key = self._build_chunk_embedding_cache_key(chunk)
            embedding_logger.info(
                "EMBEDDING_CHUNK action=%s user=%s entry_id=%s chunk=%d/%d key=%s chars=%d text=%s",
                action,
                self.user_id,
                entry_id or "pending",
                index,
                len(chunks),
                chunk_key,
                len(chunk),
                prepared_chunk,
            )

    def _log_query_embedding_input(self, query_text: str, query_key: str) -> None:
        if not EMBEDDING_LOG_ENABLED:
            return

        prepared_text = self._prepare_embedding_log_text(query_text)
        embedding_logger.info(
            "EMBEDDING_QUERY_INPUT user=%s key=%s chars=%d text=%s",
            self.user_id,
            query_key,
            len(query_text),
            prepared_text,
        )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured LLM provider."""
        if monotonic() < self._embedding_provider_cooldown_until:
            raise RuntimeError("Embedding provider is temporarily unavailable due to recent failures")

        try:
            from ..llm import service as llm_service_module
        except ImportError as e:
            raise RuntimeError(f"Embedding dependencies are unavailable: {e}") from e

        llm_service = llm_service_module._llm_service
        if not llm_service or not llm_service._initialized:
            raise RuntimeError("LLM service is not initialized for embedding generation")

        try:
            request = EmbeddingRequest(text=text)
            response = await llm_service.generate_embedding(request)
        except Exception as e:
            error_text = str(e).lower()
            if (
                "no healthy providers available" in error_text
                or "provider_unavailable" in error_text
                or "connection error" in error_text
                or "timed out" in error_text
                or "timeout" in error_text
                or "api connection" in error_text
            ):
                self._embedding_provider_cooldown_until = monotonic() + EMBEDDING_PROVIDER_COOLDOWN_SECONDS
            raise RuntimeError(f"Embedding generation failed: {e}") from e

        embedding = list(response.embedding or [])
        if not self._embedding_has_signal(embedding):
            raise RuntimeError("Embedding provider returned a zero-signal vector")

        self._embedding_provider_cooldown_until = 0.0
        return embedding

    def _normalize_embedding_label(self, value: Any) -> str:
        if hasattr(value, "value"):
            value = value.value

        return " ".join(str(value or "").split()).strip().lower()

    def _stringify_embedding_value(
        self,
        value: Any,
        max_items: int = 8,
        max_chars: int = 220,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, str):
            normalized = " ".join(value.split()).strip()
            if len(normalized) <= max_chars:
                return normalized
            return f"{normalized[:max_chars - 3]}..."

        if isinstance(value, list):
            normalized_items = []
            for item in value[:max_items]:
                item_text = self._stringify_embedding_value(item, max_items=max_items, max_chars=max_chars)
                if item_text:
                    normalized_items.append(item_text)
            return " | ".join(normalized_items)

        if isinstance(value, dict):
            normalized_pairs = []
            for key in sorted(value.keys())[:max_items]:
                item_text = self._stringify_embedding_value(value.get(key), max_items=max_items, max_chars=max_chars)
                if item_text:
                    normalized_pairs.append(f"{key}={item_text}")
            return " | ".join(normalized_pairs)

        return self._stringify_embedding_value(str(value), max_items=max_items, max_chars=max_chars)

    def _looks_like_time_anchor(
        self,
        *,
        normalized_title: str,
        normalized_content: str,
        normalized_tags: List[str],
        metadata_payload: Dict[str, Any],
    ) -> bool:
        hint_text = " ".join([normalized_title, normalized_content, " ".join(normalized_tags)]).strip().lower()
        has_anchor_keyword = bool(re.search(r"\b(wake\s*up|wakeup|wake|sleep|bedtime|check[- ]?in)\b", hint_text))
        has_time_token = bool(re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", hint_text)) or bool(
            re.search(r"\b(1[0-2]|0?[1-9])\s?(am|pm)\b", hint_text)
        )

        availability_payload = metadata_payload.get("availability") if isinstance(metadata_payload.get("availability"), dict) else {}
        check_in_payload = availability_payload.get("checkIn") if isinstance(availability_payload.get("checkIn"), dict) else {}
        has_schedule_signal = bool(check_in_payload.get("preferredTime") or availability_payload.get("workHours"))

        return (has_anchor_keyword and (has_time_token or has_schedule_signal)) or has_schedule_signal

    def _resolve_embedding_strategy_category(
        self,
        category: Optional[str],
        *,
        title: str = "",
        content: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_type: Optional[Any] = None,
        entry_sub_type: Optional[Any] = None,
    ) -> str:
        metadata_payload = metadata if isinstance(metadata, dict) else {}
        normalized_category = self._normalize_embedding_label(category or "uncategorized")
        normalized_entry_type = self._normalize_embedding_label(entry_type or "unknown")
        normalized_entry_sub_type = self._normalize_embedding_label(entry_sub_type or "unknown")
        normalized_title = " ".join(str(title or "").split()).strip().lower()
        normalized_content = str(content or "").strip().lower()
        normalized_tags = [" ".join(str(tag).split()).strip().lower() for tag in (tags or []) if str(tag).strip()]

        context_payload = metadata_payload.get("context") if isinstance(metadata_payload.get("context"), dict) else {}
        source = str(context_payload.get("source", "")).strip().lower()
        source_action = str(context_payload.get("source_action", "")).strip().lower()

        if (
            normalized_category == "time_entry"
            or source == "alterego_timetracker"
            or "time_entry" in source_action
            or context_payload.get("time_entry_id") is not None
        ):
            return "time_entry"

        if normalized_entry_sub_type == "user profile" or normalized_category in {"user_profile", "profile"}:
            return "user_profile"

        if normalized_entry_sub_type == "schedule" or normalized_category in {"planner", "schedule"}:
            if self._looks_like_time_anchor(
                normalized_title=normalized_title,
                normalized_content=normalized_content,
                normalized_tags=normalized_tags,
                metadata_payload=metadata_payload,
            ):
                return "time_anchor"
            return "planner"

        if normalized_entry_sub_type == "goal" or normalized_category in {"goal", "goals"}:
            return "goals"

        if normalized_entry_type == "insight" or normalized_entry_sub_type.endswith("insight"):
            return "insight"

        if normalized_category in {"habit_snapshot", "habit_progress"}:
            return "habit_snapshot"

        if normalized_category in {"system", "config", "configuration"}:
            return "system"

        return normalized_category or "default"

    def _select_embedding_chunk_profile(
        self,
        strategy_category: str,
    ) -> Dict[str, Any]:
        return EMBEDDING_CHUNK_PROFILE.get(strategy_category, EMBEDDING_CHUNK_PROFILE["default"])

    def _extract_system_snapshot_facts(
        self,
        content: str,
    ) -> List[str]:
        if not content:
            return []

        try:
            parsed = json.loads(content)
        except Exception:
            return []

        if not isinstance(parsed, dict):
            return []

        facts: List[str] = []

        productivity = parsed.get("productivity") if isinstance(parsed.get("productivity"), dict) else {}
        if productivity.get("work_hours"):
            facts.append(f"Work hours: {self._stringify_embedding_value(productivity.get('work_hours'))}")

        health = parsed.get("health") if isinstance(parsed.get("health"), dict) else {}
        if health.get("dietary_preferences"):
            facts.append(f"Dietary preferences: {self._stringify_embedding_value(health.get('dietary_preferences'))}")

        finance = parsed.get("finance") if isinstance(parsed.get("finance"), dict) else {}
        if finance.get("expense_tracking"):
            facts.append(f"Expense tracking cadence: {self._stringify_embedding_value(finance.get('expense_tracking'))}")

        general = parsed.get("general") if isinstance(parsed.get("general"), dict) else {}
        if general.get("timezone"):
            facts.append(f"Timezone: {self._stringify_embedding_value(general.get('timezone'))}")

        return facts

    def _build_embedding_facts(
        self,
        *,
        strategy_category: str,
        normalized_title: str,
        normalized_content: str,
        normalized_tags: List[str],
        metadata_payload: Dict[str, Any],
    ) -> List[str]:
        context_payload = metadata_payload.get("context") if isinstance(metadata_payload.get("context"), dict) else {}

        def add_fact(bucket: List[str], label: str, value: Any) -> None:
            normalized_value = self._stringify_embedding_value(value)
            if not normalized_value:
                return
            bucket.append(f"{label}: {normalized_value}")

        facts: List[str] = []

        if strategy_category == "time_anchor":
            time_matches = re.findall(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", f"{normalized_title} {normalized_content}")
            if time_matches:
                hour, minute = time_matches[0]
                add_fact(facts, "Anchor time", f"{hour}:{minute}")

            availability_payload = metadata_payload.get("availability") if isinstance(metadata_payload.get("availability"), dict) else {}
            check_in_payload = availability_payload.get("checkIn") if isinstance(availability_payload.get("checkIn"), dict) else {}
            add_fact(facts, "Preferred check-in", check_in_payload.get("preferredTime"))
            add_fact(facts, "Check-in frequency", check_in_payload.get("frequency"))
            add_fact(facts, "Timezone", availability_payload.get("timezone"))
            add_fact(facts, "Routine", normalized_title or normalized_content)
            return facts

        if strategy_category == "time_entry":
            add_fact(facts, "Task", context_payload.get("description") or context_payload.get("task_name") or normalized_title)
            add_fact(facts, "Project", context_payload.get("project_name"))

            duration_minutes = context_payload.get("duration_minutes")
            if duration_minutes is None and context_payload.get("duration_seconds") is not None:
                try:
                    duration_minutes = round(float(context_payload.get("duration_seconds")) / 60.0, 1)
                except (TypeError, ValueError):
                    duration_minutes = None
            add_fact(facts, "Duration minutes", duration_minutes)

            add_fact(facts, "Time window", f"{context_payload.get('start_time')} -> {context_payload.get('end_time')}")
            add_fact(facts, "Billable", context_payload.get("billable"))
            add_fact(facts, "Linked goal", context_payload.get("linked_goal"))
            add_fact(facts, "Focus score", context_payload.get("focus_score"))
            add_fact(facts, "Energy score", context_payload.get("energy_score"))
            add_fact(facts, "Blockers", context_payload.get("blockers"))
            return facts

        if strategy_category == "user_profile":
            add_fact(facts, "Role", metadata_payload.get("role"))
            add_fact(facts, "Preferences", metadata_payload.get("preferences"))

            mentor_payload = metadata_payload.get("mentor") if isinstance(metadata_payload.get("mentor"), dict) else {}
            add_fact(facts, "Mentor", mentor_payload.get("name") or mentor_payload)
            add_fact(facts, "Mentor style", mentor_payload.get("style"))
            add_fact(facts, "Preferred tone", metadata_payload.get("preferredTone"))
            return facts

        if strategy_category == "planner":
            availability_payload = metadata_payload.get("availability") if isinstance(metadata_payload.get("availability"), dict) else {}
            work_hours_payload = availability_payload.get("workHours") if isinstance(availability_payload.get("workHours"), dict) else {}
            check_in_payload = availability_payload.get("checkIn") if isinstance(availability_payload.get("checkIn"), dict) else {}
            add_fact(
                facts,
                "Work hours",
                f"{work_hours_payload.get('start')} - {work_hours_payload.get('end')}" if work_hours_payload else None,
            )
            add_fact(facts, "Timezone", availability_payload.get("timezone"))
            add_fact(facts, "Check-in time", check_in_payload.get("preferredTime"))
            add_fact(facts, "Check-in frequency", check_in_payload.get("frequency"))
            add_fact(facts, "Integrations", metadata_payload.get("integrations"))
            return facts

        if strategy_category == "goals":
            add_fact(facts, "Goal", normalized_title)
            add_fact(facts, "Category", metadata_payload.get("category"))
            add_fact(facts, "Priority", metadata_payload.get("priority"))
            add_fact(facts, "Milestones", metadata_payload.get("milestones"))
            add_fact(facts, "Target date", metadata_payload.get("endDate"))
            add_fact(facts, "Why it matters", metadata_payload.get("whyItMatters"))
            return facts

        if strategy_category == "insight":
            add_fact(facts, "Insight", normalized_title)
            add_fact(facts, "Observation", normalized_content)
            add_fact(facts, "Source", metadata_payload.get("source"))
            return facts

        if strategy_category == "habit_snapshot":
            add_fact(facts, "Snapshot", normalized_title or "Habit progress snapshot")
            add_fact(facts, "Captured at", context_payload.get("captured_at") or metadata_payload.get("captured_at"))
            add_fact(facts, "Total habits", context_payload.get("total_habits") or metadata_payload.get("total_habits"))
            add_fact(
                facts,
                "Completion events",
                context_payload.get("total_completion_events") or metadata_payload.get("total_completion_events"),
            )
            add_fact(facts, "Active days", context_payload.get("active_days") or metadata_payload.get("active_days"))
            add_fact(facts, "Current run", context_payload.get("current_run") or metadata_payload.get("current_run"))
            add_fact(facts, "Longest run", context_payload.get("longest_run") or metadata_payload.get("longest_run"))
            add_fact(
                facts,
                "Highlights",
                context_payload.get("habit_highlights") or metadata_payload.get("habit_highlights"),
            )
            add_fact(
                facts,
                "Daily trend",
                context_payload.get("daily_completion_digest") or metadata_payload.get("daily_completion_digest"),
            )
            return facts

        if strategy_category == "system":
            distilled_facts = self._extract_system_snapshot_facts(normalized_content)
            if distilled_facts:
                return distilled_facts
            add_fact(facts, "System preference snapshot", normalized_title or normalized_content)
            return facts

        add_fact(facts, "Summary", normalized_title)
        add_fact(facts, "Details", normalized_content)
        add_fact(facts, "Tags", normalized_tags)
        return facts

    def _extract_embedding_metadata_signals(
        self,
        category: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[str]:
        metadata_payload = metadata if isinstance(metadata, dict) else {}
        normalized_category = self._normalize_embedding_label(category)
        context_payload = metadata_payload.get("context") if isinstance(metadata_payload.get("context"), dict) else {}

        metadata_keys = EMBEDDING_METADATA_ALLOWLIST.get(normalized_category, EMBEDDING_METADATA_ALLOWLIST["default"])
        context_keys = EMBEDDING_CONTEXT_ALLOWLIST.get(normalized_category, EMBEDDING_CONTEXT_ALLOWLIST["default"])

        signals: Dict[str, str] = {}

        def add_signal(label: str, value: Any) -> None:
            if label in signals:
                return

            text = self._stringify_embedding_value(value)
            if not text:
                return

            signals[label] = text

        for key in sorted(metadata_keys):
            add_signal(key, metadata_payload.get(key))

        for key in sorted(context_keys):
            add_signal(key, context_payload.get(key))

        if normalized_category == "time_entry" and "duration_minutes" not in signals:
            add_signal("duration_minutes", context_payload.get("duration_seconds"))

        return [f"{label}: {value}" for label, value in sorted(signals.items())]

    def _build_embedding_document(
        self,
        *,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_type: Optional[Any] = None,
        entry_sub_type: Optional[Any] = None,
    ) -> Tuple[str, List[str], str]:
        metadata_payload = metadata if isinstance(metadata, dict) else {}
        normalized_title = " ".join(str(title or "").split()).strip()
        normalized_content = str(content or "").strip()
        normalized_tags = [" ".join(str(tag).split()).strip() for tag in (tags or []) if str(tag).strip()]
        normalized_entry_type = self._normalize_embedding_label(entry_type or "unknown")
        normalized_entry_sub_type = self._normalize_embedding_label(entry_sub_type or "unknown")

        strategy_category = self._resolve_embedding_strategy_category(
            category,
            title=normalized_title,
            content=normalized_content,
            tags=normalized_tags,
            metadata=metadata_payload,
            entry_type=entry_type,
            entry_sub_type=entry_sub_type,
        )

        profile = self._select_embedding_chunk_profile(strategy_category)
        semantic_sections: List[str] = []

        summary_tokens = [token for token in [normalized_title, strategy_category.replace("_", " ")] if token]
        semantic_sections.append("Summary: " + " - ".join(summary_tokens) if summary_tokens else "Summary: knowledge entry")

        facts = self._build_embedding_facts(
            strategy_category=strategy_category,
            normalized_title=normalized_title,
            normalized_content=normalized_content,
            normalized_tags=normalized_tags,
            metadata_payload=metadata_payload,
        )
        semantic_sections.extend(facts)

        metadata_signals = self._extract_embedding_metadata_signals(strategy_category, metadata_payload)
        semantic_sections.extend(metadata_signals)

        if normalized_content and strategy_category not in {"system", "time_anchor"}:
            content_cap = int(profile.get("max_chars", EMBEDDING_SEMANTIC_SECTION_MAX_CHARS))
            content_excerpt = self._stringify_embedding_value(normalized_content, max_chars=max(content_cap, 220))
            if content_excerpt:
                semantic_sections.append(f"Content excerpt: {content_excerpt}")

        deduped_sections: List[str] = []
        seen_sections: Set[str] = set()
        for section in semantic_sections:
            normalized_section = " ".join(str(section or "").split()).strip()
            if not normalized_section or normalized_section in seen_sections:
                continue
            seen_sections.add(normalized_section)
            deduped_sections.append(normalized_section)

        document_parts = [
            f"entry_type: {normalized_entry_type}",
            f"entry_sub_type: {normalized_entry_sub_type}",
            f"category: {strategy_category}",
            f"title: {normalized_title}",
        ]
        if normalized_tags:
            document_parts.append(f"tags: {', '.join(normalized_tags)}")
        if deduped_sections:
            document_parts.append("semantic_facts:\n" + "\n".join(f"- {section}" for section in deduped_sections))

        embedding_text = "\n\n".join(part for part in document_parts if part).strip()
        return embedding_text, deduped_sections, strategy_category

    def _build_embedding_text(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_type: Optional[Any] = None,
        entry_sub_type: Optional[Any] = None,
    ) -> str:
        embedding_text, _, _ = self._build_embedding_document(
            title=title,
            content=content,
            tags=tags,
            category=category,
            metadata=metadata,
            entry_type=entry_type,
            entry_sub_type=entry_sub_type,
        )
        return embedding_text

    def _build_embedding_cache_key(self, embedding_text: str) -> str:
        normalized_text = " ".join(str(embedding_text or "").split()).strip().lower() or "empty"
        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    def _build_chunk_embedding_cache_key(self, chunk_text: str) -> str:
        chunk_hash = self._build_embedding_cache_key(chunk_text)
        return f"chunk::{chunk_hash}"

    def _chunk_embedding_text(
        self,
        embedding_text: str,
        *,
        category: Optional[str] = None,
        entry_type: Optional[Any] = None,
        semantic_sections: Optional[List[str]] = None,
    ) -> List[str]:
        normalized_text = " ".join(str(embedding_text or "").split()).strip()
        if not normalized_text:
            return ["empty"]

        strategy_category = self._resolve_embedding_strategy_category(
            category,
            entry_type=entry_type,
        )
        profile = self._select_embedding_chunk_profile(strategy_category)
        max_chars = int(profile.get("max_chars", EMBEDDING_MAX_CHARS_PER_CHUNK))
        overlap_chars = int(profile.get("overlap", EMBEDDING_CHUNK_OVERLAP_CHARS))
        max_chunks = int(profile.get("max_chunks", EMBEDDING_MAX_CHUNKS_PER_ENTRY))
        use_semantic = bool(profile.get("semantic", True))

        if len(normalized_text) <= max_chars:
            return [normalized_text]

        if use_semantic and semantic_sections:
            semantic_units: List[str] = []
            for section in semantic_sections:
                normalized_section = " ".join(str(section or "").split()).strip()
                if not normalized_section:
                    continue

                if len(normalized_section) <= EMBEDDING_SEMANTIC_SECTION_MAX_CHARS:
                    semantic_units.append(normalized_section)
                    continue

                sentence_parts = re.split(r"(?<=[.!?])\s+", normalized_section)
                buffered = ""
                for part in sentence_parts:
                    part = " ".join(part.split()).strip()
                    if not part:
                        continue
                    candidate = f"{buffered} {part}".strip() if buffered else part
                    if len(candidate) <= EMBEDDING_SEMANTIC_SECTION_MAX_CHARS:
                        buffered = candidate
                        continue

                    if buffered:
                        semantic_units.append(buffered)
                        buffered = ""

                    if len(part) <= EMBEDDING_SEMANTIC_SECTION_MAX_CHARS:
                        buffered = part
                        continue

                    words = part.split()
                    word_buffer = ""
                    for word in words:
                        word_candidate = f"{word_buffer} {word}".strip() if word_buffer else word
                        if len(word_candidate) <= EMBEDDING_SEMANTIC_SECTION_MAX_CHARS:
                            word_buffer = word_candidate
                        else:
                            if word_buffer:
                                semantic_units.append(word_buffer)
                            word_buffer = word
                    if word_buffer:
                        buffered = word_buffer

                if buffered:
                    semantic_units.append(buffered)

            chunks: List[str] = []
            current_chunk = ""
            section_index = 0

            while section_index < len(semantic_units) and len(chunks) < max_chunks:
                unit = semantic_units[section_index]
                candidate = f"{current_chunk} {unit}".strip() if current_chunk else unit
                if len(candidate) <= max_chars:
                    current_chunk = candidate
                    section_index += 1
                    continue

                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    continue

                chunks.append(unit[:max_chars].strip())
                section_index += 1

            if current_chunk and len(chunks) < max_chunks:
                chunks.append(current_chunk)

            if section_index < len(semantic_units) and chunks:
                tail = " ".join(semantic_units[section_index:]).strip()
                if tail:
                    merged_tail = f"{chunks[-1]} {tail}".strip()
                    chunks[-1] = merged_tail[:max_chars].strip()

            if chunks:
                return chunks[:max_chunks]

        chunks: List[str] = []
        cursor = 0

        while cursor < len(normalized_text) and len(chunks) < max_chunks:
            max_end = min(len(normalized_text), cursor + max_chars)
            split_end = max_end

            if max_end < len(normalized_text):
                preferred_break = normalized_text.rfind(". ", cursor + max_chars // 2, max_end)
                if preferred_break == -1:
                    preferred_break = normalized_text.rfind(" ", cursor + max_chars // 2, max_end)
                if preferred_break > cursor:
                    split_end = preferred_break + 1

            chunk = normalized_text[cursor:split_end].strip()
            if chunk:
                chunks.append(chunk)

            if split_end >= len(normalized_text):
                break

            next_cursor = max(cursor + 1, split_end - overlap_chars)
            if next_cursor <= cursor:
                next_cursor = split_end
            cursor = next_cursor

        if cursor < len(normalized_text) and len(chunks) >= max_chunks:
            tail = normalized_text[cursor:].strip()
            if tail:
                chunks[-1] = f"{chunks[-1]} {tail}".strip()

        return chunks or [normalized_text]

    def _fit_embedding_dimension(self, embedding: List[float]) -> List[float]:
        target_dimension = self.vector_store.dimension
        if len(embedding) == target_dimension:
            return list(embedding)

        if len(embedding) > target_dimension:
            return list(embedding[:target_dimension])

        return list(embedding) + [0.0] * (target_dimension - len(embedding))

    async def _resolve_query_embedding(self, query_text: str) -> List[float]:
        query_key = f"query::{self._build_embedding_cache_key(query_text)}"
        self._log_query_embedding_input(query_text, query_key)

        cached_embedding = self._embedding_cache.get(query_key)
        if cached_embedding:
            if EMBEDDING_LOG_ENABLED:
                embedding_logger.info(
                    "EMBEDDING_QUERY_CACHE_HIT user=%s key=%s",
                    self.user_id,
                    query_key,
                )
            return list(cached_embedding)

        generated_embedding = await self._generate_embedding(query_text)
        self._cache_embedding_value(query_key, generated_embedding)
        if EMBEDDING_LOG_ENABLED:
            embedding_logger.info(
                "EMBEDDING_QUERY_GENERATED user=%s key=%s dimension=%d",
                self.user_id,
                query_key,
                len(generated_embedding),
            )
        return generated_embedding

    async def _generate_embedding_for_text(
        self,
        embedding_text: str,
        *,
        category: Optional[str] = None,
        entry_type: Optional[Any] = None,
        semantic_sections: Optional[List[str]] = None,
    ) -> List[float]:
        chunks = self._chunk_embedding_text(
            embedding_text,
            category=category,
            entry_type=entry_type,
            semantic_sections=semantic_sections,
        )
        if len(chunks) == 1:
            return await self._generate_embedding(chunks[0])

        weighted_embedding = [0.0] * self.vector_store.dimension
        total_weight = 0.0

        for chunk in chunks:
            chunk_key = self._build_chunk_embedding_cache_key(chunk)
            chunk_embedding = self._embedding_cache.get(chunk_key)
            if chunk_embedding:
                resolved_chunk_embedding = list(chunk_embedding)
            else:
                resolved_chunk_embedding = await self._generate_embedding(chunk)
                self._cache_embedding_value(chunk_key, resolved_chunk_embedding)

            normalized_chunk_embedding = self._fit_embedding_dimension(resolved_chunk_embedding)
            chunk_weight = float(max(1, len(chunk.split())))
            total_weight += chunk_weight

            for idx, value in enumerate(normalized_chunk_embedding):
                weighted_embedding[idx] += value * chunk_weight

        if total_weight <= 0:
            return await self._generate_embedding(embedding_text)

        averaged_embedding = [value / total_weight for value in weighted_embedding]
        norm = sum(value * value for value in averaged_embedding) ** 0.5
        if norm > 0:
            averaged_embedding = [value / norm for value in averaged_embedding]

        return averaged_embedding

    def _extract_embedding_cache_key(self, entry: KnowledgeEntry) -> str:
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        cached_key = str(metadata.get(EMBEDDING_CACHE_KEY_FIELD, "")).strip()
        if cached_key:
            return cached_key

        fallback_text = self._build_embedding_text(
            title=entry.title,
            content=entry.content,
            tags=entry.tags,
            category=entry.category,
            metadata=entry.metadata,
            entry_type=entry.entry_type,
            entry_sub_type=entry.entry_sub_type,
        )
        return self._build_embedding_cache_key(fallback_text)

    def _cache_embedding_value(self, embedding_key: str, embedding: Optional[List[float]]) -> None:
        if not embedding_key or not self._embedding_has_signal(embedding):
            return
        self._embedding_cache[embedding_key] = list(embedding)

    def _cache_entry_embedding(self, entry: Optional[KnowledgeEntry]) -> None:
        if not entry or not entry.embedding:
            return
        self._cache_embedding_value(self._extract_embedding_cache_key(entry), entry.embedding)

    async def _ensure_embedding_cache_loaded(self) -> None:
        if self._embedding_cache_loaded:
            return

        try:
            existing_entries = await self.get_all_entries()
            for entry in existing_entries:
                self._cache_entry_embedding(entry)
        finally:
            self._embedding_cache_loaded = True

    def _preserve_embeddings_for_entry_ids(self, entry_ids: List[str]) -> None:
        for entry_id in entry_ids:
            entry = self.vector_store.get_entry(entry_id)
            self._cache_entry_embedding(entry)

    def _embedding_has_signal(self, embedding: Optional[List[float]]) -> bool:
        if not embedding:
            return False

        return any(abs(float(value)) > 1e-9 for value in embedding)

    async def _resolve_embedding(
        self,
        embedding_text: str,
        embedding_key: str,
        existing_entry: Optional[KnowledgeEntry] = None,
        category: Optional[str] = None,
        entry_type: Optional[Any] = None,
        semantic_sections: Optional[List[str]] = None,
    ) -> List[float]:
        await self._ensure_embedding_cache_loaded()

        if existing_entry and self._embedding_has_signal(existing_entry.embedding):
            existing_key = self._extract_embedding_cache_key(existing_entry)
            if existing_key == embedding_key:
                resolved = list(existing_entry.embedding)
                self._cache_embedding_value(embedding_key, resolved)
                return resolved

        cached_embedding = self._embedding_cache.get(embedding_key)
        if self._embedding_has_signal(cached_embedding):
            return list(cached_embedding)

        generated_embedding = await self._generate_embedding_for_text(
            embedding_text,
            category=category,
            entry_type=entry_type,
            semantic_sections=semantic_sections,
        )
        if not self._embedding_has_signal(generated_embedding):
            raise RuntimeError("Generated embedding has no semantic signal")
        self._cache_embedding_value(embedding_key, generated_embedding)
        return generated_embedding

    async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant knowledge entries based on a query."""
        try:
            # Generate embedding for the query
            query_embedding = await self._resolve_query_embedding(query)
            
            # Search the vector store
            results = self.vector_store.search(query_embedding, limit)
            
            # Format results
            formatted_results = []
            for result in results:
                entry = result.entry if hasattr(result, "entry") else None
                formatted_results.append({
                    "content": getattr(entry, "content", ""),
                    "score": float(getattr(result, "similarity_score", 0.0)),
                    "metadata": getattr(entry, "metadata", {}) if entry else {},
                })
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Knowledge search failed: {e}")
            # Return empty results for graceful degradation
            return []
    
    async def create_entry(self, 
                          entry_type: KnowledgeEntryType,
                          entry_sub_type:KnowledgeEntrySubType,
                          category: str,
                          title: str,
                          content: str,
                          metadata: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> KnowledgeEntry:
        """
        Create a new knowledge base entry.
        
        Args:
            entry_type: Type of the entry
            category: Category of the entry
            title: Human-readable title
            content: The actual content
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            The created knowledge entry
        """
        try:
            metadata_payload = dict(metadata or {})

            embedding_text, semantic_sections, strategy_category = self._build_embedding_document(
                title=title,
                content=content,
                tags=tags,
                category=category,
                metadata=metadata_payload,
                entry_type=entry_type,
                entry_sub_type=entry_sub_type,
            )
            embedding_key = self._build_embedding_cache_key(embedding_text)
            metadata_payload[EMBEDDING_CACHE_KEY_FIELD] = embedding_key

            entry_chunks = self._chunk_embedding_text(
                embedding_text,
                category=strategy_category,
                entry_type=entry_type,
                semantic_sections=semantic_sections,
            )
            self._log_embedding_payload(
                action="create",
                embedding_key=embedding_key,
                embedding_text=embedding_text,
                chunks=entry_chunks,
                category=strategy_category,
            )

            embedding = await self._resolve_embedding(
                embedding_text,
                embedding_key,
                category=strategy_category,
                entry_type=entry_type,
                semantic_sections=semantic_sections,
            )

            # Generate unique ID
            entry_id = str(uuid.uuid4())
            
            # Create entry
            entry = KnowledgeEntry(
                entry_id=entry_id,
                user_id=self.user_id,
                entry_type=entry_type,
                category=category,
                entry_sub_type=entry_sub_type,
                title=title,
                content=content,
                metadata=metadata_payload,
                tags=tags or []
            )
            
            # Add to vector store
            self.vector_store.add_entry(entry, embedding)
            self._index_sync_event_key(entry)
            self._cache_entry_embedding(entry)
            self._persist_entry_to_db(entry)
            
            logger.info(f"Created knowledge entry: {entry_id}")
            return entry
        except Exception as e:
            logger.error(f"Failed to create knowledge entry: {e}")
            raise
    
    async def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Retrieve a knowledge entry by ID.
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            The knowledge entry if found, None otherwise
        """
        try:
            if self.knowledge_db_store and self.knowledge_db_store.is_available:
                db_entry = self.knowledge_db_store.get_entry(self.user_id, entry_id)
                if db_entry:
                    return self._attach_embedding_to_entry(db_entry)

            return self.vector_store.get_entry(entry_id)
        except Exception as e:
            logger.error(f"Failed to get knowledge entry {entry_id}: {e}")
            return None
    
    async def update_entry(self, 
                          entry_id: str,
                          title: Optional[str] = None,
                          content: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> Optional[KnowledgeEntry]:
        """
        Update an existing knowledge entry.
        
        Args:
            entry_id: ID of the entry to update
            title: New title (optional)
            content: New content (optional)
            metadata: New metadata (optional)
            tags: New tags (optional)
            
        Returns:
            The updated entry if successful, None otherwise
        """
        try:
            # Get existing entry
            existing_entry = self.vector_store.get_entry(entry_id)
            if not existing_entry:
                logger.warning(f"Entry {entry_id} not found for update")
                return None
            
            # Update fields
            updated_entry = existing_entry.model_copy()
            if not isinstance(updated_entry.metadata, dict):
                updated_entry.metadata = {}

            if title is not None:
                updated_entry.title = title
            if content is not None:
                updated_entry.content = content
            if metadata is not None:
                updated_entry.metadata.update(metadata)
            if tags is not None:
                updated_entry.tags = tags

            embedding_text, semantic_sections, strategy_category = self._build_embedding_document(
                title=updated_entry.title,
                content=updated_entry.content,
                tags=updated_entry.tags,
                category=updated_entry.category,
                metadata=updated_entry.metadata,
                entry_type=updated_entry.entry_type,
                entry_sub_type=updated_entry.entry_sub_type,
            )
            embedding_key = self._build_embedding_cache_key(embedding_text)
            updated_entry.metadata[EMBEDDING_CACHE_KEY_FIELD] = embedding_key

            entry_chunks = self._chunk_embedding_text(
                embedding_text,
                category=strategy_category,
                entry_type=updated_entry.entry_type,
                semantic_sections=semantic_sections,
            )
            self._log_embedding_payload(
                action="update",
                embedding_key=embedding_key,
                embedding_text=embedding_text,
                chunks=entry_chunks,
                entry_id=updated_entry.entry_id,
                category=strategy_category,
            )
            
            updated_entry.updated_at = datetime.utcnow()
            
            embedding = await self._resolve_embedding(
                embedding_text,
                embedding_key,
                existing_entry=existing_entry,
                category=strategy_category,
                entry_type=updated_entry.entry_type,
                semantic_sections=semantic_sections,
            )
            
            # Update in vector store
            self.vector_store.update_entry(updated_entry, embedding)
            self._index_sync_event_key(updated_entry)
            self._cache_entry_embedding(updated_entry)
            self._persist_entry_to_db(updated_entry)
            
            logger.info(f"Updated knowledge entry: {entry_id}")
            return updated_entry
        except Exception as e:
            logger.error(f"Failed to update knowledge entry {entry_id}: {e}")
            return None
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self._preserve_embeddings_for_entry_ids([entry_id])
            self._remove_indexed_sync_event_for_entry(entry_id)
            success = self.vector_store.remove_entry(entry_id)
            if success:
                logger.info(f"Deleted knowledge entry: {entry_id}")
            else:
                logger.warning(f"Entry {entry_id} not found for deletion")

            self._remove_entry_from_db(entry_id)
            return success
        except Exception as e:
            logger.error(f"Failed to delete knowledge entry {entry_id}: {e}")
            return False

    async def delete_entries(self, entry_ids: List[str]) -> int:
        """Delete multiple knowledge entries in a single vector index rebuild pass."""
        try:
            normalized_ids = [entry_id for entry_id in entry_ids if entry_id]
            if not normalized_ids:
                return 0

            self._preserve_embeddings_for_entry_ids(normalized_ids)

            for entry_id in normalized_ids:
                self._remove_indexed_sync_event_for_entry(entry_id)

            removed = self.vector_store.remove_entries(normalized_ids)
            self._remove_entries_from_db(normalized_ids)
            if removed > 0:
                logger.info("Deleted %d knowledge entries in bulk", removed)
            return removed
        except Exception as e:
            logger.error(f"Failed to bulk delete knowledge entries: {e}")
            return 0
    
    async def search(self, query: KnowledgeQuery) -> List[KnowledgeSearchResult]:
        """
        Search the knowledge base using RAG.
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        try:
            # Generate embedding for query
            query_embedding = await self._resolve_query_embedding(query.query_text)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=query.limit,
                similarity_threshold=query.similarity_threshold
            )
            
            # Filter by categories, types, and tags if specified
            filtered_results = []
            for result in results:
                entry = result.entry
                
                # Filter by categories
                if query.categories and entry.category not in query.categories:
                    continue
                
                # Filter by entry types
                if query.entry_types and entry.entry_type not in query.entry_types:
                    continue
                
                # Filter by tags
                if query.tags and not any(tag in entry.tags for tag in query.tags):
                    continue
                
                filtered_results.append(result)
            
            logger.debug(f"Knowledge search returned {len(filtered_results)} results")
            return filtered_results
        except ImportError as e:
            logger.warning(f"Knowledge search failed due to missing dependencies: {e}")
            return []
        except Exception as e:
            logger.warning(f"Failed to search knowledge base: {e}")
            return []
    
    async def get_all_entries(self, 
                             category: Optional[str] = None,
                             entry_type: Optional[KnowledgeEntryType] = None) -> List[KnowledgeEntry]:
        """
        Get all knowledge entries, optionally filtered by category or type.
        
        Args:
            category: Filter by category (optional)
            entry_type: Filter by entry type (optional)
            
        Returns:
            List of knowledge entries
        """
        try:
            if self.knowledge_db_store and self.knowledge_db_store.is_available:
                db_entries = self.knowledge_db_store.list_entries(
                    self.user_id,
                    category=category,
                    entry_type=entry_type,
                )

                hydrated_entries: List[KnowledgeEntry] = []
                for entry in db_entries:
                    hydrated = self._attach_embedding_to_entry(entry)
                    if hydrated:
                        hydrated_entries.append(hydrated)

                return hydrated_entries

            all_entries = self.vector_store.get_all_entries()
            
            # Apply filters
            filtered_entries = []
            for entry in all_entries:
                if getattr(entry, "user_id", self.user_id) != self.user_id:
                    continue
                if category and entry.category != category:
                    continue
                if entry_type and entry.entry_type != entry_type:
                    continue
                filtered_entries.append(entry)
            
            return filtered_entries
        except Exception as e:
            logger.error(f"Failed to get all entries: {e}")
            return []
    
    async def get_user_preferences(self) -> UserPreferences:
        """
        Get user preferences, loading from knowledge base or creating defaults.
        
        Returns:
            User preferences object
        """
        try:
            if self._user_preferences is not None:
                return self._user_preferences

            db_preferences = self._load_preferences_from_db_store()
            if db_preferences is not None:
                self._user_preferences = db_preferences
                return self._user_preferences

            # Try to hydrate preferences from persisted knowledge entries first.
            try:
                all_entries = await self.get_all_entries()
                prefs_dict = UserPreferences().model_dump()
                loaded_from_knowledge = False

                system_pref_entries = [
                    entry
                    for entry in all_entries
                    if entry.entry_type == KnowledgeEntryType.PREFERENCE
                    and str(entry.category).strip().lower() == "system"
                    and str(entry.title).strip().lower() == "user preferences"
                ]

                if system_pref_entries:
                    latest_system_entry = max(system_pref_entries, key=lambda entry: entry.updated_at)
                    try:
                        snapshot = json.loads(latest_system_entry.content) if latest_system_entry.content else {}
                        if isinstance(snapshot, dict):
                            for section, value in snapshot.items():
                                if (
                                    section in prefs_dict
                                    and isinstance(prefs_dict[section], dict)
                                    and isinstance(value, dict)
                                ):
                                    prefs_dict[section].update(value)
                                else:
                                    prefs_dict[section] = value
                            loaded_from_knowledge = True
                    except Exception as snapshot_error:
                        logger.warning(f"Failed to parse system preference snapshot: {snapshot_error}")

                sectioned_pref_entries = [
                    entry
                    for entry in all_entries
                    if entry.entry_type == KnowledgeEntryType.PREFERENCE
                    and isinstance(entry.metadata, dict)
                    and str(entry.metadata.get("preference_section") or "").strip().lower() in prefs_dict
                ]

                for entry in sectioned_pref_entries:
                    metadata = entry.metadata or {}
                    section = str(metadata.get("preference_section") or "").strip().lower()
                    if section not in prefs_dict or not isinstance(prefs_dict.get(section), dict):
                        continue

                    section_values = metadata.get("preference_values")
                    if isinstance(section_values, dict):
                        prefs_dict[section].update(section_values)
                        loaded_from_knowledge = True

                user_entries = [
                    entry
                    for entry in all_entries
                    if entry.entry_type == KnowledgeEntryType.USER_PREFERENCE
                ]

                if user_entries:
                    loaded_from_knowledge = True
                    for entry in user_entries:
                        if entry.entry_sub_type == KnowledgeEntrySubType.USER_PROFILE:
                            metadata = entry.metadata or {}
                            prefs_dict["general"]["role"] = metadata.get("role", "professional")
                            prefs_dict["general"]["priorities"] = metadata.get("preferences", [])
                            prefs_dict["general"]["mentor"] = metadata.get("mentor", {})
                            prefs_dict["general"]["onboarding_completed"] = metadata.get("onboarding_completed", True)

                        elif entry.entry_sub_type == KnowledgeEntrySubType.GOAL:
                            metadata = entry.metadata or {}
                            category = metadata.get("category", "general").lower()
                            section_map = {
                                "career": "productivity",
                                "work": "productivity",
                                "productivity": "productivity",
                                "health": "health",
                                "wellness": "health",
                                "finance": "finance",
                                "financial": "finance",
                                "money": "finance",
                                "journal": "journal",
                                "reflection": "journal",
                            }
                            target_section = section_map.get(category, "productivity")
                            if "goals" not in prefs_dict[target_section]:
                                prefs_dict[target_section]["goals"] = []
                            prefs_dict[target_section]["goals"].append({
                                "title": entry.title,
                                "priority": metadata.get("priority", "Medium"),
                                "milestones": metadata.get("milestones", []),
                            })

                        elif entry.entry_sub_type == KnowledgeEntrySubType.SCHEDULE:
                            metadata = entry.metadata or {}
                            availability = metadata.get("availability", {})
                            prefs_dict["general"]["work_hours"] = (
                                f"{availability.get('workHours', {}).get('start', '09:00')}-"
                                f"{availability.get('workHours', {}).get('end', '17:00')}"
                            )
                            prefs_dict["general"]["timezone"] = availability.get("timezone", "UTC")
                            prefs_dict["journal"]["check_in_time"] = availability.get("checkIn", {}).get("preferredTime", "09:00")

                if loaded_from_knowledge:
                    logger.info(f"Loaded user preferences from knowledge base: {prefs_dict}")
                    prefs_dict["user_id"] = self.user_id
                    self._user_preferences = UserPreferences(**prefs_dict)
                    self._persist_preferences_to_db_store(self._user_preferences)
                    return self._user_preferences

            except Exception as e:
                logger.warning(f"Failed to load preferences from knowledge base: {e}, trying JSON file")

            # Fallback to JSON file
            prefs_path = os.path.join(os.path.dirname(__file__), "user_preferences.json")
            try:
                if os.path.exists(prefs_path):
                    with open(prefs_path) as f:
                        data = f.read().strip()
                        if not data:
                            raise ValueError("Empty preferences file")
                        prefs_dict = json.loads(data)
                        prefs_dict["user_id"] = self.user_id
                        self._user_preferences = UserPreferences(**prefs_dict)
                        self._persist_preferences_to_db_store(self._user_preferences)
                        return self._user_preferences
            except Exception as e:
                logger.warning(f"Failed to parse stored preferences: {e}. Using defaults.")

            self._user_preferences = UserPreferences(user_id=self.user_id)
            self._persist_preferences_to_db_store(self._user_preferences)
            return self._user_preferences
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return UserPreferences(user_id=self.user_id)

    def _is_time_entry_entry(self, entry: KnowledgeEntry) -> bool:
        """Detect AlterEgo time entries that are persisted as interaction events."""
        metadata = entry.metadata or {}
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

        category = str(entry.category or "").strip().lower()
        checkup_type = str(metadata.get("checkup_type") or "").strip().lower()

        # Daily checkup insights can carry legacy time_entry tags but should remain insights.
        if category == "daily_checkup" or checkup_type in {"morning", "evening"}:
            return False

        source = str(context.get("source", "")).strip().lower()
        source_action = str(context.get("source_action", "")).strip().lower()
        has_time_entry_id = context.get("time_entry_id") is not None
        has_time_entry_tag = any(str(tag).strip().lower() == "time_entry" for tag in (entry.tags or []))

        return (
            category == "time_entry"
            or source == "alterego_timetracker"
            or "time_entry" in source_action
            or has_time_entry_id
            or has_time_entry_tag
        )

    def _normalize_visual_category(self, entry: KnowledgeEntry) -> str:
        if self._is_time_entry_entry(entry):
            return "time_entry"

        category = str(entry.category or "uncategorized").strip().lower()
        return category if category else "uncategorized"

    def _normalize_visual_entry_type_label(self, entry_type: Any) -> str:
        """Normalize enum-like labels (including legacy serialized values) for UI filtering."""
        if hasattr(entry_type, "value"):
            raw_value = str(entry_type.value)
        else:
            raw_value = str(entry_type or "")

        normalized = raw_value.strip().lower()
        if not normalized:
            return "memory"

        if "." in normalized:
            normalized = normalized.split(".")[-1]

        return normalized or "memory"

    def _normalize_visual_type(self, entry: KnowledgeEntry, normalized_category: str) -> str:
        if normalized_category == "time_entry":
            return "time_entry"

        return self._normalize_visual_entry_type_label(entry.entry_type)

    async def _ensure_embedding_for_visualization_entry(self, entry: KnowledgeEntry) -> Optional[List[float]]:
        """Backfill embeddings for legacy entries that were saved without vectors."""
        existing_embedding = self.vector_store.get_embedding(entry.entry_id)
        if self._embedding_has_signal(existing_embedding):
            return existing_embedding

        metadata_payload = dict(entry.metadata or {})
        normalized_entry_type = self._normalize_visual_entry_type_label(entry.entry_type)
        try:
            resolved_entry_type = KnowledgeEntryType(normalized_entry_type)
        except Exception:
            resolved_entry_type = KnowledgeEntryType.MEMORY

        try:
            if isinstance(entry.entry_sub_type, KnowledgeEntrySubType):
                resolved_entry_sub_type = entry.entry_sub_type
            else:
                resolved_entry_sub_type = KnowledgeEntrySubType(str(entry.entry_sub_type))
        except Exception:
            resolved_entry_sub_type = KnowledgeEntrySubType.MISC_INTERACTION

        embedding_text, semantic_sections, strategy_category = self._build_embedding_document(
            title=entry.title,
            content=entry.content,
            tags=entry.tags,
            category=entry.category,
            metadata=metadata_payload,
            entry_type=resolved_entry_type,
            entry_sub_type=resolved_entry_sub_type,
        )
        embedding_key = self._build_embedding_cache_key(embedding_text)
        metadata_payload[EMBEDDING_CACHE_KEY_FIELD] = embedding_key

        entry_chunks = self._chunk_embedding_text(
            embedding_text,
            category=strategy_category,
            entry_type=resolved_entry_type,
            semantic_sections=semantic_sections,
        )

        self._log_embedding_payload(
            action="backfill",
            embedding_key=embedding_key,
            embedding_text=embedding_text,
            chunks=entry_chunks,
            entry_id=entry.entry_id,
            category=strategy_category,
        )

        generated_embedding = await self._resolve_embedding(
            embedding_text,
            embedding_key,
            existing_entry=entry,
            category=strategy_category,
            entry_type=resolved_entry_type,
            semantic_sections=semantic_sections,
        )
        if not generated_embedding:
            return None

        updated_entry = entry.model_copy()
        updated_entry.metadata = metadata_payload
        updated_entry.updated_at = datetime.utcnow()
        self.vector_store.update_entry(updated_entry, generated_embedding)
        self._persist_entry_to_db(updated_entry)
        self._index_sync_event_key(updated_entry)
        self._cache_entry_embedding(updated_entry)

        logger.info("Backfilled missing embedding for entry %s", entry.entry_id)
        return generated_embedding

    def _infer_interaction_category_and_sub_type(
        self,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, KnowledgeEntrySubType, List[str]]:
        """Derive interaction grouping so synced events surface in the right knowledge category."""
        context_payload = context or {}
        normalized_agent = (agent_type or "general").strip().lower() or "general"

        source = str(context_payload.get("source", "")).strip().lower()
        source_action = str(context_payload.get("source_action", "")).strip().lower()
        forced_category = str(context_payload.get("category", "")).strip().lower()
        has_time_entry_id = context_payload.get("time_entry_id") is not None
        approval_payload = context_payload.get("approval") if isinstance(context_payload.get("approval"), dict) else {}
        approved_as_insight = bool(
            context_payload.get("approved_as_insight")
            or approval_payload.get("approved_as_insight")
        )

        is_time_entry = (
            forced_category == "time_entry"
            or source == "alterego_timetracker"
            or "time_entry" in source_action
            or has_time_entry_id
        )
        if is_time_entry:
            return (
                "time_entry",
                KnowledgeEntrySubType.WORK_INTERACTION,
                ["interaction", "history", "time_entry", "alterego_sync", normalized_agent],
            )

        if forced_category in {"insight", "important_insight", "deep_insight"} or approved_as_insight:
            return (
                "insight",
                KnowledgeEntrySubType.IMPORTANT_INSIGHT,
                ["insight", "approved", normalized_agent],
            )

        if forced_category in {
            "habit_snapshot",
            "habit_progress",
            "project_catalog",
            "tag_catalog",
            "planner",
            "schedule",
            "goal",
            "goals",
        }:
            if normalized_agent == "health":
                sub_type = KnowledgeEntrySubType.HEALTH_INTERACTION
            elif normalized_agent in {"productivity", "finance", "scheduling", "habit_progress"}:
                sub_type = KnowledgeEntrySubType.WORK_INTERACTION
            else:
                sub_type = KnowledgeEntrySubType.MISC_INTERACTION

            return (
                forced_category,
                sub_type,
                ["interaction", "history", forced_category, normalized_agent],
            )

        if normalized_agent == "health":
            sub_type = KnowledgeEntrySubType.HEALTH_INTERACTION
        elif normalized_agent in {"productivity", "finance", "scheduling"}:
            sub_type = KnowledgeEntrySubType.WORK_INTERACTION
        elif normalized_agent in {"journal", "general", "orchestrator"}:
            sub_type = KnowledgeEntrySubType.PERSONAL_INTERACTION
        else:
            sub_type = KnowledgeEntrySubType.MISC_INTERACTION

        return normalized_agent, sub_type, ["interaction", "history", normalized_agent]

    def _build_time_entry_title(self, context_payload: Optional[Dict[str, Any]] = None) -> str:
        """Build a human-meaningful title for synced time entries."""
        payload = context_payload or {}

        project_name = str(payload.get("project_name", "")).strip()
        description = str(payload.get("description", "")).strip()
        task_name = str(payload.get("task_name", "")).strip()

        activity = task_name or description

        if project_name and activity and project_name.lower() != activity.lower():
            base_title = f"{project_name}: {activity}"
        else:
            base_title = activity or project_name

        duration_suffix = ""
        raw_duration = payload.get("duration_minutes")
        try:
            duration_minutes = int(round(float(raw_duration))) if raw_duration is not None else 0
            if duration_minutes > 0:
                duration_suffix = f" ({duration_minutes}m)"
        except (TypeError, ValueError):
            duration_suffix = ""

        if base_title:
            compact_title = base_title if len(base_title) <= 90 else f"{base_title[:87]}..."
            return f"Time Entry - {compact_title}{duration_suffix}"

        return f"Time Entry{duration_suffix}" if duration_suffix else "Time Entry"

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return default

    def _build_habit_highlights(self, habits_payload: Any, limit: int = 4) -> List[str]:
        if not isinstance(habits_payload, list):
            return []

        scored_habits: List[Tuple[int, str]] = []
        for item in habits_payload:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or item.get("title") or item.get("habit") or "").strip()
            if not name:
                continue

            completed = self._safe_int(
                item.get("completedCount")
                or item.get("completionCount")
                or item.get("completed")
                or 0,
                default=0,
            )
            streak = self._safe_int(item.get("streak") or item.get("currentStreak") or 0, default=0)
            score = max(completed, 0) + max(streak, 0)

            if completed > 0 and streak > 0:
                descriptor = f"{name}: {completed} completions, {streak}-day streak"
            elif completed > 0:
                descriptor = f"{name}: {completed} completions"
            elif streak > 0:
                descriptor = f"{name}: {streak}-day streak"
            else:
                descriptor = f"{name}: active"

            scored_habits.append((score, descriptor))

        if not scored_habits:
            return []

        scored_habits.sort(key=lambda item: item[0], reverse=True)
        return [descriptor for _, descriptor in scored_habits[: max(1, limit)]]

    def _build_daily_completion_digest(self, counts_payload: Any) -> Dict[str, Any]:
        if not isinstance(counts_payload, dict):
            return {}

        normalized_pairs: List[Tuple[str, int]] = []
        for raw_day, raw_count in counts_payload.items():
            day = str(raw_day or "").strip()
            if not day:
                continue
            normalized_pairs.append((day, max(0, self._safe_int(raw_count, default=0))))

        if not normalized_pairs:
            return {}

        normalized_pairs.sort(key=lambda item: item[0])
        total = sum(count for _, count in normalized_pairs)
        best_day, best_count = max(normalized_pairs, key=lambda item: item[1])

        recent = normalized_pairs[-7:]
        return {
            "total_events": total,
            "active_days": sum(1 for _, count in normalized_pairs if count > 0),
            "best_day": best_day,
            "best_day_count": best_count,
            "recent": [{"day": day, "count": count} for day, count in recent],
        }

    def _compact_interaction_context_for_storage(
        self,
        category: str,
        context_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if category not in {"habit_snapshot", "habit_progress"}:
            return context_payload

        summary_payload = context_payload.get("summary") if isinstance(context_payload.get("summary"), dict) else {}
        habits_payload = context_payload.get("habits")
        daily_counts_payload = context_payload.get("daily_completion_counts")

        total_habits = self._safe_int(
            context_payload.get("total_habits") or summary_payload.get("totalHabits") or 0,
            default=0,
        )
        total_events = self._safe_int(
            context_payload.get("total_completion_events") or summary_payload.get("totalCompletionEvents") or 0,
            default=0,
        )
        active_days = self._safe_int(
            context_payload.get("active_days") or summary_payload.get("activeDays") or 0,
            default=0,
        )
        current_run = self._safe_int(
            context_payload.get("current_run") or summary_payload.get("currentRun") or 0,
            default=0,
        )
        longest_run = self._safe_int(
            context_payload.get("longest_run") or summary_payload.get("longestRun") or 0,
            default=0,
        )

        compact_context: Dict[str, Any] = {
            "source": context_payload.get("source"),
            "source_action": context_payload.get("source_action"),
            "category": "habit_snapshot",
            "captured_at": context_payload.get("captured_at"),
            "sync_event_key": context_payload.get("sync_event_key"),
            "user_id": context_payload.get("user_id"),
            "user_email": context_payload.get("user_email"),
            "total_habits": total_habits,
            "total_completion_events": total_events,
            "active_days": active_days,
            "current_run": current_run,
            "longest_run": longest_run,
        }

        highlights = self._build_habit_highlights(habits_payload)
        if highlights:
            compact_context["habit_highlights"] = highlights

        digest = self._build_daily_completion_digest(daily_counts_payload)
        if digest:
            compact_context["daily_completion_digest"] = digest

        return {key: value for key, value in compact_context.items() if value not in (None, "", [], {})}

    def _build_habit_snapshot_title(self, context_payload: Optional[Dict[str, Any]] = None) -> str:
        payload = context_payload or {}
        captured_at = str(payload.get("captured_at") or "").strip()
        total_habits = self._safe_int(payload.get("total_habits"), default=0)
        total_events = self._safe_int(payload.get("total_completion_events"), default=0)

        date_hint = captured_at[:10] if len(captured_at) >= 10 else ""
        if date_hint and total_habits > 0:
            return f"Habit Snapshot - {date_hint} ({total_habits} habits, {total_events} events)"
        if date_hint:
            return f"Habit Snapshot - {date_hint}"
        if total_habits > 0:
            return f"Habit Snapshot ({total_habits} habits, {total_events} events)"
        return "Habit Snapshot"

    def _build_habit_snapshot_content(
        self,
        user_input: str,
        agent_response: str,
        context_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = context_payload or {}
        captured_at = str(payload.get("captured_at") or "").strip() or "unknown"
        total_habits = self._safe_int(payload.get("total_habits"), default=0)
        total_events = self._safe_int(payload.get("total_completion_events"), default=0)
        active_days = self._safe_int(payload.get("active_days"), default=0)
        current_run = self._safe_int(payload.get("current_run"), default=0)
        longest_run = self._safe_int(payload.get("longest_run"), default=0)
        highlights = payload.get("habit_highlights") if isinstance(payload.get("habit_highlights"), list) else []

        digest_payload = payload.get("daily_completion_digest") if isinstance(payload.get("daily_completion_digest"), dict) else {}
        best_day = str(digest_payload.get("best_day") or "").strip()
        best_day_count = self._safe_int(digest_payload.get("best_day_count"), default=0)

        lines: List[str] = [
            "Habit progress snapshot recorded.",
            f"Captured at: {captured_at}",
            (
                "Progress metrics: "
                f"{total_habits} habits, {total_events} completion events, "
                f"{active_days} active days, {current_run}-day current run, {longest_run}-day longest run"
            ),
        ]

        if highlights:
            lines.append("Habit highlights: " + "; ".join(str(item) for item in highlights[:4]))

        if best_day and best_day_count > 0:
            lines.append(f"Best completion day: {best_day} ({best_day_count} completions)")

        if user_input.strip():
            lines.append(f"Sync trigger: {user_input.strip()}")

        if agent_response.strip():
            lines.append(f"Source summary: {agent_response.strip()}")

        return "\n".join(lines)

    def _build_interaction_content(
        self,
        *,
        category: str,
        agent_type: str,
        user_input: str,
        agent_response: str,
        context_payload: Dict[str, Any],
        approved_by_user: bool,
        approved_at: Optional[str],
        knowledge_sources: List[Any],
    ) -> str:
        if category == "time_entry":
            return f"User: {user_input}\nAgent ({agent_type}): {agent_response}"

        if category in {"habit_snapshot", "habit_progress"}:
            return self._build_habit_snapshot_content(user_input, agent_response, context_payload)

        lines = [
            f"Interaction category: {category}",
            f"Agent: {agent_type}",
            f"User input: {user_input.strip() or 'n/a'}",
            f"Agent response: {agent_response.strip() or 'n/a'}",
        ]
        source_action = str(context_payload.get("source_action") or "").strip()
        if source_action:
            lines.append(f"Source action: {source_action}")
        if approved_by_user:
            lines.append(f"Approved by user at: {approved_at or 'confirmed'}")
        if knowledge_sources:
            lines.append(f"Knowledge sources referenced: {len(knowledge_sources)}")
        return "\n".join(lines)

    def _build_interaction_title(
        self,
        category: str,
        context_payload: Optional[Dict[str, Any]],
    ) -> str:
        if category == "time_entry":
            return self._build_time_entry_title(context_payload)

        if category in {"habit_snapshot", "habit_progress"}:
            return self._build_habit_snapshot_title(context_payload)

        if category == "insight":
            payload = context_payload or {}
            agent_hint = str(payload.get("agent_type") or payload.get("source_action") or "").strip()
            if agent_hint:
                return f"Approved Insight - {agent_hint.replace('_', ' ').title()}"
            return "Approved Insight"

        title_source = category.replace("_", " ").title()
        return f"Interaction with {title_source}"

    def _extract_sync_event_key(self, entry: KnowledgeEntry) -> str:
        metadata = entry.metadata or {}
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        return str(context.get("sync_event_key", "")).strip()

    def _index_sync_event_key(self, entry: KnowledgeEntry) -> None:
        if entry.entry_type != KnowledgeEntryType.INTERACTION:
            return

        sync_event_key = self._extract_sync_event_key(entry)
        if sync_event_key:
            self._sync_event_index[sync_event_key] = entry.entry_id

    def _remove_indexed_sync_event_for_entry(self, entry_id: str) -> None:
        stale_keys = [key for key, mapped_entry_id in self._sync_event_index.items() if mapped_entry_id == entry_id]
        for key in stale_keys:
            self._sync_event_index.pop(key, None)

    async def _ensure_sync_event_index_loaded(self) -> None:
        if self._sync_event_index_loaded:
            return

        try:
            interaction_entries = await self.get_all_entries(entry_type=KnowledgeEntryType.INTERACTION)
            self._sync_event_index = {}
            for entry in interaction_entries:
                self._index_sync_event_key(entry)
        finally:
            self._sync_event_index_loaded = True

    async def _find_interaction_by_sync_event_key(self, sync_event_key: str) -> Optional[KnowledgeEntry]:
        """Find an existing interaction entry by external sync key."""
        normalized_sync_key = str(sync_event_key or "").strip()
        if not normalized_sync_key:
            return None

        await self._ensure_sync_event_index_loaded()

        entry_id = self._sync_event_index.get(normalized_sync_key)
        if entry_id:
            entry = self.vector_store.get_entry(entry_id)
            if entry:
                return entry
            self._sync_event_index.pop(normalized_sync_key, None)

        # Fallback scan for robustness in case index becomes stale.
        existing_entries = await self.get_all_entries(entry_type=KnowledgeEntryType.INTERACTION)
        for entry in existing_entries:
            if self._extract_sync_event_key(entry) == normalized_sync_key:
                self._sync_event_index[normalized_sync_key] = entry.entry_id
                return entry

        return None
    
    async def update_user_preferences(self, preferences: UserPreferences) -> bool:
        """
        Update user preferences in the knowledge base.
        
        Args:
            preferences: Updated preferences
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._user_preferences = preferences
            return await self._save_user_preferences()
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False
    
    async def add_user_preference(self, category: str, key: str, value: Any, description: Optional[str] = None) -> bool:
        """
        Add a new user preference.
        
        Args:
            category: Category of the preference (e.g., 'productivity', 'health')
            key: Key name for the preference
            value: Value of the preference
            description: Optional description of the preference
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current preferences
            current_prefs = await self.get_user_preferences()
            
            # Convert to dict for manipulation
            prefs_dict = current_prefs.model_dump()
            
            # Ensure category exists
            if category not in prefs_dict:
                prefs_dict[category] = {}
            
            # Add the new preference
            prefs_dict[category][key] = value
            
            # If description provided, store it in metadata
            if description:
                metadata_key = f"__{key}_description"
                prefs_dict[category][metadata_key] = description
            
            # Update preferences
            updated_prefs = UserPreferences(**prefs_dict)
            success = await self.update_user_preferences(updated_prefs)
            
            if success:
                logger.info(f"Added user preference: {category}.{key} = {value}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to add user preference {category}.{key}: {e}")
            return False
    
    async def remove_user_preference(self, category: str, key: str) -> bool:
        """
        Remove a user preference.
        
        Args:
            category: Category of the preference
            key: Key name for the preference
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current preferences
            current_prefs = await self.get_user_preferences()
            
            # Convert to dict for manipulation
            prefs_dict = current_prefs.model_dump()
            
            # Check if category and key exist
            if category not in prefs_dict or key not in prefs_dict[category]:
                logger.warning(f"Preference {category}.{key} not found for removal")
                return False
            
            # Remove the preference
            del prefs_dict[category][key]
            
            # Also remove description if it exists
            description_key = f"__{key}_description"
            if description_key in prefs_dict[category]:
                del prefs_dict[category][description_key]
            
            # Update preferences
            updated_prefs = UserPreferences(**prefs_dict)
            success = await self.update_user_preferences(updated_prefs)
            
            if success:
                logger.info(f"Removed user preference: {category}.{key}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to remove user preference {category}.{key}: {e}")
            return False
    
    async def get_preference_categories(self) -> List[str]:
        """
        Get all available preference categories.
        
        Returns:
            List of preference category names
        """
        try:
            if self.preference_db_store and self.preference_db_store.is_available:
                categories = self.preference_db_store.list_categories(self.user_id)
                if categories:
                    return categories

            current_prefs = await self.get_user_preferences()
            prefs_dict = current_prefs.model_dump()
            return [key for key, value in prefs_dict.items() if key != "user_id" and isinstance(value, dict)]
        except Exception as e:
            logger.error(f"Failed to get preference categories: {e}")
            return []

    def _is_legacy_system_preference_entry(self, entry: KnowledgeEntry) -> bool:
        if entry.entry_type != KnowledgeEntryType.PREFERENCE:
            return False

        return (
            str(entry.category or "").strip().lower() == "system"
            and str(entry.title or "").strip().lower() == "user preferences"
        )

    def _normalize_preference_snapshot_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            normalized: Dict[str, Any] = {}
            for key, nested_value in value.items():
                key_text = str(key or "").strip()
                if not key_text or key_text.startswith("__"):
                    continue
                normalized[key_text] = self._normalize_preference_snapshot_value(nested_value)
            return normalized

        if isinstance(value, list):
            normalized_items = [self._normalize_preference_snapshot_value(item) for item in value[:8]]
            return [item for item in normalized_items if item not in (None, "", [], {})]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    def _render_preference_section_content(self, section: str, values: Dict[str, Any]) -> str:
        header = f"{section.replace('_', ' ').title()} preference snapshot"
        lines = [header]

        for key in sorted(values.keys()):
            rendered = self._stringify_embedding_value(values.get(key), max_chars=180)
            if not rendered:
                continue
            lines.append(f"{key.replace('_', ' ').title()}: {rendered}")

        return "\n".join(lines[:14])

    async def _sync_preference_snapshot_entries(self, preferences: UserPreferences) -> bool:
        prefs_payload = preferences.model_dump() if isinstance(preferences, UserPreferences) else {}
        if not isinstance(prefs_payload, dict):
            return False

        existing_entries = await self.get_all_entries(entry_type=KnowledgeEntryType.PREFERENCE)
        section_entries: Dict[str, KnowledgeEntry] = {}
        legacy_entries: List[KnowledgeEntry] = []

        for entry in existing_entries:
            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            section = str(metadata.get("preference_section") or "").strip().lower()
            if section:
                section_entries[section] = entry

            if self._is_legacy_system_preference_entry(entry):
                legacy_entries.append(entry)

        operation_succeeded = True
        timestamp_iso = datetime.utcnow().isoformat()

        for section, raw_values in prefs_payload.items():
            if section == "user_id" or not isinstance(raw_values, dict):
                continue

            normalized_values = self._normalize_preference_snapshot_value(raw_values)
            if not isinstance(normalized_values, dict):
                normalized_values = {}

            title = f"Preference Snapshot - {section.replace('_', ' ').title()}"
            content = self._render_preference_section_content(section, normalized_values)
            metadata_payload = {
                "preference_section": section,
                "preference_values": normalized_values,
                "source": "preferences_snapshot",
                "last_updated": timestamp_iso,
            }
            tags = ["preferences", "settings", section]

            existing_entry = section_entries.get(section)
            if existing_entry:
                existing_metadata = existing_entry.metadata if isinstance(existing_entry.metadata, dict) else {}
                existing_values = existing_metadata.get("preference_values") if isinstance(existing_metadata.get("preference_values"), dict) else {}

                if (
                    existing_entry.title == title
                    and str(existing_entry.content or "").strip() == str(content or "").strip()
                    and existing_values == normalized_values
                ):
                    continue

                updated = await self.update_entry(
                    entry_id=existing_entry.entry_id,
                    title=title,
                    content=content,
                    metadata=metadata_payload,
                    tags=tags,
                )
                operation_succeeded = operation_succeeded and updated is not None
                continue

            try:
                await self.create_entry(
                    entry_type=KnowledgeEntryType.PREFERENCE,
                    entry_sub_type=KnowledgeEntrySubType.OTHER_PREFERENCE,
                    category=section,
                    title=title,
                    content=content,
                    metadata=metadata_payload,
                    tags=tags,
                )
            except Exception:
                operation_succeeded = False

        for entry in legacy_entries:
            deleted = await self.delete_entry(entry.entry_id)
            operation_succeeded = operation_succeeded and deleted

        return operation_succeeded
    
    async def _save_user_preferences(self) -> bool:
        """Save user preferences to dedicated preference storage."""
        try:
            if not self._user_preferences:
                return False

            persisted_to_db = False

            if self.preference_db_store and self.preference_db_store.is_available:
                persisted_to_db = self._persist_preferences_to_db_store(self._user_preferences)
                if not persisted_to_db:
                    logger.warning("Falling back to legacy knowledge-entry preference persistence")

            sectioned_persistence_ok = await self._sync_preference_snapshot_entries(self._user_preferences)
            if persisted_to_db and sectioned_persistence_ok:
                return True

            if persisted_to_db:
                return True

            if sectioned_persistence_ok:
                return True

            # Last-resort legacy fallback when both dedicated DB and sectioned snapshots fail.
            existing_entries = await self.get_all_entries(
                category="system",
                entry_type=KnowledgeEntryType.PREFERENCE,
            )

            prefs_json = self._user_preferences.model_dump_json(indent=2)
            if existing_entries:
                entry = existing_entries[0]
                await self.update_entry(
                    entry_id=entry.entry_id,
                    content=prefs_json,
                    metadata={"last_updated": datetime.utcnow().isoformat()},
                )
                return True

            await self.create_entry(
                entry_type=KnowledgeEntryType.PREFERENCE,
                category="system",
                entry_sub_type=KnowledgeEntrySubType.OTHER_PREFERENCE,
                title="User Preferences",
                content=prefs_json,
                metadata={"created": datetime.utcnow().isoformat()},
                tags=["preferences", "settings", "configuration"],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            return False
    
    async def add_interaction_history(self, 
                                    agent_type: str,
                                    user_input: str,
                                    agent_response: str,
                                    context: Optional[Dict[str, Any]] = None) -> KnowledgeEntry:
        """
        Add an interaction to the history for learning purposes.
        
        Args:
            agent_type: Type of agent that handled the interaction
            user_input: User's input
            agent_response: Agent's response
            context: Additional context information
        
        Returns:
            The created interaction entry
        """
        try:
            context_payload = dict(context or {})
            category, entry_sub_type, tags = self._infer_interaction_category_and_sub_type(
                agent_type=agent_type,
                context=context_payload,
            )
            context_payload = self._compact_interaction_context_for_storage(category, context_payload)

            entry_type = KnowledgeEntryType.INSIGHT if category == "insight" else KnowledgeEntryType.INTERACTION

            approval_payload = context_payload.get("approval") if isinstance(context_payload.get("approval"), dict) else {}
            approved_by_user = bool(context_payload.get("approved_by_user") or approval_payload.get("approved"))
            approved_at = context_payload.get("approved_at") or approval_payload.get("approved_at")
            knowledge_sources = context_payload.get("knowledge_sources")
            if not isinstance(knowledge_sources, list):
                knowledge_sources = []

            interaction_title = self._build_interaction_title(category, context_payload)
            interaction_content = self._build_interaction_content(
                category=category,
                agent_type=agent_type,
                user_input=user_input,
                agent_response=agent_response,
                context_payload=context_payload,
                approved_by_user=approved_by_user,
                approved_at=approved_at,
                knowledge_sources=knowledge_sources,
            )

            metadata_payload = {
                "agent_type": agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context_payload,
                "user_input_length": len(user_input),
                "response_length": len(agent_response),
                "approved_by_user": approved_by_user,
                "approved_at": approved_at,
                "knowledge_source_count": len(knowledge_sources),
                "user_input": user_input[:400],
                "agent_response": agent_response[:800],
            }

            sync_event_key = str(context_payload.get("sync_event_key", "")).strip()
            if sync_event_key:
                existing_entry = await self._find_interaction_by_sync_event_key(sync_event_key)
                if existing_entry:
                    updated_entry = await self.update_entry(
                        entry_id=existing_entry.entry_id,
                        title=interaction_title,
                        content=interaction_content,
                        metadata=metadata_payload,
                        tags=tags,
                    )
                    if updated_entry:
                        return updated_entry

            return await self.create_entry(
                entry_type=entry_type,
                entry_sub_type=entry_sub_type,
                category=category,
                title=interaction_title,
                content=interaction_content,
                metadata=metadata_payload,
                tags=tags
            )
        except Exception as e:
            logger.error(f"Failed to add interaction history: {e}")
            raise

    async def extract_and_store_preferences(self, 
                                          user_input: str, 
                                          agent_type: str,
                                          agent_response: str) -> List[KnowledgeEntry]:
        """
        Extract and store user preferences from conversation.
        
        Args:
            user_input: User's input
            agent_type: Type of agent handling the request
            agent_response: Agent's response
            
        Returns:
            List of created preference entries
        """
        try:
            # Use LLM to extract preferences from the conversation
            llm_service = await get_llm_service()
            if not llm_service:
                logger.warning("LLM service not available for preference extraction")
                return []
            
            # Simplified and faster preference extraction prompt
            extraction_prompt = f"""
            Extract user preferences from this conversation:

            User: {user_input}
            Agent: {agent_response}

            Find explicit preferences like:
            - Foods they like/dislike
            - Exercise habits
            - Goals mentioned
            - Schedule preferences
            - Budget constraints

            Return JSON list:
            [{{"category": "health", "key": "preference_name", "value": "preference_value"}}]

            Return [] if no clear preferences found.
            """
            
            from ..llm.base import CompletionRequest, ChatMessage
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=extraction_prompt)],
                temperature=0.1,
                max_tokens=500  # Reduced for faster response
            )
            
            response = await llm_service.chat_completion(request)
            
            # Parse the response with better error handling
            try:
                import json
                # Try to extract JSON from response
                response_payload = getattr(response, "content", response)
                if isinstance(response_payload, str):
                    response_text = response_payload.strip()
                elif isinstance(response_payload, dict):
                    for key in ("content", "response", "message", "text", "output"):
                        if key in response_payload and response_payload[key]:
                            response_text = str(response_payload[key]).strip()
                            break
                    else:
                        response_text = json.dumps(response_payload)
                elif isinstance(response_payload, (list, tuple)):
                    response_text = "\n".join(
                        str(part).strip() for part in response_payload if str(part).strip()
                    )
                else:
                    response_text = str(response_payload).strip()
                
                # Handle cases where response might have extra text
                if '[' in response_text and ']' in response_text:
                    start = response_text.find('[')
                    end = response_text.rfind(']') + 1
                    json_text = response_text[start:end]
                else:
                    json_text = response_text
                
                preferences_data = json.loads(json_text)
                created_entries = []
                
                # Handle simplified format
                for pref in preferences_data:
                    if isinstance(pref, dict) and 'category' in pref and 'key' in pref and 'value' in pref:
                        # Store preference using the existing method
                        success = await self.add_user_preference(
                            category=pref['category'],
                            key=pref['key'],
                            value=pref['value'],
                            description=f"Extracted from conversation: {pref['value']}"
                        )
                        if success:
                            # Create a knowledge entry for tracking
                            entry = await self.create_entry(
                                entry_type=KnowledgeEntryType.PREFERENCE,
                                entry_sub_type=KnowledgeEntrySubType.PERSONAL_PREFERENCE,
                                category=pref['category'],
                                title=f"{pref['key']} preference",
                                content=f"User preference: {pref['key']} = {pref['value']}",
                                metadata={
                                    "extracted_from_interaction": True,
                                    "agent_type": agent_type,
                                    "timestamp": datetime.utcnow().isoformat()
                                },
                                tags=[pref['category'], "preference", "extracted", agent_type]
                            )
                            if entry:
                                created_entries.append(entry)
                
                logger.info(f"Successfully extracted {len(created_entries)} preferences")
                return created_entries
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse preferences extraction response: {e}")
                logger.debug(f"Response content: {response.content}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract preferences: {e}")
            return []

    def _merge_ranked_results(
        self,
        *result_groups: List[KnowledgeSearchResult],
        limit: Optional[int] = None,
    ) -> List[KnowledgeSearchResult]:
        """Merge result groups by keeping the highest score per entry and sorting globally."""
        best_by_entry: Dict[str, KnowledgeSearchResult] = {}

        for group in result_groups:
            for result in group or []:
                entry = getattr(result, "entry", None)
                entry_id = getattr(entry, "entry_id", None)
                if not entry_id:
                    continue

                existing = best_by_entry.get(entry_id)
                if existing is None or float(result.similarity_score) > float(existing.similarity_score):
                    best_by_entry[entry_id] = result

        merged = sorted(
            best_by_entry.values(),
            key=lambda item: (
                float(item.similarity_score),
                getattr(getattr(item, "entry", None), "updated_at", datetime.min),
            ),
            reverse=True,
        )

        if limit is not None:
            return merged[:limit]
        return merged

    def _tokenize_for_lexical_fallback(self, text: str) -> Set[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())
        return {
            token
            for token in tokens
            if len(token) >= 3 and token not in LEXICAL_FALLBACK_STOPWORDS
        }

    def _build_entry_retrieval_text(self, entry: KnowledgeEntry) -> str:
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        context_payload = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

        context_fragments = []
        for key in (
            "description",
            "task_name",
            "project_name",
            "priority",
            "summary",
            "notes",
            "focus_target",
            "tomorrow_focus",
            "role",
            "source_action",
        ):
            value = context_payload.get(key, metadata.get(key))
            normalized = self._stringify_embedding_value(value)
            if normalized:
                context_fragments.append(normalized)

        parts = [
            str(entry.title or ""),
            str(entry.content or ""),
            str(entry.category or ""),
            " ".join(str(tag) for tag in (entry.tags or [])),
            " ".join(context_fragments),
        ]

        return " ".join(part for part in parts if part).strip()

    def _score_entry_lexical_relevance(self, query_tokens: Set[str], entry: KnowledgeEntry) -> float:
        if not query_tokens:
            return 0.0

        entry_tokens = self._tokenize_for_lexical_fallback(self._build_entry_retrieval_text(entry))
        if not entry_tokens:
            return 0.0

        overlap = query_tokens.intersection(entry_tokens)
        if not overlap:
            return 0.0

        coverage = len(overlap) / max(len(query_tokens), 1)
        density = len(overlap) / max(len(entry_tokens), 1)
        density_weight = min(0.25, 0.06 + (len(query_tokens) * 0.02))
        score = (coverage * (1.0 - density_weight)) + (density * density_weight)

        if self._is_time_entry_entry(entry):
            score += 0.03

        return min(0.95, max(0.0, score))

    def _entry_matches_query_filters(
        self,
        entry: KnowledgeEntry,
        *,
        categories: Optional[List[str]] = None,
        entry_types: Optional[List[KnowledgeEntryType]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        if categories and entry.category not in categories:
            return False

        if entry_types and entry.entry_type not in entry_types:
            return False

        if tags and not any(tag in entry.tags for tag in tags):
            return False

        return True

    async def _build_lexical_fallback_results(
        self,
        *,
        query_text: str,
        limit: int,
        exclude_entry_ids: Optional[Set[str]] = None,
        categories: Optional[List[str]] = None,
        entry_types: Optional[List[KnowledgeEntryType]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[KnowledgeSearchResult]:
        if not RAG_LEXICAL_FALLBACK_ENABLED:
            return []

        query_tokens = self._tokenize_for_lexical_fallback(query_text)
        if not query_tokens:
            return []

        excluded_ids = exclude_entry_ids or set()
        all_entries = await self.get_all_entries()

        candidates: List[KnowledgeSearchResult] = []
        for entry in all_entries:
            if entry.entry_id in excluded_ids:
                continue

            if not self._entry_matches_query_filters(
                entry,
                categories=categories,
                entry_types=entry_types,
                tags=tags,
            ):
                continue

            lexical_score = self._score_entry_lexical_relevance(query_tokens, entry)
            if lexical_score < 0.16:
                continue

            candidates.append(
                KnowledgeSearchResult(
                    entry=entry,
                    similarity_score=float(round(lexical_score, 4)),
                    relevance_explanation="lexical_fallback",
                )
            )

        candidates.sort(
            key=lambda item: (float(item.similarity_score), item.entry.updated_at),
            reverse=True,
        )
        return candidates[:limit]

    def _is_action_planning_query(self, user_input: str, agent_type: str) -> bool:
        normalized_input = str(user_input or "").lower()
        normalized_agent = str(agent_type or "").strip().lower()

        if normalized_agent not in {"general", "productivity", "scheduling"}:
            return False

        planning_keywords = {
            "prioritize", "priority", "focus", "next", "work", "today", "plan", "planning", "task", "tasks", "schedule", "do now",
        }
        return any(keyword in normalized_input for keyword in planning_keywords)

    async def _build_recent_history_fallback_results(
        self,
        *,
        limit: int,
        exclude_entry_ids: Optional[Set[str]] = None,
    ) -> List[KnowledgeSearchResult]:
        excluded_ids = exclude_entry_ids or set()
        interaction_entries = await self.get_all_entries(entry_type=KnowledgeEntryType.INTERACTION)

        time_entries = [
            entry
            for entry in interaction_entries
            if entry.entry_id not in excluded_ids and self._is_time_entry_entry(entry)
        ]
        time_entries.sort(key=lambda entry: entry.updated_at, reverse=True)

        fallback_results: List[KnowledgeSearchResult] = []
        now = datetime.utcnow()

        for index, entry in enumerate(time_entries[:limit]):
            age_hours = max(0.0, (now - entry.updated_at).total_seconds() / 3600.0)
            recency_penalty = min(0.18, math.log1p(age_hours) * 0.035)
            base_score = 0.46 - (index * 0.04)
            score = max(0.18, base_score - recency_penalty)

            fallback_results.append(
                KnowledgeSearchResult(
                    entry=entry,
                    similarity_score=float(round(score, 4)),
                    relevance_explanation="recent_history_fallback",
                )
            )

        return fallback_results

    def _parse_datetime_value(self, value: Any) -> Optional[datetime]:
        """Parse timestamps from heterogeneous metadata payloads into naive UTC datetimes."""
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return value.astimezone(timezone.utc).replace(tzinfo=None)
            return value

        text = str(value or "").strip()
        if not text:
            return None

        normalized = text.replace("Z", "+00:00")

        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(normalized, fmt)
                    break
                except ValueError:
                    continue

            if parsed is None:
                return None

        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _infer_time_window(self, user_input: str, reference_time: datetime) -> Dict[str, Any]:
        """Infer the reporting time window from the user query."""
        text = str(user_input or "").strip().lower()
        day_start = datetime(reference_time.year, reference_time.month, reference_time.day)

        if re.search(r"\byesterday\b", text):
            start = day_start - timedelta(days=1)
            return {
                "key": "yesterday",
                "label": "Yesterday",
                "start": start,
                "end": start + timedelta(days=1),
            }

        if re.search(r"\btoday\b", text):
            return {
                "key": "today",
                "label": "Today",
                "start": day_start,
                "end": day_start + timedelta(days=1),
            }

        if re.search(r"\b(this week|weekly|week)\b", text):
            week_start = day_start - timedelta(days=day_start.weekday())
            return {
                "key": "this_week",
                "label": "This Week",
                "start": week_start,
                "end": week_start + timedelta(days=7),
            }

        return {
            "key": "unspecified",
            "label": "Recent",
            "start": None,
            "end": None,
        }

    def _collect_time_entry_records(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Normalize persisted time-entry interactions into a unified record format."""
        records: List[Dict[str, Any]] = []

        for entry in entries:
            if not self._is_time_entry_entry(entry):
                continue

            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            context_payload = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

            start_time = self._parse_datetime_value(context_payload.get("start_time"))
            end_time = self._parse_datetime_value(context_payload.get("end_time"))

            raw_duration = context_payload.get("duration_minutes")
            if raw_duration is None and context_payload.get("duration_seconds") is not None:
                try:
                    raw_duration = float(context_payload.get("duration_seconds", 0.0)) / 60.0
                except (TypeError, ValueError):
                    raw_duration = None

            duration_minutes: Optional[float]
            try:
                duration_minutes = float(raw_duration) if raw_duration is not None else None
            except (TypeError, ValueError):
                duration_minutes = None

            if start_time and end_time and duration_minutes is None:
                duration_minutes = max(0.0, (end_time - start_time).total_seconds() / 60.0)

            if start_time and duration_minutes is not None and end_time is None:
                end_time = start_time + timedelta(minutes=max(0.0, duration_minutes))

            if end_time and duration_minutes is not None and start_time is None:
                start_time = end_time - timedelta(minutes=max(0.0, duration_minutes))

            fallback_time = entry.updated_at if isinstance(entry.updated_at, datetime) else entry.created_at
            effective_start = start_time or fallback_time
            effective_end = end_time or effective_start

            if duration_minutes is None:
                duration_minutes = max(0.0, (effective_end - effective_start).total_seconds() / 60.0)

            records.append(
                {
                    "entry_id": entry.entry_id,
                    "project_name": str(context_payload.get("project_name") or "Unassigned").strip() or "Unassigned",
                    "description": str(context_payload.get("description") or context_payload.get("task_name") or entry.title or "work session").strip(),
                    "duration_minutes": max(0.0, float(duration_minutes or 0.0)),
                    "billable": bool(context_payload.get("billable", False)),
                    "start_dt": effective_start,
                    "end_dt": effective_end,
                    "start_time": context_payload.get("start_time") or effective_start.isoformat(timespec="minutes"),
                    "end_time": context_payload.get("end_time") or effective_end.isoformat(timespec="minutes"),
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "source_action": str(context_payload.get("source_action") or "").strip(),
                }
            )

        records.sort(key=lambda item: item.get("start_dt") or datetime.min, reverse=True)
        return records

    def _filter_records_for_window(self, records: List[Dict[str, Any]], window: Dict[str, Any]) -> List[Dict[str, Any]]:
        start = window.get("start")
        end = window.get("end")
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            return list(records)

        selected: List[Dict[str, Any]] = []
        for record in records:
            record_start = record.get("start_dt") or datetime.min
            record_end = record.get("end_dt") or record_start
            if record_end > start and record_start < end:
                selected.append(record)

        return selected

    def _summarize_time_window_records(
        self,
        records: List[Dict[str, Any]],
        window: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not records:
            return {
                "window_key": window.get("key", "unspecified"),
                "window_label": window.get("label", "Recent"),
                "has_data": False,
                "entry_count": 0,
                "total_logged_minutes": 0.0,
                "active_minutes": 0.0,
                "span_minutes": 0.0,
                "idle_minutes": 0.0,
                "gap_count": 0,
                "top_projects": [],
                "top_entries": [],
            }

        total_logged_minutes = sum(float(item.get("duration_minutes") or 0.0) for item in records)

        project_minutes: Dict[str, float] = {}
        for record in records:
            project_name = str(record.get("project_name") or "Unassigned").strip() or "Unassigned"
            project_minutes[project_name] = project_minutes.get(project_name, 0.0) + float(record.get("duration_minutes") or 0.0)

        top_projects = [
            {
                "project_name": project,
                "minutes": round(minutes, 1),
            }
            for project, minutes in sorted(project_minutes.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

        intervals = []
        for record in records:
            start_dt = record.get("start_dt")
            end_dt = record.get("end_dt")
            if isinstance(start_dt, datetime) and isinstance(end_dt, datetime) and end_dt >= start_dt:
                intervals.append((start_dt, end_dt))

        intervals.sort(key=lambda item: item[0])
        merged_intervals: List[List[datetime]] = []
        for start_dt, end_dt in intervals:
            if not merged_intervals or start_dt > merged_intervals[-1][1]:
                merged_intervals.append([start_dt, end_dt])
            elif end_dt > merged_intervals[-1][1]:
                merged_intervals[-1][1] = end_dt

        active_minutes = sum(
            max(0.0, (interval_end - interval_start).total_seconds() / 60.0)
            for interval_start, interval_end in merged_intervals
        )

        span_minutes = 0.0
        gap_count = 0
        if merged_intervals:
            span_minutes = max(0.0, (merged_intervals[-1][1] - merged_intervals[0][0]).total_seconds() / 60.0)
            for idx in range(1, len(merged_intervals)):
                gap_duration = (merged_intervals[idx][0] - merged_intervals[idx - 1][1]).total_seconds() / 60.0
                if gap_duration > 0:
                    gap_count += 1

        idle_minutes = max(0.0, span_minutes - active_minutes)

        top_entries = [
            {
                "entry_id": record.get("entry_id"),
                "project_name": record.get("project_name"),
                "description": record.get("description"),
                "duration_minutes": round(float(record.get("duration_minutes") or 0.0), 1),
                "billable": bool(record.get("billable", False)),
                "start_time": record.get("start_time"),
                "end_time": record.get("end_time"),
            }
            for record in sorted(records, key=lambda item: float(item.get("duration_minutes") or 0.0), reverse=True)[:8]
        ]

        return {
            "window_key": window.get("key", "unspecified"),
            "window_label": window.get("label", "Recent"),
            "has_data": True,
            "entry_count": len(records),
            "total_logged_minutes": round(total_logged_minutes, 1),
            "active_minutes": round(active_minutes, 1),
            "span_minutes": round(span_minutes, 1),
            "idle_minutes": round(idle_minutes, 1),
            "gap_count": gap_count,
            "top_projects": top_projects,
            "top_entries": top_entries,
        }

    async def _build_time_window_summary(self, user_input: str) -> Dict[str, Any]:
        """Build an all-entry time summary for review queries (today/yesterday/week)."""
        reference_time = datetime.utcnow()
        window = self._infer_time_window(user_input, reference_time)

        interaction_entries = await self.get_all_entries(entry_type=KnowledgeEntryType.INTERACTION)
        time_records = self._collect_time_entry_records(interaction_entries)
        filtered_records = self._filter_records_for_window(time_records, window)

        summary = self._summarize_time_window_records(filtered_records, window)
        summary["total_time_entry_records_available"] = len(time_records)
        return summary

    def _build_preference_context_from_model(
        self,
        preferences: UserPreferences,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        pref_payload = preferences.model_dump()
        snippets: List[Dict[str, Any]] = []

        for category, values in pref_payload.items():
            if category == "user_id" or not isinstance(values, dict):
                continue

            for key, value in values.items():
                if str(key).startswith("__"):
                    continue

                normalized_value = self._stringify_embedding_value(value, max_chars=160)
                if not normalized_value:
                    continue

                snippets.append(
                    {
                        "content": f"{key}: {normalized_value}",
                        "category": category,
                        "metadata": {"source": "preference_store", "key": key},
                        "similarity": 0.5,
                    }
                )

                if len(snippets) >= limit:
                    return snippets

        return snippets

    async def get_contextual_knowledge_for_agent(self, 
                                                user_input: str,
                                                agent_type: str,
                                                max_results: int = 10) -> Dict[str, Any]:
        """
        Get relevant knowledge context for an agent based on user input.
        
        Args:
            user_input: User's current input
            agent_type: Type of agent requesting context
            max_results: Maximum number of results per category
            
        Returns:
            Dictionary containing relevant context organized by type
        """
        try:
            # Get user preferences for this agent type
            preferences = await self.get_user_preferences()
            agent_preferences = getattr(preferences, agent_type.lower(), {})

            context_entry_types = [
                KnowledgeEntryType.INTERACTION,
                KnowledgeEntryType.PREFERENCE,
                KnowledgeEntryType.USER_PREFERENCE,
                KnowledgeEntryType.PATTERN,
                KnowledgeEntryType.INSIGHT,
            ]

            retrieval_limit = max(max_results * 2, max_results)
            max_combined_results = max(retrieval_limit, max_results + RAG_MIN_CONTEXT_RESULTS)
            fallback_modes: List[str] = []

            # Stage 1: strict semantic retrieval.
            search_results = await self.search(
                KnowledgeQuery(
                    query_text=user_input,
                    categories=[agent_type],
                    entry_types=context_entry_types,
                    limit=retrieval_limit,
                    similarity_threshold=RAG_PRIMARY_SIMILARITY_THRESHOLD,
                )
            )

            general_results = await self.search(
                KnowledgeQuery(
                    query_text=user_input,
                    entry_types=context_entry_types,
                    limit=retrieval_limit,
                    similarity_threshold=RAG_PRIMARY_SIMILARITY_THRESHOLD,
                )
            )

            combined_results = self._merge_ranked_results(
                search_results,
                general_results,
                limit=max_combined_results,
            )

            seen_entry_ids: Set[str] = {
                result.entry.entry_id
                for result in combined_results
                if getattr(result, "entry", None) and getattr(result.entry, "entry_id", None)
            }

            # Stage 2: relaxed semantic retrieval when strict retrieval is sparse.
            if len(combined_results) < RAG_MIN_CONTEXT_RESULTS:
                relaxed_agent_results = await self.search(
                    KnowledgeQuery(
                        query_text=user_input,
                        categories=[agent_type],
                        entry_types=context_entry_types,
                        limit=retrieval_limit,
                        similarity_threshold=RAG_RELAXED_SIMILARITY_THRESHOLD,
                    )
                )
                relaxed_general_results = await self.search(
                    KnowledgeQuery(
                        query_text=user_input,
                        entry_types=context_entry_types,
                        limit=retrieval_limit,
                        similarity_threshold=RAG_RELAXED_SIMILARITY_THRESHOLD,
                    )
                )

                previous_count = len(combined_results)
                combined_results = self._merge_ranked_results(
                    combined_results,
                    relaxed_agent_results,
                    relaxed_general_results,
                    limit=max_combined_results,
                )
                if len(combined_results) > previous_count:
                    fallback_modes.append("relaxed_similarity")

                seen_entry_ids.update(
                    result.entry.entry_id
                    for result in combined_results
                    if getattr(result, "entry", None) and getattr(result.entry, "entry_id", None)
                )

            # Stage 3: lexical retrieval fallback to avoid empty-context generic responses.
            if len(combined_results) < RAG_MIN_CONTEXT_RESULTS:
                lexical_results = await self._build_lexical_fallback_results(
                    query_text=user_input,
                    limit=max_combined_results,
                    exclude_entry_ids=seen_entry_ids,
                    entry_types=context_entry_types,
                )

                previous_count = len(combined_results)
                combined_results = self._merge_ranked_results(
                    combined_results,
                    lexical_results,
                    limit=max_combined_results,
                )
                if len(combined_results) > previous_count:
                    fallback_modes.append("lexical")

                seen_entry_ids.update(
                    result.entry.entry_id
                    for result in combined_results
                    if getattr(result, "entry", None) and getattr(result.entry, "entry_id", None)
                )

            # Stage 4: prioritization-specific recent history fallback for planning queries.
            if len(combined_results) < RAG_MIN_CONTEXT_RESULTS and self._is_action_planning_query(user_input, agent_type):
                recent_fallback_results = await self._build_recent_history_fallback_results(
                    limit=max(RAG_RECENT_FALLBACK_LIMIT, min(max_results, 6)),
                    exclude_entry_ids=seen_entry_ids,
                )

                previous_count = len(combined_results)
                combined_results = self._merge_ranked_results(
                    combined_results,
                    recent_fallback_results,
                    limit=max_combined_results,
                )
                if len(combined_results) > previous_count:
                    fallback_modes.append("recent_history")

            interaction_results = [
                result
                for result in combined_results
                if result.entry.entry_type == KnowledgeEntryType.INTERACTION
            ]
            preference_results = [
                result
                for result in combined_results
                if result.entry.entry_type in [KnowledgeEntryType.PREFERENCE, KnowledgeEntryType.USER_PREFERENCE]
            ]
            pattern_results = [
                result
                for result in combined_results
                if result.entry.entry_type in [KnowledgeEntryType.PATTERN, KnowledgeEntryType.INSIGHT]
            ]

            recent_time_entries = self._extract_recent_time_entries(interaction_results)
            if not recent_time_entries:
                fallback_time_results = await self._build_recent_history_fallback_results(
                    limit=RAG_RECENT_FALLBACK_LIMIT,
                    exclude_entry_ids=set(),
                )
                recent_time_entries = self._extract_recent_time_entries(
                    fallback_time_results,
                    limit=RAG_RECENT_FALLBACK_LIMIT,
                )
                if recent_time_entries and "recent_history" not in fallback_modes:
                    fallback_modes.append("recent_time_entries_only")

            time_window_summary = await self._build_time_window_summary(user_input)
            window_key = str(time_window_summary.get("window_key") or "unspecified")
            if time_window_summary.get("has_data") and window_key in {"today", "yesterday", "this_week"}:
                window_entries = [
                    {
                        "entry_id": item.get("entry_id"),
                        "project_name": item.get("project_name"),
                        "description": item.get("description"),
                        "duration_minutes": item.get("duration_minutes"),
                        "billable": item.get("billable"),
                        "start_time": item.get("start_time"),
                        "end_time": item.get("end_time"),
                        "created_at": item.get("start_time"),
                        "similarity": 0.5,
                    }
                    for item in time_window_summary.get("top_entries", [])
                ]
                if window_entries:
                    recent_time_entries = window_entries[: max(RAG_RECENT_FALLBACK_LIMIT, 8)]
                    if "window_summary" not in fallback_modes:
                        fallback_modes.append("window_summary")

            context_summary = self._generate_context_summary(user_input, agent_type, combined_results)
            if time_window_summary.get("has_data"):
                context_summary = (
                    f"{context_summary} "
                    f"{time_window_summary.get('window_label', 'Recent')} logged "
                    f"{time_window_summary.get('total_logged_minutes', 0)} minutes across "
                    f"{time_window_summary.get('entry_count', 0)} entries; "
                    f"idle gaps {time_window_summary.get('gap_count', 0)} "
                    f"(~{time_window_summary.get('idle_minutes', 0)} minutes)."
                ).strip()
            elif window_key in {"today", "yesterday", "this_week"}:
                context_summary = (
                    f"{context_summary} "
                    f"No tracked time entries found for {str(time_window_summary.get('window_label', 'the requested window')).lower()}."
                ).strip()

            if fallback_modes:
                context_summary = f"{context_summary} Retrieval fallback: {', '.join(fallback_modes)}."

            preference_context = [
                {
                    "content": result.entry.content,
                    "category": result.entry.category,
                    "metadata": result.entry.metadata,
                    "similarity": result.similarity_score,
                }
                for result in preference_results
            ][:5]

            if not preference_context:
                preference_context = self._build_preference_context_from_model(preferences, limit=5)
            
            # Organize results by type
            context = {
                "agent_preferences": agent_preferences,
                "relevant_interactions": [
                    {
                        "content": result.entry.content,
                        "metadata": result.entry.metadata,
                        "similarity": result.similarity_score,
                        "created_at": result.entry.created_at.isoformat(),
                        "category": self._normalize_visual_category(result.entry),
                        "is_time_entry": self._is_time_entry_entry(result.entry),
                    }
                    for result in interaction_results
                ][:6],
                "user_preferences": preference_context,
                "patterns_and_insights": [
                    {
                        "content": result.entry.content,
                        "metadata": result.entry.metadata,
                        "similarity": result.similarity_score
                    }
                    for result in pattern_results
                ][:3],
                "recent_time_entries": recent_time_entries,
                "time_window_summary": time_window_summary,
                "context_summary": context_summary
            }

            top_matches = [
                {
                    "category": self._normalize_visual_category(result.entry),
                    "type": self._normalize_visual_type(result.entry, self._normalize_visual_category(result.entry)),
                    "score": round(float(result.similarity_score), 3),
                }
                for result in combined_results[:3]
            ]

            logger.info(
                "[RAG_CONTEXT] agent=%s query=%s total=%d interactions=%d preferences=%d patterns=%d recent_time_entries=%d window=%s window_entries=%s fallback=%s top_matches=%s",
                agent_type,
                self._truncate_for_log(user_input, 150),
                len(combined_results),
                len(interaction_results),
                len(preference_results),
                len(pattern_results),
                len(recent_time_entries),
                time_window_summary.get("window_key"),
                time_window_summary.get("entry_count"),
                fallback_modes or ["none"],
                top_matches,
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get contextual knowledge: {e}")
            return {
                "agent_preferences": {},
                "relevant_interactions": [],
                "user_preferences": [],
                "patterns_and_insights": [],
                "recent_time_entries": [],
                "context_summary": "Unable to retrieve context due to system error."
            }

    def _extract_recent_time_entries(
        self,
        interaction_results: List[KnowledgeSearchResult],
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """Extract concise recent time-entry context from interaction search results."""
        extracted: List[Dict[str, Any]] = []

        for result in interaction_results:
            entry = result.entry
            if not self._is_time_entry_entry(entry):
                continue

            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            context_payload = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

            duration_minutes = context_payload.get("duration_minutes")
            if duration_minutes is None and context_payload.get("duration_seconds") is not None:
                try:
                    duration_minutes = float(context_payload.get("duration_seconds")) / 60.0
                except (TypeError, ValueError):
                    duration_minutes = None

            try:
                normalized_duration = round(max(0.0, float(duration_minutes)), 1) if duration_minutes is not None else None
            except (TypeError, ValueError):
                normalized_duration = None

            extracted.append(
                {
                    "entry_id": entry.entry_id,
                    "project_name": str(context_payload.get("project_name") or "").strip() or "Unassigned",
                    "description": str(context_payload.get("description") or context_payload.get("task_name") or "").strip(),
                    "duration_minutes": normalized_duration,
                    "billable": bool(context_payload.get("billable", False)),
                    "start_time": context_payload.get("start_time"),
                    "end_time": context_payload.get("end_time"),
                    "created_at": entry.created_at.isoformat(),
                    "similarity": result.similarity_score,
                }
            )

        extracted.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return extracted[:limit]

    def _generate_context_summary(self, user_input: str, agent_type: str, search_results: List) -> str:
        """Generate a context summary for the agent."""
        if not search_results:
            return f"No previous context found for {agent_type} requests."
        
        relevant_count = len([r for r in search_results if r.similarity_score > 0.7])
        categories = set(r.entry.category for r in search_results)
        
        summary = f"Found {len(search_results)} related entries ({relevant_count} highly relevant) "
        summary += f"across categories: {', '.join(categories)}. "
        
        # Get most recent interaction
        recent_interactions = [r for r in search_results if r.entry.entry_type == KnowledgeEntryType.INTERACTION]
        if recent_interactions:
            most_recent = max(recent_interactions, key=lambda x: x.entry.created_at)
            summary += f"Most recent similar interaction was on {most_recent.entry.created_at.strftime('%Y-%m-%d')}."

        time_entry_results = [r for r in search_results if self._is_time_entry_entry(r.entry)]
        if time_entry_results:
            most_recent_time_entry = max(time_entry_results, key=lambda x: x.entry.created_at).entry
            metadata = most_recent_time_entry.metadata if isinstance(most_recent_time_entry.metadata, dict) else {}
            context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
            project_name = str(context.get("project_name") or "Unassigned").strip()
            description = str(context.get("description") or context.get("task_name") or "work session").strip()
            summary += f" Recent tracked focus: {project_name} - {description}."
        
        return summary
    
    async def get_relevant_context(self, 
                                  query: str,
                                  agent_type: Optional[str] = None,
                                  max_results: int = 5) -> List[KnowledgeSearchResult]:
        """
        Get relevant context for an agent query using RAG.
        
        Args:
            query: The query to find context for
            agent_type: Filter by specific agent type (optional)
            max_results: Maximum number of context entries to return
            
        Returns:
            List of relevant knowledge entries
        """
        try:
            search_query = KnowledgeQuery(
                query_text=query,
                categories=[agent_type] if agent_type else None,
                limit=max_results,
                similarity_threshold=RAG_PRIMARY_SIMILARITY_THRESHOLD
            )
            
            return await self.search(search_query)
        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return []
    
    async def get_stats(self) -> KnowledgeStats:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Knowledge base statistics
        """
        try:
            all_entries = await self.get_all_entries()
            
            # Count by type
            entries_by_type = {}
            for entry_type in KnowledgeEntryType:
                entries_by_type[entry_type] = sum(1 for e in all_entries if e.entry_type == entry_type)
            
            # Count by category
            entries_by_category = {}
            for entry in all_entries:
                entries_by_category[entry.category] = entries_by_category.get(entry.category, 0) + 1
            
            # Get current LLM model for embedding info (but don't initialize if not available)
            embedding_model = "unknown"
            try:
                from ..llm import service as llm_service_module
                if llm_service_module._llm_service and llm_service_module._llm_service._initialized:
                    llm_service = llm_service_module._llm_service
                    current_provider = llm_service.get_current_provider()
                    embedding_model = f"{current_provider}_embedding" if current_provider else "unknown"
            except Exception:
                # Don't fail stats if LLM service is not available
                pass
            
            return KnowledgeStats(
                total_entries=len(all_entries),
                entries_by_type=entries_by_type,
                entries_by_category=entries_by_category,
                last_updated=max((e.updated_at for e in all_entries), default=datetime.utcnow()),
                embedding_model=embedding_model
            )
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return KnowledgeStats(
                total_entries=0,
                entries_by_type={},
                entries_by_category={},
                last_updated=datetime.utcnow(),
                embedding_model="unknown"
            )
    
    async def clear_all(self) -> bool:
        """
        Clear all entries from the knowledge base.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.clear()
            self._user_preferences = None
            self._sync_event_index = {}
            self._sync_event_index_loaded = False
            self._embedding_cache = {}
            self._embedding_cache_loaded = False
            self._embedding_provider_cooldown_until = 0.0
            self._clear_entries_from_db()
            logger.info("Cleared all knowledge base entries")
            return True
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
            return False
    
    async def get_embeddings_visualization_data(self) -> List[Dict[str, Any]]:
        """
        Get all embeddings with 3D coordinates for visualization.
        
        Returns:
            List of embedding visualization data
        """
        try:
            # Get all entries and their embeddings
            all_entries = self.vector_store.get_all_entries()
            embeddings_data = []
            
            if not all_entries:
                return []
            
            # Get embeddings from vector store
            embeddings = []
            entries_info = []
            
            for entry in all_entries:
                try:
                    embedding = await self._ensure_embedding_for_visualization_entry(entry)
                except Exception as embedding_error:
                    logger.warning(
                        "Skipping entry %s in visualization due to invalid embedding: %s",
                        entry.entry_id,
                        embedding_error,
                    )
                    continue

                if self._embedding_has_signal(embedding):
                    embeddings.append(embedding)
                    entries_info.append(entry)
            
            if not embeddings:
                return []

            import numpy as np

            embeddings_array = np.array(embeddings, dtype=float)
            if not np.isfinite(embeddings_array).all():
                raise ValueError("Embeddings contain non-finite values")

            positions_3d: np.ndarray
            pca_error_reason: Optional[str] = None

            # Primary path: PCA from full embedding space.
            if (
                embeddings_array.shape[0] >= 3
                and embeddings_array.shape[1] >= 3
                and not np.allclose(embeddings_array, embeddings_array[0], atol=1e-9)
            ):
                try:
                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=3)
                    pca_positions = pca.fit_transform(embeddings_array)
                    if not np.isfinite(pca_positions).all():
                        raise ValueError("PCA returned non-finite coordinates")

                    positions_3d = pca_positions
                    logger.info("Using PCA for dimensionality reduction")
                except Exception as pca_error:
                    pca_error_reason = str(pca_error)
                    positions_3d = np.empty((0, 3), dtype=float)
            else:
                positions_3d = np.empty((0, 3), dtype=float)

            # Strict secondary path: direct projection from real embedding dimensions.
            if positions_3d.size == 0:
                if embeddings_array.shape[1] < 3:
                    raise ValueError("Embedding dimension is below 3; cannot project to 3D")

                if pca_error_reason:
                    logger.warning("PCA projection unavailable: %s. Using direct semantic projection.", pca_error_reason)

                positions_3d = embeddings_array[:, :3].copy()
                centroid = positions_3d.mean(axis=0)
                positions_3d = positions_3d - centroid

                raw_norms = np.linalg.norm(positions_3d, axis=1)
                max_raw_norm = float(np.max(raw_norms)) if raw_norms.size > 0 else 0.0
                if max_raw_norm > 0:
                    positions_3d = (positions_3d / max_raw_norm) * 35.0

                logger.info("Using direct semantic projection from embedding dimensions")

            positions_array = np.array(positions_3d, dtype=float)

            if positions_array.shape[0] > 0:
                max_norm = float(np.max(np.linalg.norm(positions_array, axis=1)))
                if max_norm > 0:
                    positions_array = (positions_array / max_norm) * 35.0

            positions_3d = positions_array
            
            # Create visualization data with similarity connections
            for i, (entry, position) in enumerate(zip(entries_info, positions_3d)):
                # Calculate similarities to other entries for connection data
                similarities = []
                current_embedding = embeddings[i]
                
                for j, other_embedding in enumerate(embeddings):
                    if i != j:
                        # Calculate cosine similarity
                        norm_current = np.linalg.norm(current_embedding)
                        norm_other = np.linalg.norm(other_embedding)
                        denominator = norm_current * norm_other

                        if denominator <= 1e-12:
                            continue

                        similarity = float(np.dot(current_embedding, other_embedding) / denominator)

                        if not np.isfinite(similarity):
                            continue

                        if similarity > 0.45:
                            similarities.append({
                                "target_id": entries_info[j].entry_id,
                                "similarity": similarity,
                            })

                similarities.sort(key=lambda value: value["similarity"], reverse=True)
                if similarities:
                    anchor_index = min(2, len(similarities) - 1)
                    adaptive_floor = max(0.55, similarities[anchor_index]["similarity"] - 0.05)
                    similarities = [
                        edge for edge in similarities[:4]
                        if edge["similarity"] >= adaptive_floor
                    ]
                
                visualization_data = {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "content": entry.content[:200] + "..." if len(entry.content) > 200 else entry.content,
                    "category": self._normalize_visual_category(entry),
                    "entry_type": self._normalize_visual_type(entry, self._normalize_visual_category(entry)),
                    "tags": entry.tags,
                    "embedding": embeddings[i][:10] if len(embeddings[i]) > 10 else embeddings[i],  # First 10 dims for preview
                    "position_3d": position if isinstance(position, list) else position.tolist(),
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "similarities": similarities,
                }
                embeddings_data.append(visualization_data)
            
            logger.info(f"Generated visualization data for {len(embeddings_data)} embeddings")
            return embeddings_data
            
        except Exception as e:
            logger.error(f"Failed to get embeddings visualization data: {e}")
            return []
    
    async def get_embedding_details(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific embedding.
        
        Args:
            entry_id: ID of the entry to get details for
            
        Returns:
            Detailed embedding information or None if not found
        """
        try:
            entry = self.vector_store.get_entry(entry_id)
            if not entry:
                return None

            try:
                embedding = await self._ensure_embedding_for_visualization_entry(entry)
            except Exception as embedding_error:
                logger.warning(
                    "Embedding details for %s have no valid vector: %s",
                    entry_id,
                    embedding_error,
                )
                embedding = None
            
            # Find similar entries
            if embedding:
                similar_results = self.vector_store.search(
                    query_embedding=embedding,
                    k=6,  # Get 6 to exclude the entry itself
                    similarity_threshold=0.5
                )
                
                # Filter out the entry itself
                similar_entries = [
                    {
                        "entry_id": result.entry.entry_id,
                        "title": result.entry.title,
                        "similarity_score": result.similarity_score,
                        "category": self._normalize_visual_category(result.entry),
                        "entry_type": self._normalize_visual_type(
                            result.entry,
                            self._normalize_visual_category(result.entry),
                        )
                    }
                    for result in similar_results
                    if result.entry.entry_id != entry_id
                ][:5]  # Top 5 similar entries
            else:
                similar_entries = []
            
            details = {
                "entry": {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "content": entry.content,
                    "category": self._normalize_visual_category(entry),
                    "entry_type": self._normalize_visual_type(entry, self._normalize_visual_category(entry)),
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat()
                },
                "embedding_info": {
                    "has_embedding": embedding is not None,
                    "embedding_dimension": len(embedding) if embedding else 0,
                    "embedding_preview": embedding[:10] if embedding else None  # First 10 dimensions
                },
                "similar_entries": similar_entries,
                "statistics": {
                    "content_length": len(entry.content),
                    "tag_count": len(entry.tags),
                    "metadata_keys": list(entry.metadata.keys())
                }
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Failed to get embedding details for {entry_id}: {e}")
            return None

    async def get_embedding_quality_report(self) -> Dict[str, Any]:
        """Return quality diagnostics for stored embeddings so UI can expose integrity status."""
        try:
            all_entries = self.vector_store.get_all_entries()
            if not all_entries:
                return {
                    "checked_entries": 0,
                    "signal_embeddings": 0,
                    "zero_signal_embeddings": 0,
                    "coverage": 0.0,
                    "insight_total": 0,
                    "insight_signal": 0,
                    "insight_coverage": 0.0,
                    "status": "empty",
                    "avg_embedding_norm": 0.0,
                    "min_embedding_norm": 0.0,
                    "max_embedding_norm": 0.0,
                    "dimension_histogram": {},
                    "categories": [],
                    "suspicious_entries": [],
                    "sample_entries": [],
                    "checked_at": datetime.utcnow().isoformat(),
                }

            checked_entries = 0
            signal_embeddings = 0
            insight_total = 0
            insight_signal = 0
            norms: List[float] = []
            dimension_histogram: Dict[str, int] = {}
            category_stats: Dict[str, Dict[str, int]] = {}
            suspicious_entries: List[Dict[str, Any]] = []
            sample_entries: List[Dict[str, Any]] = []

            for entry in all_entries:
                embedding = self.vector_store.get_embedding(entry.entry_id)
                checked_entries += 1

                normalized_category = self._normalize_visual_category(entry)
                normalized_type = self._normalize_visual_type(entry, normalized_category)
                has_signal = self._embedding_has_signal(embedding)
                dimension = len(embedding) if embedding else 0

                dimension_histogram[str(dimension)] = dimension_histogram.get(str(dimension), 0) + 1

                category_bucket = category_stats.setdefault(normalized_category, {"total": 0, "signal": 0})
                category_bucket["total"] += 1

                if normalized_type == "insight" or normalized_category == "insight":
                    insight_total += 1

                if has_signal:
                    signal_embeddings += 1
                    category_bucket["signal"] += 1
                    if normalized_type == "insight" or normalized_category == "insight":
                        insight_signal += 1

                    resolved_norm = math.sqrt(sum(float(value) * float(value) for value in (embedding or [])))
                    norms.append(float(resolved_norm))
                else:
                    if len(suspicious_entries) < 25:
                        suspicious_entries.append({
                            "entry_id": entry.entry_id,
                            "title": entry.title,
                            "category": normalized_category,
                            "entry_type": normalized_type,
                            "created_at": entry.created_at.isoformat(),
                            "updated_at": entry.updated_at.isoformat(),
                        })

                if len(sample_entries) < 8:
                    sample_entries.append({
                        "entry_id": entry.entry_id,
                        "title": entry.title,
                        "category": normalized_category,
                        "entry_type": normalized_type,
                        "dimension": dimension,
                        "has_signal": has_signal,
                        "embedding_preview": (embedding[:8] if embedding else []),
                    })

            coverage = signal_embeddings / checked_entries if checked_entries else 0.0
            insight_coverage = insight_signal / insight_total if insight_total else 1.0
            zero_signal_embeddings = checked_entries - signal_embeddings

            if checked_entries == 0:
                status = "empty"
            elif coverage < 0.5 or insight_coverage < 0.5:
                status = "critical"
            elif coverage < 0.85 or insight_coverage < 0.85:
                status = "degraded"
            else:
                status = "healthy"

            categories = [
                {
                    "category": category,
                    "total": stats["total"],
                    "signal": stats["signal"],
                    "coverage": (stats["signal"] / stats["total"]) if stats["total"] else 0.0,
                }
                for category, stats in category_stats.items()
            ]
            categories.sort(key=lambda item: item["total"], reverse=True)

            return {
                "checked_entries": checked_entries,
                "signal_embeddings": signal_embeddings,
                "zero_signal_embeddings": zero_signal_embeddings,
                "coverage": coverage,
                "insight_total": insight_total,
                "insight_signal": insight_signal,
                "insight_coverage": insight_coverage,
                "status": status,
                "avg_embedding_norm": (sum(norms) / len(norms)) if norms else 0.0,
                "min_embedding_norm": min(norms) if norms else 0.0,
                "max_embedding_norm": max(norms) if norms else 0.0,
                "dimension_histogram": dimension_histogram,
                "categories": categories,
                "suspicious_entries": suspicious_entries,
                "sample_entries": sample_entries,
                "checked_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error("Failed to compute embedding quality report: %s", e)
            return {
                "checked_entries": 0,
                "signal_embeddings": 0,
                "zero_signal_embeddings": 0,
                "coverage": 0.0,
                "insight_total": 0,
                "insight_signal": 0,
                "insight_coverage": 0.0,
                "status": "error",
                "avg_embedding_norm": 0.0,
                "min_embedding_norm": 0.0,
                "max_embedding_norm": 0.0,
                "dimension_histogram": {},
                "categories": [],
                "suspicious_entries": [],
                "sample_entries": [],
                "checked_at": datetime.utcnow().isoformat(),
                "error": str(e),
            }

    async def rebuild_zero_signal_embeddings(self, limit: int = 0) -> Dict[str, Any]:
        """Rebuild embeddings that are missing signal and return post-repair quality metrics."""
        try:
            all_entries = self.vector_store.get_all_entries()
            candidates: List[KnowledgeEntry] = []

            for entry in all_entries:
                existing_embedding = self.vector_store.get_embedding(entry.entry_id)
                if not self._embedding_has_signal(existing_embedding):
                    candidates.append(entry)

            if limit > 0:
                candidates = candidates[:limit]

            rebuilt_count = 0
            failed_count = 0
            rebuilt_entry_ids: List[str] = []
            failed_entry_ids: List[str] = []

            for entry in candidates:
                try:
                    resolved_embedding = await self._ensure_embedding_for_visualization_entry(entry)
                    if self._embedding_has_signal(resolved_embedding):
                        rebuilt_count += 1
                        if len(rebuilt_entry_ids) < 50:
                            rebuilt_entry_ids.append(entry.entry_id)
                    else:
                        failed_count += 1
                        if len(failed_entry_ids) < 50:
                            failed_entry_ids.append(entry.entry_id)
                except Exception as repair_error:
                    logger.warning(
                        "Failed to rebuild embedding for entry %s: %s",
                        entry.entry_id,
                        repair_error,
                    )
                    failed_count += 1
                    if len(failed_entry_ids) < 50:
                        failed_entry_ids.append(entry.entry_id)

            post_repair_quality = await self.get_embedding_quality_report()
            return {
                "requested_limit": limit,
                "total_candidates": len(candidates),
                "rebuilt_count": rebuilt_count,
                "failed_count": failed_count,
                "rebuilt_entry_ids": rebuilt_entry_ids,
                "failed_entry_ids": failed_entry_ids,
                "repaired_at": datetime.utcnow().isoformat(),
                "post_repair_quality": post_repair_quality,
            }
        except Exception as e:
            logger.error("Failed to rebuild zero-signal embeddings: %s", e)
            return {
                "requested_limit": limit,
                "total_candidates": 0,
                "rebuilt_count": 0,
                "failed_count": 0,
                "rebuilt_entry_ids": [],
                "failed_entry_ids": [],
                "repaired_at": datetime.utcnow().isoformat(),
                "error": str(e),
            }


# Per-user service instances
_knowledge_base_services: Dict[str, KnowledgeBaseService] = {}


def get_knowledge_base_service(user_id: Optional[str] = None) -> KnowledgeBaseService:
    """Get a user-scoped knowledge base service instance."""
    from app.auth.user_context import get_current_user_id, normalize_user_storage_key

    resolved_user_id = normalize_user_storage_key(user_id or get_current_user_id())
    if resolved_user_id not in _knowledge_base_services:
        _knowledge_base_services[resolved_user_id] = KnowledgeBaseService(resolved_user_id)

    return _knowledge_base_services[resolved_user_id]


def reset_knowledge_base_service(user_id: Optional[str] = None) -> KnowledgeBaseService:
    """Force rebuild the user-scoped knowledge service and reload persisted vector data."""
    from app.auth.user_context import get_current_user_id, normalize_user_storage_key
    from .vector_store import reset_vector_store

    resolved_user_id = normalize_user_storage_key(user_id or get_current_user_id())

    # Reload vector index from disk first, then rebuild service-level caches.
    reset_vector_store(resolved_user_id)
    _knowledge_base_services.pop(resolved_user_id, None)
    _knowledge_base_services[resolved_user_id] = KnowledgeBaseService(resolved_user_id)

    return _knowledge_base_services[resolved_user_id]