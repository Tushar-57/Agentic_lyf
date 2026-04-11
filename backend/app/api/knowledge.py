"""
API endpoints for knowledge base operations.
"""

import logging
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from html import escape
from typing import List, Optional, Dict, Any, Set
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.models.knowledge import (
    KnowledgeEntry,
    KnowledgeEntryType,
    KnowledgeQuery,
    KnowledgeSearchResult,
    KnowledgeStats,
    UserPreferences,
    KnowledgeEntrySubType
)
from app.services.knowledge_base import get_knowledge_base_service
from app.services.knowledge_base import reset_knowledge_base_service
from app.services.checkup_store import DailyCheckupRecord, get_daily_checkup_store
from app.services.ai_notifications_store import AINotificationRecord, get_ai_notification_store
from app.auth.user_context import get_current_user

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)


class CreateEntryRequest(BaseModel):
    """Request model for creating knowledge entries."""
    entry_type: KnowledgeEntryType
    entry_sub_type: KnowledgeEntrySubType
    category: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class UpdateEntryRequest(BaseModel):
    """Request model for updating knowledge entries."""
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class SearchRequest(BaseModel):
    """Request model for searching knowledge base."""
    query_text: str
    categories: Optional[List[str]] = None
    entry_types: Optional[List[KnowledgeEntryType]] = None
    tags: Optional[List[str]] = None
    limit: int = 10
    similarity_threshold: float = 0.7


class InteractionHistoryRequest(BaseModel):
    """Request model for adding interaction history."""
    agent_type: str
    user_input: str
    agent_response: str
    context: Optional[Dict[str, Any]] = None


class NotificationAcknowledgeRequest(BaseModel):
    """Request payload for acknowledging/unacknowledging notifications."""
    acknowledged: bool = True


class AINotificationResponse(BaseModel):
    """Response model for AI notification entries."""
    id: int
    notification_key: str
    kind: str
    severity: str
    status: str
    title: str
    summary: str
    details: Optional[str] = None
    score: Optional[float] = None
    recommended_actions: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    first_seen_at: str
    last_seen_at: str
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    updated_at: str


@router.post("/entries", response_model=KnowledgeEntry)
async def create_entry(request: CreateEntryRequest):
    """Create a new knowledge base entry."""
    try:
        kb_service = get_knowledge_base_service()
        entry = await kb_service.create_entry(
            entry_type=request.entry_type,
            category=request.category,
            entry_sub_type=request.entry_sub_type,
            title=request.title,
            content=request.content,
            metadata=request.metadata,
            tags=request.tags
        )
        return entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create entry: {str(e)}")


@router.get("/entries/{entry_id}", response_model=Optional[KnowledgeEntry])
async def get_entry(entry_id: str):
    """Get a knowledge entry by ID."""
    try:
        kb_service = get_knowledge_base_service()
        entry = await kb_service.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entry: {str(e)}")


@router.put("/entries/{entry_id}", response_model=Optional[KnowledgeEntry])
async def update_entry(entry_id: str, request: UpdateEntryRequest):
    """Update a knowledge entry."""
    try:
        kb_service = get_knowledge_base_service()
        entry = await kb_service.update_entry(
            entry_id=entry_id,
            title=request.title,
            content=request.content,
            metadata=request.metadata,
            tags=request.tags
        )
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update entry: {str(e)}")


@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str):
    """Delete a knowledge entry."""
    try:
        kb_service = get_knowledge_base_service()
        success = await kb_service.delete_entry(entry_id)
        if not success:
            raise HTTPException(status_code=404, detail="Entry not found")
        return {"message": "Entry deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete entry: {str(e)}")


@router.post("/search", response_model=List[KnowledgeSearchResult])
async def search_knowledge_base(request: SearchRequest):
    """Search the knowledge base using RAG."""
    try:
        kb_service = get_knowledge_base_service()
        query = KnowledgeQuery(
            query_text=request.query_text,
            categories=request.categories,
            entry_types=request.entry_types,
            tags=request.tags,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        results = await kb_service.search(query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search knowledge base: {str(e)}")


@router.get("/entries", response_model=List[KnowledgeEntry])
async def get_all_entries(
    category: Optional[str] = Query(None, description="Filter by category"),
    entry_type: Optional[KnowledgeEntryType] = Query(None, description="Filter by entry type")
):
    """Get all knowledge entries with optional filters."""
    try:
        kb_service = get_knowledge_base_service()
        await _sync_missing_db_checkup_insights(kb_service)
        entries = await kb_service.get_all_entries(category=category, entry_type=entry_type)
        entries = _merge_db_checkup_entries(entries, category=category, entry_type=entry_type)
        entries = _dedupe_external_sync_entries(entries)
        entries = _prune_legacy_system_preference_entries(entries)
        return entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entries: {str(e)}")


@router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences():
    """Get user preferences."""
    try:
        kb_service = get_knowledge_base_service()
        preferences = await kb_service.get_user_preferences()
        return preferences
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")


@router.put("/preferences", response_model=Dict[str, bool])
async def update_user_preferences(preferences: UserPreferences):
    """Update user preferences."""
    try:
        kb_service = get_knowledge_base_service()
        success = await kb_service.update_user_preferences(preferences)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


class AddPreferenceRequest(BaseModel):
    """Request model for adding new preferences."""
    category: str
    key: str
    value: Any
    description: Optional[str] = None


@router.post("/preferences/add", response_model=Dict[str, bool])
async def add_user_preference(request: AddPreferenceRequest):
    """Add a new user preference."""
    try:
        kb_service = get_knowledge_base_service()
        success = await kb_service.add_user_preference(
            category=request.category,
            key=request.key,
            value=request.value,
            description=request.description
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add preference: {str(e)}")


@router.delete("/preferences/{category}/{key}", response_model=Dict[str, bool])
async def remove_user_preference(category: str, key: str):
    """Remove a user preference."""
    try:
        kb_service = get_knowledge_base_service()
        success = await kb_service.remove_user_preference(category, key)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove preference: {str(e)}")


@router.get("/preferences/categories", response_model=List[str])
async def get_preference_categories():
    """Get all available preference categories."""
    try:
        kb_service = get_knowledge_base_service()
        categories = await kb_service.get_preference_categories()
        return categories
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preference categories: {str(e)}")


@router.post("/interactions", response_model=KnowledgeEntry)
async def add_interaction_history(request: InteractionHistoryRequest):
    """Add an interaction to the history."""
    try:
        kb_service = get_knowledge_base_service()
        entry = await kb_service.add_interaction_history(
            agent_type=request.agent_type,
            user_input=request.user_input,
            agent_response=request.agent_response,
            context=request.context
        )
        return entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add interaction history: {str(e)}")


@router.get("/context", response_model=List[KnowledgeSearchResult])
async def get_relevant_context(
    query: str = Query(..., description="Query to find relevant context for"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    max_results: int = Query(5, description="Maximum number of results")
):
    """Get relevant context for an agent query using RAG."""
    try:
        kb_service = get_knowledge_base_service()
        context = await kb_service.get_relevant_context(
            query=query,
            agent_type=agent_type,
            max_results=max_results
        )
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get relevant context: {str(e)}")


@router.get("/stats", response_model=KnowledgeStats)
async def get_knowledge_stats():
    """Get knowledge base statistics."""
    try:
        kb_service = get_knowledge_base_service()
        await _sync_missing_db_checkup_insights(kb_service)
        stats = await kb_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/refresh")
async def force_refresh_knowledge_state():
    """Force reload user-scoped knowledge data from persisted storage files."""
    try:
        request_user = get_current_user()
        kb_service = reset_knowledge_base_service()
        await _sync_missing_db_checkup_insights(kb_service)
        stats = await kb_service.get_stats()

        return {
            "success": True,
            "user_scope": {
                "storage_key": request_user.storage_key,
                "source": request_user.source,
                "authenticated": request_user.authenticated,
            },
            "stats": {
                "total_entries": stats.total_entries,
                "entries_by_category": stats.entries_by_category,
                "entries_by_type": stats.entries_by_type,
                "last_updated": stats.last_updated.isoformat(),
                "embedding_model": stats.embedding_model,
            },
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh knowledge state: {str(e)}")


@router.delete("/clear")
async def clear_knowledge_base():
    """Clear all entries from the knowledge base."""
    try:
        kb_service = get_knowledge_base_service()
        success = await kb_service.clear_all()
        return {"success": success, "message": "Knowledge base cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")


class EmbeddingVisualizationData(BaseModel):
    """Data model for embedding visualization."""
    class SimilarityEdge(BaseModel):
        target_id: str
        similarity: float

    entry_id: str
    title: str
    content: str
    category: str
    entry_type: str
    tags: List[str]
    embedding: List[float]
    position_3d: List[float]  # [x, y, z] coordinates for 3D visualization
    created_at: str
    updated_at: str
    similarities: List[SimilarityEdge] = Field(default_factory=list)


@router.get("/embeddings/visualization", response_model=List[EmbeddingVisualizationData])
async def get_embeddings_for_visualization():
    """Get all embeddings with 3D coordinates for visualization."""
    try:
        kb_service = get_knowledge_base_service()
        sync_report = await _sync_missing_db_checkup_insights(kb_service)
        if sync_report["created"] > 0:
            logger.info(
                "Materialized %s missing checkup insights before visualization (records=%s failed=%s)",
                sync_report["created"],
                sync_report["records"],
                sync_report["failed"],
            )
        visualization_data = await kb_service.get_embeddings_visualization_data()
        return visualization_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embeddings visualization data: {str(e)}")


@router.get("/embeddings/{entry_id}/details")
async def get_embedding_details(entry_id: str):
    """Get detailed information about a specific embedding."""
    try:
        kb_service = get_knowledge_base_service()
        details = await kb_service.get_embedding_details(entry_id)
        if not details:
            raise HTTPException(status_code=404, detail="Embedding not found")
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding details: {str(e)}")


@router.get("/embeddings/quality")
async def get_embeddings_quality_report():
    """Return diagnostics about embedding integrity and semantic signal coverage."""
    try:
        kb_service = get_knowledge_base_service()
        await _sync_missing_db_checkup_insights(kb_service)
        return await kb_service.get_embedding_quality_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding quality report: {str(e)}")


@router.post("/embeddings/quality/rebuild")
async def rebuild_zero_signal_embeddings(
    limit: int = Query(0, ge=0, le=500, description="Max zero-signal entries to rebuild (0 = all)")
):
    """Rebuild missing/zero-signal embeddings and return refreshed quality diagnostics."""
    try:
        kb_service = get_knowledge_base_service()
        await _sync_missing_db_checkup_insights(kb_service)
        return await kb_service.rebuild_zero_signal_embeddings(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild zero-signal embeddings: {str(e)}")


class OnboardingData(BaseModel):
    """Model for onboarding data."""
    role: str
    goals: List[Dict[str, Any]]
    preferences: List[str]
    mentor: Dict[str, Any]
    planner: Dict[str, Any]
    preferredTone: Optional[str] = None
    coach_preferences: Optional[Dict[str, Any]] = None
    domain_preferences: Optional[Dict[str, Dict[str, Any]]] = None
    preference_profile: Optional[Dict[str, Dict[str, Any]]] = None
    coachPreferences: Optional[Dict[str, Any]] = None
    domainPreferences: Optional[Dict[str, Dict[str, Any]]] = None
    preferenceProfile: Optional[Dict[str, Dict[str, Any]]] = None


class DailyCheckupRequest(BaseModel):
    """Request model for morning/evening checkup APIs."""
    date: Optional[str] = None
    timezone: Optional[str] = None
    note: Optional[str] = None
    perspective: Optional[Dict[str, Any]] = None
    context_snapshot: Optional[Dict[str, Any]] = None
    contextSnapshot: Optional[Dict[str, Any]] = None


def _resolve_date_range(time_range: str) -> int:
    """Resolve time range token to number of days."""
    range_map = {
        "7d": 7,
        "30d": 30,
        "90d": 90,
    }
    return range_map.get(time_range, 30)


def _format_iso_date(dt: datetime) -> str:
    """Format datetime as ISO date string (YYYY-MM-DD)."""
    return dt.date().isoformat()


def _week_bucket_label(dt: datetime) -> str:
    """Build a stable weekly bucket label."""
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _normalize_entry_category(entry: KnowledgeEntry) -> str:
    """Normalize category labels so synced time entries surface consistently."""
    category = str(entry.category or "uncategorized").strip().lower()
    metadata = entry.metadata or {}
    context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

    source = str(context.get("source", "")).strip().lower()
    source_action = str(context.get("source_action", "")).strip().lower()
    has_time_entry_id = context.get("time_entry_id") is not None

    if (
        category == "time_entry"
        or source == "alterego_timetracker"
        or "time_entry" in source_action
        or has_time_entry_id
    ):
        return "time_entry"

    return category if category else "uncategorized"


def _ensure_timezone(value: datetime) -> datetime:
    """Normalize datetimes to timezone-aware UTC values."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _sanitize_timezone_name(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    return normalized or None


def _resolve_timezone(raw_value: Optional[str]):
    timezone_name = _sanitize_timezone_name(raw_value)
    if not timezone_name:
        return timezone.utc

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Unknown timezone '%s'; falling back to UTC", timezone_name)
        return timezone.utc


def _to_timezone(value: datetime, tzinfo) -> datetime:
    return _ensure_timezone(value).astimezone(tzinfo)


def _extract_preferences_timezone(preferences: Optional[UserPreferences]) -> Optional[str]:
    if not preferences:
        return None

    if isinstance(preferences.general, dict):
        resolved = _sanitize_timezone_name(preferences.general.get("timezone"))
        if resolved:
            return resolved

    if isinstance(preferences.productivity, dict):
        resolved = _sanitize_timezone_name(preferences.productivity.get("timezone"))
        if resolved:
            return resolved

    return None


def _resolve_checkup_timezone(
    request_timezone: Optional[str],
    context_snapshot: Optional[Dict[str, Any]],
    preferences: Optional[UserPreferences],
) -> Optional[str]:
    request_tz = _sanitize_timezone_name(request_timezone)
    if request_tz:
        return request_tz

    if isinstance(context_snapshot, dict):
        context_tz = _sanitize_timezone_name(context_snapshot.get("timezone"))
        if context_tz:
            return context_tz

    return _extract_preferences_timezone(preferences)


def _entry_context(entry: KnowledgeEntry) -> Dict[str, Any]:
    metadata = entry.metadata or {}
    context = metadata.get("context")
    if isinstance(context, dict):
        return context
    return {}


def _parse_context_datetime(raw_value: Any) -> Optional[datetime]:
    """Parse datetime values stored in context payloads."""
    if raw_value is None:
        return None

    if isinstance(raw_value, datetime):
        return _ensure_timezone(raw_value)

    if not isinstance(raw_value, str):
        return None

    candidate = raw_value.strip()
    if not candidate:
        return None

    normalized_candidate = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate

    try:
        parsed = datetime.fromisoformat(normalized_candidate)
    except ValueError:
        return None

    return _ensure_timezone(parsed)


def _resolve_entry_event_timestamp(entry: KnowledgeEntry, fallback_ts: Optional[datetime] = None) -> datetime:
    """Resolve the semantic timestamp for analytics and checkups."""
    base_timestamp = _ensure_timezone(fallback_ts or entry.created_at)

    if _normalize_entry_category(entry) != "time_entry":
        return base_timestamp

    context = _entry_context(entry)
    for key in ("start_time", "end_time", "timestamp"):
        parsed = _parse_context_datetime(context.get(key))
        if parsed:
            return parsed

    return base_timestamp


def _safe_float(raw_value: Any, default: float = 0.0) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _safe_int(raw_value: Any, default: int = 0) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _parse_requested_date(date_token: Optional[str], timezone_name: Optional[str] = None) -> date:
    """Parse YYYY-MM-DD dates from API payloads."""
    if not date_token:
        return datetime.now(_resolve_timezone(timezone_name)).date()

    try:
        return date.fromisoformat(date_token)
    except ValueError as parse_error:
        raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD") from parse_error


def _format_minutes(total_minutes: float) -> str:
    normalized_minutes = max(0, int(round(total_minutes)))
    hours = normalized_minutes // 60
    minutes = normalized_minutes % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalized_string_list(raw_value: Any) -> List[str]:
    if not isinstance(raw_value, list):
        return []

    normalized: List[str] = []
    for item in raw_value:
        item_text = _normalized_text(item)
        if item_text:
            normalized.append(item_text)
    return normalized


def _strip_html_tags(value: str) -> str:
    """Convert HTML fragments to plain text for storage/search compatibility."""
    if not value:
        return ""

    stripped = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", value)
    stripped = re.sub(r"(?i)<br\s*/?>", "\n", stripped)
    stripped = re.sub(r"(?i)</p\s*>", "\n", stripped)
    stripped = re.sub(r"(?i)<li\b[^>]*>", "\n- ", stripped)
    stripped = re.sub(r"<[^>]+>", "", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _looks_like_html(value: str) -> bool:
    if not value:
        return False
    return bool(re.search(r"</?[a-zA-Z][^>]*>", value))


def _is_structured_morning_checkup_html(value: str) -> bool:
    """Validate that generated morning checkup HTML contains required semantic sections."""
    if not _looks_like_html(value):
        return False

    normalized = value.lower()
    required_tokens = [
        "daily-checkup",
        "dc-header",
        "dc-metrics",
        "daily-schedule",
        "dc-timeline",
        "execution-notes",
        "dc-journal",
        "dc-block",
    ]
    return all(token in normalized for token in required_tokens)


def _is_structured_evening_checkup_html(value: str) -> bool:
    """Validate that generated evening checkup HTML contains required semantic sections."""
    if not _looks_like_html(value):
        return False

    normalized = value.lower()
    required_tokens = [
        "daily-checkup",
        "evening-checkup",
        "dc-header",
        "dc-metrics",
        "daily-schedule",
        "dc-timeline",
        "execution-notes",
        "dc-journal",
        "dc-block",
    ]
    return all(token in normalized for token in required_tokens)


def _parse_hhmm_to_minutes(raw_value: str, default_minutes: int) -> int:
    match = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", raw_value or "")
    if not match:
        return default_minutes

    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return default_minutes
    return hour * 60 + minute


def _parse_work_hours_range(work_hours: str) -> tuple[int, int]:
    default_start = 9 * 60
    default_end = 17 * 60

    if not isinstance(work_hours, str) or "-" not in work_hours:
        return default_start, default_end

    start_token, end_token = work_hours.split("-", 1)
    start_minutes = _parse_hhmm_to_minutes(start_token, default_start)
    end_minutes = _parse_hhmm_to_minutes(end_token, default_end)

    if end_minutes <= start_minutes:
        end_minutes = min(24 * 60, start_minutes + 8 * 60)

    if end_minutes - start_minutes < 120:
        end_minutes = min(24 * 60, start_minutes + 120)

    return max(0, start_minutes), min(24 * 60, end_minutes)


def _minutes_to_hhmm(total_minutes: int) -> str:
    normalized = max(0, min(24 * 60, int(total_minutes)))
    return f"{normalized // 60:02d}:{normalized % 60:02d}"


def _minutes_to_display(total_minutes: int) -> str:
    normalized = max(0, min(24 * 60, int(total_minutes)))
    hour = normalized // 60
    minute = normalized % 60
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12}:{minute:02d} {period}"


def _build_morning_schedule_blocks(
    *,
    focus_target: str,
    focus_task_titles: List[str],
    work_hours: str,
    check_in_time: str,
    schedule_end_time: str,
    run_anchor_time: Optional[str],
    planned_deep_work_minutes: float,
    overdue_tasks: int,
    due_today_tasks: int,
    habits_total: int,
    habits_completed_today: int,
    avg_daily_minutes: float,
) -> List[Dict[str, Any]]:
    start_minutes, end_minutes = _parse_work_hours_range(work_hours)
    check_in_minutes = _parse_hhmm_to_minutes(check_in_time, start_minutes)
    schedule_end_minutes = _parse_hhmm_to_minutes(schedule_end_time, end_minutes)
    runtime_anchor_minutes = (
        _parse_hhmm_to_minutes(run_anchor_time, check_in_minutes)
        if isinstance(run_anchor_time, str) and run_anchor_time.strip()
        else check_in_minutes
    )

    if schedule_end_minutes <= start_minutes:
        schedule_end_minutes = min(24 * 60, start_minutes + 12 * 60)

    if schedule_end_minutes - start_minutes < 120:
        schedule_end_minutes = min(24 * 60, start_minutes + 120)

    # If a user runs morning checkup late, start from "now" instead of preferred check-in.
    if runtime_anchor_minutes > check_in_minutes:
        check_in_minutes = runtime_anchor_minutes

    cursor = max(start_minutes, min(schedule_end_minutes - 15, check_in_minutes))
    end_minutes = max(cursor + 15, schedule_end_minutes)

    schedule_blocks: List[Dict[str, Any]] = []

    def add_block(duration: int, title: str, reason: str, priority: str = "medium") -> None:
        nonlocal cursor
        if cursor >= end_minutes:
            return

        safe_duration = max(10, int(duration))
        block_start = cursor
        block_end = min(end_minutes, block_start + safe_duration)
        if block_end - block_start < 10:
            return

        schedule_blocks.append(
            {
                "start": _minutes_to_hhmm(block_start),
                "end": _minutes_to_hhmm(block_end),
                "start_label": _minutes_to_display(block_start),
                "end_label": _minutes_to_display(block_end),
                "title": title,
                "reason": reason,
                "priority": priority,
            }
        )
        cursor = block_end

    add_block(15, "Check-in and intent setup", "Anchor your day before context switching.", "high")
    add_block(25, "Plan top priorities", "Lock the execution sequence around outcomes and constraints.", "high")

    if overdue_tasks > 0 or due_today_tasks > 0:
        add_block(
            30,
            "Deadline triage",
            f"Resolve urgency first ({overdue_tasks} overdue, {due_today_tasks} due today).",
            "high",
        )

    deep_work_seed = planned_deep_work_minutes if planned_deep_work_minutes > 0 else max(90.0, avg_daily_minutes * 0.45)
    deep_work_target = int(max(60, min(240, round(deep_work_seed))))

    focus_pool = [task for task in focus_task_titles if task] or ([focus_target] if focus_target else [])
    primary_focus = focus_pool[0] if focus_pool else "Top-priority execution block"
    secondary_focus = focus_pool[1] if len(focus_pool) > 1 else focus_target or primary_focus

    first_block = min(120, max(50, deep_work_target // 2 if deep_work_target > 100 else deep_work_target))
    remaining_deep_work = max(0, deep_work_target - first_block)

    add_block(first_block, f"Deep work: {primary_focus}", "Protect focused time for highest-leverage progress.", "high")

    if remaining_deep_work >= 40:
        add_block(15, "Reset break", "Short reset to sustain decision quality.", "medium")
        add_block(remaining_deep_work, f"Deep work: {secondary_focus}", "Advance the next critical task before reactive work.", "high")

    if habits_total > 0:
        add_block(
            15,
            "Habit anchor",
            f"Maintain consistency ({habits_completed_today}/{habits_total} habits complete today).",
            "medium",
        )

    remaining_minutes = end_minutes - cursor
    if remaining_minutes >= 20:
        add_block(
            min(45, remaining_minutes),
            "Admin, comms, and contingency buffer",
            "Absorb interruptions without breaking deep-work outcomes.",
            "medium",
        )

    return schedule_blocks


def _build_morning_schedule_html(
    *,
    checkup_date: date,
    focus_target: str,
    fallback_lines: List[str],
    schedule_blocks: List[Dict[str, Any]],
) -> str:
    total_scheduled_minutes = 0
    high_priority_blocks = 0

    for block in schedule_blocks:
        block_start = _parse_hhmm_to_minutes(str(block.get("start", "")), 0)
        block_end = _parse_hhmm_to_minutes(str(block.get("end", "")), block_start)
        total_scheduled_minutes += max(0, block_end - block_start)

        if str(block.get("priority", "")).strip().lower() == "high":
            high_priority_blocks += 1

    if not schedule_blocks:
        schedule_blocks = [
            {
                "start": "09:00",
                "end": "09:45",
                "start_label": "9:00 AM",
                "end_label": "9:45 AM",
                "title": focus_target or "Primary focus block",
                "reason": "Start with the highest-leverage task before reactive work.",
                "priority": "high",
            }
        ]
        total_scheduled_minutes = 45
        high_priority_blocks = 1

    block_items = "".join(
        [
            (
                f"<li class=\"dc-block dc-block--{escape(str(block.get('priority', 'medium')).strip().lower() or 'medium')}\">"
                "<div class=\"dc-time-wrap\">"
                f"<span class=\"dc-time\">{escape(str(block.get('start_label', '')))} - {escape(str(block.get('end_label', '')))}</span>"
                f"<span class=\"dc-priority\">{escape(str(block.get('priority', 'medium')).strip().title() or 'Medium')} Priority</span>"
                "</div>"
                "<div class=\"dc-block-copy\">"
                f"<p class=\"dc-block-title\">{escape(str(block.get('title', 'Focus Block')))}</p>"
                f"<p class=\"dc-block-reason\">{escape(str(block.get('reason', '')))}</p>"
                "</div>"
                "</li>"
            )
            for block in schedule_blocks
        ]
    )

    highlights = "".join([f"<li>{escape(line)}</li>" for line in fallback_lines[:6]])

    return (
        "<section class=\"daily-checkup\">"
        "<header class=\"dc-header\">"
        "<div class=\"dc-badge-row\">"
        "<span class=\"dc-kicker\">Morning Checkup</span>"
        f"<span class=\"dc-date\">{escape(checkup_date.isoformat())}</span>"
        "</div>"
        f"<h3 class=\"dc-focus\">Primary Focus: {escape(focus_target or 'Execute the highest-priority task first')}</h3>"
        "<p class=\"dc-subtitle\">Built from your goals, priorities, deadlines, habits, and tracked time context.</p>"
        "</header>"
        "<section class=\"dc-metrics\">"
        f"<div class=\"dc-metric\"><p class=\"dc-metric-label\">Scheduled Blocks</p><p class=\"dc-metric-value\">{len(schedule_blocks)}</p></div>"
        f"<div class=\"dc-metric\"><p class=\"dc-metric-label\">High Priority</p><p class=\"dc-metric-value\">{high_priority_blocks}</p></div>"
        f"<div class=\"dc-metric\"><p class=\"dc-metric-label\">Planned Duration</p><p class=\"dc-metric-value\">{escape(_format_minutes(total_scheduled_minutes))}</p></div>"
        "</section>"
        "<section class=\"daily-schedule dc-panel\">"
        "<div class=\"dc-panel-head\">"
        "<p class=\"dc-panel-title\">Time-Blocked Plan</p>"
        "<p class=\"dc-panel-subtitle\">Protect deep work first, then absorb reactive work with intent.</p>"
        "</div>"
        f"<ol class=\"dc-timeline\">{block_items}</ol>"
        "</section>"
        "<section class=\"execution-notes dc-panel\">"
        "<p class=\"dc-panel-title\">Execution Notes</p>"
        f"<ul class=\"dc-notes\">{highlights}</ul>"
        "</section>"
        "<section class=\"journal dc-panel dc-journal\">"
        "<p class=\"dc-panel-title\">Accountability + Journal</p>"
        "<p class=\"dc-journal-q\">Accountability: Which schedule block will you protect first if your day compresses?</p>"
        "<p class=\"dc-journal-q\">Journal prompt: What one behavior makes today a win even if everything else changes?</p>"
        "</section>"
        "</section>"
    )


def _build_evening_reflection_html(
    *,
    checkup_date: date,
    recap_line: str,
    total_minutes: float,
    billable_minutes: float,
    performance_score: float,
    avg_focus: Optional[float],
    avg_energy: Optional[float],
    wins: List[str],
    blockers: List[str],
    tomorrow_focus: List[str],
    focus_task_titles: List[str],
    top_projects: List[str],
) -> str:
    focus_energy_label = (
        f"{round(avg_focus, 1)}/10 focus - {round(avg_energy, 1)}/10 energy"
        if avg_focus is not None and avg_energy is not None
        else f"{round(avg_focus, 1)}/10 focus"
        if avg_focus is not None
        else f"{round(avg_energy, 1)}/10 energy"
        if avg_energy is not None
        else "n/a"
    )

    normalized_wins = wins[:3] if wins else ["You maintained execution momentum by reflecting on your day."]
    normalized_blockers = blockers[:2]
    normalized_tomorrow = tomorrow_focus[:4] if tomorrow_focus else ["Define tomorrow's top task before you start."]

    action_items = "".join(
        [
            (
                f"<li class=\"dc-block dc-block--{'high' if index == 0 else 'medium'}\">"
                "<div class=\"dc-time-wrap\">"
                f"<span class=\"dc-time\">Action {index + 1}</span>"
                f"<span class=\"dc-priority\">{'High' if index == 0 else 'Medium'} Priority</span>"
                "</div>"
                "<div class=\"dc-block-copy\">"
                f"<p class=\"dc-block-title\">{escape(item)}</p>"
                "<p class=\"dc-block-reason\">"
                "Tie this to deadline pressure, habit consistency, or deep-work protection."
                "</p>"
                "</div>"
                "</li>"
            )
            for index, item in enumerate(normalized_tomorrow)
        ]
    )

    note_lines = [f"Win: {item}" for item in normalized_wins]
    if normalized_blockers:
        note_lines.extend([f"Friction to reduce: {item}" for item in normalized_blockers])
    if focus_task_titles:
        note_lines.append(f"Focus tasks reviewed: {', '.join(focus_task_titles[:3])}")
    if top_projects:
        note_lines.append(f"Project momentum: {', '.join(top_projects[:2])}")
    notes_html = "".join([f"<li>{escape(line)}</li>" for line in note_lines])

    return (
        "<section class=\"daily-checkup evening-checkup\">"
        "<header class=\"dc-header\">"
        "<div class=\"dc-badge-row\">"
        "<span class=\"dc-kicker\">Evening Checkup</span>"
        f"<span class=\"dc-date\">{escape(checkup_date.isoformat())}</span>"
        "</div>"
        f"<h3 class=\"dc-focus\">{escape(recap_line)}</h3>"
        "<p class=\"dc-subtitle\">Review today's outcomes, extract evidence, and lock tomorrow's first actions.</p>"
        "</header>"
        "<section class=\"dc-metrics\">"
        f"<div class=\"dc-metric\"><p class=\"dc-metric-label\">Logged Time</p><p class=\"dc-metric-value\">{escape(_format_minutes(total_minutes))}</p></div>"
        f"<div class=\"dc-metric\"><p class=\"dc-metric-label\">Billable Time</p><p class=\"dc-metric-value\">{escape(_format_minutes(billable_minutes))}</p></div>"
        f"<div class=\"dc-metric\"><p class=\"dc-metric-label\">Performance</p><p class=\"dc-metric-value\">{escape(str(round(performance_score, 1)))}/10</p></div>"
        "</section>"
        "<section class=\"daily-schedule dc-panel\">"
        "<div class=\"dc-panel-head\">"
        "<p class=\"dc-panel-title\">Tomorrow Commitments</p>"
        f"<p class=\"dc-panel-subtitle\">Focus and energy signal: {escape(focus_energy_label)}</p>"
        "</div>"
        f"<ol class=\"dc-timeline\">{action_items}</ol>"
        "</section>"
        "<section class=\"execution-notes dc-panel\">"
        "<p class=\"dc-panel-title\">Wins + Friction</p>"
        f"<ul class=\"dc-notes\">{notes_html}</ul>"
        "</section>"
        "<section class=\"journal dc-panel dc-journal\">"
        "<p class=\"dc-panel-title\">Reflection + Accountability</p>"
        "<p class=\"dc-journal-q\">Reflection: What moved your day forward the most, and why?</p>"
        "<p class=\"dc-journal-q\">Accountability: What is the first 30-minute block you will protect tomorrow?</p>"
        "</section>"
        "</section>"
    )


def _extract_communication_profile(all_entries: List[KnowledgeEntry]) -> Dict[str, Any]:
    latest_profile_entry: Optional[KnowledgeEntry] = None
    for entry in all_entries:
        if (
            entry.entry_type == KnowledgeEntryType.USER_PREFERENCE
            and entry.entry_sub_type == KnowledgeEntrySubType.USER_PROFILE
        ):
            if (
                latest_profile_entry is None
                or _ensure_timezone(entry.updated_at) > _ensure_timezone(latest_profile_entry.updated_at)
            ):
                latest_profile_entry = entry

    if latest_profile_entry is None:
        return {
            "role": "",
            "preferred_tone": "",
            "mentor_name": "",
            "mentor_archetype": "",
            "mentor_style": "",
            "preferences": [],
        }

    metadata = latest_profile_entry.metadata if isinstance(latest_profile_entry.metadata, dict) else {}
    mentor = metadata.get("mentor") if isinstance(metadata.get("mentor"), dict) else {}

    return {
        "role": _normalized_text(metadata.get("role")),
        "preferred_tone": _normalized_text(
            metadata.get("preferredTone")
            or metadata.get("preferred_tone")
            or mentor.get("tone")
        ),
        "mentor_name": _normalized_text(mentor.get("name")),
        "mentor_archetype": _normalized_text(mentor.get("archetype")),
        "mentor_style": _normalized_text(mentor.get("style")),
        "preferences": _normalized_string_list(metadata.get("preferences")),
    }


def _build_style_directive(profile: Dict[str, Any], checkup_type: str) -> str:
    fragments: List[str] = [f"This is a {checkup_type} checkup response."]

    mentor_name = _normalized_text(profile.get("mentor_name"))
    role = _normalized_text(profile.get("role"))
    preferred_tone = _normalized_text(profile.get("preferred_tone"))
    mentor_archetype = _normalized_text(profile.get("mentor_archetype"))
    mentor_style = _normalized_text(profile.get("mentor_style"))
    priorities = _normalized_string_list(profile.get("preferences"))

    if mentor_name:
        fragments.append(f"Coach identity: {mentor_name}.")
    if role:
        fragments.append(f"User context role: {role}.")
    if preferred_tone:
        fragments.append(f"Required tone: {preferred_tone}.")
    if mentor_archetype:
        fragments.append(f"Mentor archetype to mirror: {mentor_archetype}.")
    if mentor_style:
        fragments.append(f"Delivery style to mirror: {mentor_style}.")
    if priorities:
        fragments.append(f"Priorities to respect: {', '.join(priorities[:4])}.")

    fragments.append("Stay practical, concrete, and avoid generic coaching language.")
    return " ".join(fragments)


def _build_fallback_checkup_message(lines: List[str], profile: Dict[str, Any], checkup_type: str) -> str:
    mentor_name = _normalized_text(profile.get("mentor_name")) or "Coach"
    tone = _normalized_text(profile.get("preferred_tone")).lower()
    mentor_archetype = _normalized_text(profile.get("mentor_archetype"))
    mentor_style = _normalized_text(profile.get("mentor_style"))

    if "direct" in tone or "blunt" in tone or "concise" in tone:
        opener = f"{mentor_name} ({checkup_type.title()}): Straight plan."
        closer = "Choose the first action and execute it now."
    elif "warm" in tone or "friendly" in tone or "empathetic" in tone or "supportive" in tone:
        opener = f"{mentor_name} ({checkup_type.title()}): You are making progress."
        closer = "Pick one next step now and keep your momentum."
    elif "formal" in tone or "professional" in tone:
        opener = f"{mentor_name} ({checkup_type.title()}): Structured guidance."
        closer = "Confirm the first priority and time-block it immediately."
    else:
        opener = f"{mentor_name} ({checkup_type.title()}):"
        closer = "Commit to the first action before your next context switch."

    style_cues = ", ".join([cue for cue in [mentor_archetype, mentor_style] if cue])
    parts = [opener]
    if style_cues:
        parts.append(f"Style cues: {style_cues}")
    parts.extend([f"- {line}" for line in lines])
    parts.append(closer)
    return "\n".join(parts)


def _public_style_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "preferred_tone": _normalized_text(profile.get("preferred_tone")) or None,
        "mentor_name": _normalized_text(profile.get("mentor_name")) or None,
        "mentor_archetype": _normalized_text(profile.get("mentor_archetype")) or None,
        "mentor_style": _normalized_text(profile.get("mentor_style")) or None,
        "priorities": _normalized_string_list(profile.get("preferences"))[:5],
    }


async def _generate_checkup_message(
    prompt: str,
    max_tokens: int = 280,
    style_directive: Optional[str] = None,
    force_html: bool = False,
) -> Optional[str]:
    """Generate optional LLM-enhanced coaching text, if provider is initialized."""
    try:
        from app.llm import service as llm_service_module
        from app.llm.base import CompletionRequest, ChatMessage

        llm_service = llm_service_module._llm_service
        if not llm_service or not llm_service._initialized:
            return None

        if force_html:
            system_content = (
                "You are a precise productivity coach and schedule designer. "
                "Return ONLY valid HTML markup that can be directly rendered in a web UI. "
                "Do NOT use markdown, code fences, or explanations outside HTML. "
                "Do NOT emit <html>, <head>, <body>, <script>, or <style> tags."
            )
        else:
            system_content = (
                "You are a concise accountability coach. Keep responses practical, specific, "
                "and under 140 words. Use short bullet points when useful."
            )
        if style_directive:
            system_content += f" Align the voice with this communication profile: {style_directive}"

        request = CompletionRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=system_content,
                ),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=0.35,
            max_tokens=max_tokens,
        )

        response = await llm_service.chat_completion(request)
        content = (response.content or "").strip()
        return content or None
    except Exception as llm_error:
        logger.warning("Daily checkup LLM generation failed, using fallback text: %s", llm_error)
        return None


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value

    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _extract_checkup_identity_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None

    checkup_type = str(metadata.get("checkup_type", "")).strip().lower()
    checkup_date = str(metadata.get("checkup_date") or metadata.get("date") or "").strip()[:10]
    if checkup_type not in {"morning", "evening"} or not checkup_date:
        return None

    return f"{checkup_type}:{checkup_date}"


def _build_checkup_content_from_payload(payload: Dict[str, Any]) -> str:
    checkup_type = str(payload.get("checkup_type", "")).strip().lower()
    checkup_date = str(payload.get("checkup_date") or payload.get("date") or "").strip()[:10]
    coach_message = str(payload.get("coach_message") or "").strip()

    if checkup_type == "morning":
        focus_target = str(payload.get("focus_target") or "").strip() or "Most important task"
        return (
            f"Morning checkup for {checkup_date}\n"
            f"Focus: {focus_target}\n\n"
            f"Coach Guidance:\n{coach_message}"
        )

    wins = payload.get("wins") if isinstance(payload.get("wins"), list) else []
    tomorrow_focus = payload.get("tomorrow_focus") if isinstance(payload.get("tomorrow_focus"), list) else []
    wins_lines = "\n".join([f"- {str(item).strip()}" for item in wins if str(item).strip()])
    tomorrow_lines = "\n".join([f"- {str(item).strip()}" for item in tomorrow_focus if str(item).strip()])

    return (
        f"Evening checkup for {checkup_date}\n\n"
        f"Wins:\n{wins_lines or '- n/a'}\n\n"
        f"Tomorrow Focus:\n{tomorrow_lines or '- n/a'}\n\n"
        f"Coach Reflection:\n{coach_message}"
    )


def _record_to_knowledge_entry(record: DailyCheckupRecord) -> Optional[KnowledgeEntry]:
    payload = dict(record.payload or {})
    payload.setdefault("checkup_type", record.checkup_type)
    payload.setdefault("date", record.checkup_date.isoformat())
    payload.setdefault("checkup_date", record.checkup_date.isoformat())

    checkup_type = str(payload.get("checkup_type", "")).strip().lower()
    if checkup_type not in {"morning", "evening"}:
        return None

    checkup_date = str(payload.get("checkup_date") or payload.get("date") or "").strip()[:10]
    if not checkup_date:
        return None

    title_prefix = "Morning Checkup" if checkup_type == "morning" else "Evening Checkup"
    entry_id = f"db_daily_checkup::{record.user_id}::{checkup_type}::{checkup_date}"

    return KnowledgeEntry(
        entry_id=entry_id,
        user_id=record.user_id,
        entry_type=KnowledgeEntryType.INSIGHT,
        entry_sub_type=KnowledgeEntrySubType.MISC_INSIGHT,
        category="daily_checkup",
        title=f"{title_prefix} - {checkup_date}",
        content=_build_checkup_content_from_payload(payload),
        metadata=payload,
        tags=sorted({"daily_checkup", checkup_type, "insight"}),
        created_at=_coerce_datetime(record.created_at),
        updated_at=_coerce_datetime(record.updated_at),
    )


def _merge_db_checkup_entries(
    entries: List[KnowledgeEntry],
    *,
    category: Optional[str],
    entry_type: Optional[KnowledgeEntryType],
) -> List[KnowledgeEntry]:
    wants_insights = entry_type in {None, KnowledgeEntryType.INSIGHT}
    wants_checkups = category in {None, "daily_checkup"}
    if not (wants_insights and wants_checkups):
        return entries

    checkup_store = get_daily_checkup_store()
    if not checkup_store:
        return entries

    request_user = get_current_user()
    db_records = checkup_store.list_checkups_for_user(request_user.storage_key)
    if not db_records:
        return entries

    merged_entries = list(entries)
    existing_index: Dict[str, int] = {}

    for index, entry in enumerate(merged_entries):
        identity = _extract_checkup_identity_from_metadata(entry.metadata or {})
        if identity:
            existing_index[identity] = index

    for record in db_records:
        db_entry = _record_to_knowledge_entry(record)
        if not db_entry:
            continue

        identity = _extract_checkup_identity_from_metadata(db_entry.metadata or {})
        if not identity:
            continue

        existing_idx = existing_index.get(identity)
        if existing_idx is None:
            existing_index[identity] = len(merged_entries)
            merged_entries.append(db_entry)
            continue

        existing_entry = merged_entries[existing_idx]
        if _coerce_datetime(db_entry.updated_at) > _coerce_datetime(existing_entry.updated_at):
            merged_entries[existing_idx] = db_entry

    return merged_entries


def _is_legacy_system_preference_entry(entry: KnowledgeEntry) -> bool:
    return (
        entry.entry_type == KnowledgeEntryType.PREFERENCE
        and str(entry.category or "").strip().lower() == "system"
        and str(entry.title or "").strip().lower() == "user preferences"
    )


def _prune_legacy_system_preference_entries(entries: List[KnowledgeEntry]) -> List[KnowledgeEntry]:
    return [entry for entry in entries if not _is_legacy_system_preference_entry(entry)]


def _extract_sync_event_key_from_entry(entry: KnowledgeEntry) -> str:
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
    return str(context.get("sync_event_key") or "").strip()


def _dedupe_external_sync_entries(entries: List[KnowledgeEntry]) -> List[KnowledgeEntry]:
    latest_by_key: Dict[str, KnowledgeEntry] = {}

    for entry in entries:
        sync_event_key = _extract_sync_event_key_from_entry(entry)
        if not sync_event_key:
            continue

        previous = latest_by_key.get(sync_event_key)
        if previous is None or _coerce_datetime(entry.updated_at) >= _coerce_datetime(previous.updated_at):
            latest_by_key[sync_event_key] = entry

    if not latest_by_key:
        return entries

    deduped_entries: List[KnowledgeEntry] = []
    for entry in entries:
        sync_event_key = _extract_sync_event_key_from_entry(entry)
        if not sync_event_key:
            deduped_entries.append(entry)
            continue

        if latest_by_key.get(sync_event_key) is entry:
            deduped_entries.append(entry)

    return deduped_entries


def _persist_checkup_payload_to_db(checkup_type: str, checkup_date: date, payload: Dict[str, Any]) -> None:
    checkup_store = get_daily_checkup_store()
    if not checkup_store:
        return

    request_user = get_current_user()
    payload_to_store = dict(payload or {})
    payload_to_store["checkup_type"] = checkup_type
    payload_to_store["checkup_date"] = checkup_date.isoformat()
    payload_to_store["date"] = checkup_date.isoformat()

    saved = checkup_store.upsert_checkup(
        user_id=request_user.storage_key,
        checkup_type=checkup_type,
        checkup_date=checkup_date,
        payload=payload_to_store,
    )
    if not saved:
        logger.warning("Failed to persist %s checkup in database for user=%s", checkup_type, request_user.storage_key)


def _load_latest_checkups_from_db() -> Dict[str, Dict[str, Any]]:
    checkup_store = get_daily_checkup_store()
    if not checkup_store:
        return {}

    request_user = get_current_user()
    latest_records = checkup_store.get_latest_checkups_for_user(request_user.storage_key)

    resolved: Dict[str, Dict[str, Any]] = {}
    for checkup_type in ("morning", "evening"):
        record = latest_records.get(checkup_type)
        if not record:
            continue

        payload = dict(record.payload or {})
        payload.setdefault("checkup_type", checkup_type)
        payload.setdefault("date", record.checkup_date.isoformat())
        payload.setdefault("checkup_date", record.checkup_date.isoformat())
        resolved[checkup_type] = payload

    return resolved


def _safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator <= 0:
        return fallback
    return max(0.0, numerator / denominator)


def _normalize_notification_status(value: str) -> str:
    normalized = str(value or "active").strip().lower()
    if normalized in {"active", "acknowledged", "resolved"}:
        return normalized
    return "active"


def _normalize_notification_severity(value: str) -> str:
    normalized = str(value or "medium").strip().lower()
    if normalized in {"low", "medium", "high", "critical"}:
        return normalized
    return "medium"


def _coerce_datetime_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return _coerce_datetime(value).astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _notification_response_from_record(record: AINotificationRecord) -> AINotificationResponse:
    payload = record.payload if isinstance(record.payload, dict) else {}
    recommended_actions = _normalized_string_list(payload.get("recommended_actions"))

    return AINotificationResponse(
        id=record.id,
        notification_key=record.notification_key,
        kind=record.kind,
        severity=_normalize_notification_severity(record.severity),
        status=_normalize_notification_status(record.status),
        title=record.title,
        summary=record.summary,
        details=record.details,
        score=record.score,
        recommended_actions=recommended_actions,
        payload=payload,
        first_seen_at=_coerce_datetime_iso(record.first_seen_at) or datetime.now(timezone.utc).isoformat(),
        last_seen_at=_coerce_datetime_iso(record.last_seen_at) or datetime.now(timezone.utc).isoformat(),
        acknowledged_at=_coerce_datetime_iso(record.acknowledged_at),
        resolved_at=_coerce_datetime_iso(record.resolved_at),
        updated_at=_coerce_datetime_iso(record.updated_at) or datetime.now(timezone.utc).isoformat(),
    )


def _notification_response_from_candidate(candidate: Dict[str, Any], index: int) -> AINotificationResponse:
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
    payload.setdefault(
        "recommended_actions",
        _normalized_string_list(candidate.get("recommended_actions")),
    )

    return AINotificationResponse(
        id=-(index + 1),
        notification_key=str(candidate.get("notification_key") or f"ephemeral-{index}").strip(),
        kind=str(candidate.get("kind") or "signal").strip() or "signal",
        severity=_normalize_notification_severity(str(candidate.get("severity") or "medium")),
        status="active",
        title=str(candidate.get("title") or "AI Notification").strip() or "AI Notification",
        summary=str(candidate.get("summary") or "Generated AI signal").strip() or "Generated AI signal",
        details=str(candidate.get("details")).strip() if candidate.get("details") is not None else None,
        score=_safe_float(candidate.get("score"), default=0.0) if candidate.get("score") is not None else None,
        recommended_actions=_normalized_string_list(candidate.get("recommended_actions")),
        payload=payload,
        first_seen_at=now_iso,
        last_seen_at=now_iso,
        updated_at=now_iso,
    )


def _extract_goal_titles(all_entries: List[KnowledgeEntry]) -> List[str]:
    goals: List[str] = []
    for entry in all_entries:
        if entry.entry_type != KnowledgeEntryType.USER_PREFERENCE:
            continue
        if entry.entry_sub_type != KnowledgeEntrySubType.GOAL:
            continue

        title = str(entry.title or "").strip()
        if title:
            goals.append(title)

    deduped: List[str] = []
    seen: Set[str] = set()
    for goal in goals:
        normalized = goal.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(goal)
    return deduped


def _extract_latest_checkup_metric(
    latest_checkups: Dict[str, Dict[str, Any]],
    keys: List[str],
    *,
    default: float = 0.0,
) -> float:
    for checkup_type in ("evening", "morning"):
        payload = latest_checkups.get(checkup_type)
        if not isinstance(payload, dict):
            continue

        search_roots = [
            payload.get("decision_metrics") if isinstance(payload.get("decision_metrics"), dict) else {},
            payload.get("performance") if isinstance(payload.get("performance"), dict) else {},
            payload.get("stats") if isinstance(payload.get("stats"), dict) else {},
            payload,
        ]

        for root in search_roots:
            candidate = root
            found = True
            for key in keys:
                if not isinstance(candidate, dict) or key not in candidate:
                    found = False
                    break
                candidate = candidate.get(key)

            if found:
                return _safe_float(candidate, default=default)

    return default


def _recent_checkup_consistency_ratio(user_id: str, lookback_days: int = 7) -> float:
    checkup_store = get_daily_checkup_store()
    if not checkup_store:
        return 0.5

    records = checkup_store.list_checkups_for_user(user_id)
    if not records:
        return 0.0

    today = datetime.now(timezone.utc).date()
    window_start = today - timedelta(days=max(1, lookback_days - 1))
    active_dates = {record.checkup_date for record in records if record.checkup_date >= window_start}

    target_days = min(5, max(1, lookback_days))
    return min(1.0, len(active_dates) / float(target_days))


async def _resolve_latest_checkups_with_fallback(kb_service) -> Dict[str, Dict[str, Any]]:
    latest_from_db = _load_latest_checkups_from_db()

    if latest_from_db.get("morning") and latest_from_db.get("evening"):
        return {
            "morning": latest_from_db.get("morning"),
            "evening": latest_from_db.get("evening"),
        }

    existing_entries = await kb_service.get_all_entries(
        category="daily_checkup",
        entry_type=KnowledgeEntryType.INSIGHT,
    )

    sorted_entries = sorted(
        existing_entries,
        key=lambda entry: _coerce_datetime(entry.updated_at or entry.created_at),
        reverse=True,
    )

    latest_from_kb: Dict[str, Dict[str, Any]] = {}
    for entry in sorted_entries:
        entry_metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        checkup_type = str(entry_metadata.get("checkup_type", "")).strip().lower()
        checkup_date = str(entry_metadata.get("checkup_date") or entry_metadata.get("date") or "").strip()[:10]

        if checkup_type not in {"morning", "evening"}:
            continue
        if checkup_type in latest_from_kb:
            continue

        payload = dict(entry_metadata)
        payload.setdefault("checkup_type", checkup_type)
        if checkup_date:
            payload.setdefault("date", checkup_date)
            payload.setdefault("checkup_date", checkup_date)

        latest_from_kb[checkup_type] = payload
        if "morning" in latest_from_kb and "evening" in latest_from_kb:
            break

    return {
        "morning": latest_from_db.get("morning") or latest_from_kb.get("morning") or {},
        "evening": latest_from_db.get("evening") or latest_from_kb.get("evening") or {},
    }


def _build_ai_notification_candidates(
    *,
    all_entries: List[KnowledgeEntry],
    preferences: UserPreferences,
    latest_checkups: Dict[str, Dict[str, Any]],
    user_id: str,
) -> List[Dict[str, Any]]:
    goal_titles = _extract_goal_titles(all_entries)

    now_utc = datetime.now(timezone.utc)
    lookback_start = now_utc - timedelta(days=13)

    recent_time_entries: List[Dict[str, Any]] = []
    for entry in all_entries:
        if _normalize_entry_category(entry) != "time_entry":
            continue

        event_ts = _resolve_entry_event_timestamp(entry)
        if _ensure_timezone(event_ts) < lookback_start:
            continue

        context = _entry_context(entry)
        duration_minutes = _safe_float(context.get("duration_minutes"), default=0.0)
        if duration_minutes <= 0 and context.get("duration_seconds") is not None:
            duration_minutes = _safe_float(context.get("duration_seconds"), default=0.0) / 60.0

        raw_billable = context.get("billable", False)
        is_billable = raw_billable if isinstance(raw_billable, bool) else str(raw_billable).strip().lower() in {"1", "true", "yes"}

        recent_time_entries.append(
            {
                "duration_minutes": max(0.0, duration_minutes),
                "billable": is_billable,
            }
        )

    time_entry_count = len(recent_time_entries)
    billable_count = len([entry for entry in recent_time_entries if entry["billable"]])
    total_minutes = sum(entry["duration_minutes"] for entry in recent_time_entries)
    avg_time_entry_minutes = _safe_ratio(total_minutes, float(max(1, time_entry_count)), fallback=0.0)
    billable_ratio = _safe_ratio(float(billable_count), float(max(1, time_entry_count)), fallback=0.0)

    overdue_tasks = int(max(
        0,
        round(_extract_latest_checkup_metric(latest_checkups, ["overdue_tasks"], default=0.0)),
    ))
    due_today_tasks = int(max(
        0,
        round(_extract_latest_checkup_metric(latest_checkups, ["due_today_tasks"], default=0.0)),
    ))
    planned_deep_work_minutes = _extract_latest_checkup_metric(
        latest_checkups,
        ["planned_deep_work_minutes"],
        default=0.0,
    )
    deep_work_coverage_ratio = _extract_latest_checkup_metric(
        latest_checkups,
        ["deep_work_coverage_ratio"],
        default=0.0,
    )
    performance_score = _extract_latest_checkup_metric(
        latest_checkups,
        ["score"],
        default=0.0,
    )

    habits_total = int(max(0, round(_extract_latest_checkup_metric(latest_checkups, ["habits_total"], default=0.0))))
    habits_completed = int(max(0, round(_extract_latest_checkup_metric(latest_checkups, ["habits_completed_today"], default=0.0))))
    habits_completion_rate_7d = _extract_latest_checkup_metric(
        latest_checkups,
        ["habits_completion_rate_7d"],
        default=0.0,
    )

    habit_completion_ratio = (
        _safe_ratio(float(habits_completed), float(max(1, habits_total)), fallback=0.0)
        if habits_total > 0
        else max(0.0, min(1.0, habits_completion_rate_7d / 100.0))
    )

    if deep_work_coverage_ratio <= 0:
        deep_work_coverage_ratio = 0.55
    deep_work_coverage_ratio = max(0.0, min(1.0, deep_work_coverage_ratio))

    if performance_score <= 0:
        performance_score = 5.6
    performance_score = max(0.0, min(10.0, performance_score))

    if habit_completion_ratio <= 0:
        habit_completion_ratio = 0.6

    checkup_consistency = _recent_checkup_consistency_ratio(user_id)
    deadline_health_ratio = max(0.0, 1.0 - min(((overdue_tasks * 1.5) + (due_today_tasks * 0.75)) / 10.0, 1.0))

    goal_alignment_score = round(
        ((performance_score / 10.0) * 35.0)
        + (deep_work_coverage_ratio * 25.0)
        + (habit_completion_ratio * 20.0)
        + (deadline_health_ratio * 10.0)
        + (checkup_consistency * 10.0)
    )

    if goal_alignment_score < 45:
        goal_alignment_severity = "critical"
    elif goal_alignment_score < 60:
        goal_alignment_severity = "high"
    elif goal_alignment_score < 75:
        goal_alignment_severity = "medium"
    else:
        goal_alignment_severity = "low"

    goal_alignment_actions = [
        "Protect your first 90-minute deep-work block before any reactive tasks.",
        "Close at least one overdue or due-today item before midday.",
    ]
    if habit_completion_ratio < 0.7:
        goal_alignment_actions.append("Anchor one non-negotiable habit block to stabilize consistency.")

    candidates: List[Dict[str, Any]] = [
        {
            "notification_key": "goal_alignment_score",
            "kind": "goal_alignment",
            "severity": goal_alignment_severity,
            "title": f"Goal Alignment Score: {goal_alignment_score}/100",
            "summary": (
                "Execution quality, deep-work protection, deadlines, and consistency are now scored daily. "
                f"Current signal is {goal_alignment_score}/100."
            ),
            "details": (
                f"Goals tracked: {len(goal_titles)}. Overdue: {overdue_tasks}. Due today: {due_today_tasks}. "
                f"Deep-work coverage: {round(deep_work_coverage_ratio * 100)}%."
            ),
            "score": float(goal_alignment_score),
            "recommended_actions": goal_alignment_actions,
            "payload": {
                "goal_alignment_score": goal_alignment_score,
                "metrics": {
                    "performance_score": round(performance_score, 2),
                    "deep_work_coverage_ratio": round(deep_work_coverage_ratio, 2),
                    "habit_completion_ratio": round(habit_completion_ratio, 2),
                    "checkup_consistency_ratio": round(checkup_consistency, 2),
                    "deadline_health_ratio": round(deadline_health_ratio, 2),
                    "overdue_tasks": overdue_tasks,
                    "due_today_tasks": due_today_tasks,
                    "time_entry_count_14d": time_entry_count,
                    "billable_ratio_14d": round(billable_ratio, 2),
                },
                "top_goals": goal_titles[:3],
                "recommended_actions": goal_alignment_actions,
            },
        }
    ]

    if overdue_tasks > 0 or due_today_tasks >= 3:
        candidates.append(
            {
                "notification_key": "proactive.deadline_drift",
                "kind": "proactive_alert",
                "severity": "high" if overdue_tasks > 0 else "medium",
                "title": "Proactive Alert: Deadline Drift Risk",
                "summary": (
                    f"{overdue_tasks} overdue and {due_today_tasks} due-today commitments signal drift against planned outcomes."
                ),
                "details": "Re-sequence your day around the highest consequence deadlines before reactive work expands.",
                "recommended_actions": [
                    "Create a first-thing deadline triage block for 30 minutes.",
                    "Reduce WIP to one deadline-critical task until drift clears.",
                ],
                "payload": {
                    "overdue_tasks": overdue_tasks,
                    "due_today_tasks": due_today_tasks,
                    "recommended_actions": [
                        "Create a first-thing deadline triage block for 30 minutes.",
                        "Reduce WIP to one deadline-critical task until drift clears.",
                    ],
                },
            }
        )

    if planned_deep_work_minutes >= 60 and deep_work_coverage_ratio < 0.6:
        candidates.append(
            {
                "notification_key": "proactive.deep_work_gap",
                "kind": "proactive_alert",
                "severity": "high" if deep_work_coverage_ratio < 0.45 else "medium",
                "title": "Proactive Alert: Deep-Work Coverage Gap",
                "summary": (
                    f"Only {round(deep_work_coverage_ratio * 100)}% of planned deep work is landing. "
                    "Execution quality will trend down if this persists."
                ),
                "details": "Protect one uninterrupted block and move lower-leverage meetings after it.",
                "recommended_actions": [
                    "Block a no-meeting focus window at your peak energy time.",
                    "Set one success criterion for the block before you start.",
                ],
                "payload": {
                    "planned_deep_work_minutes": round(planned_deep_work_minutes, 1),
                    "deep_work_coverage_ratio": round(deep_work_coverage_ratio, 2),
                    "recommended_actions": [
                        "Block a no-meeting focus window at your peak energy time.",
                        "Set one success criterion for the block before you start.",
                    ],
                },
            }
        )

    if habits_total >= 3 and habit_completion_ratio < 0.6:
        candidates.append(
            {
                "notification_key": "proactive.habit_consistency",
                "kind": "proactive_alert",
                "severity": "medium",
                "title": "Proactive Alert: Habit Consistency Slipping",
                "summary": (
                    f"Habit completion is at {round(habit_completion_ratio * 100)}% today. "
                    "Identity-level routines are getting crowded out."
                ),
                "details": "Habits should be defended as protected blocks, not leftovers after task overflow.",
                "recommended_actions": [
                    "Schedule one protected habit block in your next available window.",
                    "Tie the habit to an existing anchor event (wake-up, lunch, shutdown).",
                ],
                "payload": {
                    "habits_total": habits_total,
                    "habits_completed_today": habits_completed,
                    "habit_completion_ratio": round(habit_completion_ratio, 2),
                    "recommended_actions": [
                        "Schedule one protected habit block in your next available window.",
                        "Tie the habit to an existing anchor event (wake-up, lunch, shutdown).",
                    ],
                },
            }
        )

    monthly_income_target = 0.0
    if isinstance(preferences.finance, dict):
        monthly_income_target = _safe_float(preferences.finance.get("monthly_income_target"), default=0.0)

    if monthly_income_target > 0 and time_entry_count >= 8 and billable_ratio < 0.45:
        candidates.append(
            {
                "notification_key": "proactive.billable_trajectory",
                "kind": "proactive_alert",
                "severity": "high" if billable_ratio < 0.3 else "medium",
                "title": "Proactive Alert: Billable Trajectory Behind",
                "summary": (
                    f"Billable ratio over recent logs is {round(billable_ratio * 100)}%, below the threshold needed "
                    "to stay on your financial target trajectory."
                ),
                "details": "Shift calendar slots toward high-value billable blocks before the week closes.",
                "recommended_actions": [
                    "Reserve two billable-first blocks in the next 48 hours.",
                    "Audit low-value tasks and defer or delegate one of them.",
                ],
                "payload": {
                    "monthly_income_target": round(monthly_income_target, 2),
                    "billable_ratio_14d": round(billable_ratio, 2),
                    "avg_time_entry_minutes": round(avg_time_entry_minutes, 1),
                    "recommended_actions": [
                        "Reserve two billable-first blocks in the next 48 hours.",
                        "Audit low-value tasks and defer or delegate one of them.",
                    ],
                },
            }
        )

    timezone_name = _extract_preferences_timezone(preferences)
    today_local = datetime.now(_resolve_timezone(timezone_name)).date().isoformat()
    latest_morning_date = str(
        (latest_checkups.get("morning") or {}).get("checkup_date")
        or (latest_checkups.get("morning") or {}).get("date")
        or ""
    ).strip()[:10]

    if latest_morning_date != today_local:
        candidates.append(
            {
                "notification_key": "proactive.morning_checkup_missing",
                "kind": "proactive_alert",
                "severity": "medium",
                "title": "Proactive Alert: Morning Check-In Missing",
                "summary": "No morning strategy check-in detected for today. Priority drift risk is elevated.",
                "details": "A quick morning alignment prevents reactive work from defining the day.",
                "recommended_actions": [
                    "Run a morning checkup before your next context switch.",
                    "Set one non-negotiable focus outcome for today.",
                ],
                "payload": {
                    "today": today_local,
                    "latest_morning_checkup": latest_morning_date or None,
                    "recommended_actions": [
                        "Run a morning checkup before your next context switch.",
                        "Set one non-negotiable focus outcome for today.",
                    ],
                },
            }
        )

    return candidates


async def _refresh_ai_notifications(kb_service, *, limit: int = 40) -> Dict[str, Any]:
    request_user = get_current_user()
    all_entries = await kb_service.get_all_entries()
    all_entries = _merge_db_checkup_entries(all_entries, category=None, entry_type=None)
    all_entries = _dedupe_external_sync_entries(all_entries)
    all_entries = _prune_legacy_system_preference_entries(all_entries)

    preferences = await kb_service.get_user_preferences()
    latest_checkups = await _resolve_latest_checkups_with_fallback(kb_service)
    candidates = _build_ai_notification_candidates(
        all_entries=all_entries,
        preferences=preferences,
        latest_checkups=latest_checkups,
        user_id=request_user.storage_key,
    )

    notification_store = get_ai_notification_store()
    generated_at = datetime.now(timezone.utc).isoformat()

    if not notification_store:
        return {
            "persistence_enabled": False,
            "notifications": [
                _notification_response_from_candidate(candidate, index).model_dump()
                for index, candidate in enumerate(candidates[: max(1, min(limit, 200))])
            ],
            "generated": len(candidates),
            "upserted": 0,
            "resolved": 0,
            "generated_at": generated_at,
        }

    active_keys: List[str] = []
    upserted_count = 0
    for candidate in candidates:
        notification_key = str(candidate.get("notification_key") or "").strip()
        if not notification_key:
            continue

        payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
        payload.setdefault("recommended_actions", _normalized_string_list(candidate.get("recommended_actions")))

        record = notification_store.upsert_notification(
            user_id=request_user.storage_key,
            notification_key=notification_key,
            kind=str(candidate.get("kind") or "signal").strip() or "signal",
            severity=str(candidate.get("severity") or "medium").strip() or "medium",
            title=str(candidate.get("title") or "AI Notification").strip() or "AI Notification",
            summary=str(candidate.get("summary") or "Generated AI signal").strip() or "Generated AI signal",
            details=str(candidate.get("details")).strip() if candidate.get("details") is not None else None,
            score=_safe_float(candidate.get("score"), default=0.0) if candidate.get("score") is not None else None,
            payload=payload,
            origin="ai_notifications_v1",
        )

        if record:
            active_keys.append(notification_key)
            upserted_count += 1

    resolved_count = notification_store.mark_stale_notifications_resolved(
        user_id=request_user.storage_key,
        active_keys=active_keys,
        origin="ai_notifications_v1",
    )

    records = notification_store.list_notifications(
        user_id=request_user.storage_key,
        limit=limit,
        include_resolved=False,
    )

    return {
        "persistence_enabled": True,
        "notifications": [_notification_response_from_record(record).model_dump() for record in records],
        "generated": len(candidates),
        "upserted": upserted_count,
        "resolved": resolved_count,
        "generated_at": generated_at,
    }


async def _sync_missing_db_checkup_insights(kb_service) -> Dict[str, int]:
    """Materialize DB-backed checkups into KB entries for embedding-driven features."""
    checkup_store = get_daily_checkup_store()
    if not checkup_store:
        return {"records": 0, "created": 0, "failed": 0}

    request_user = get_current_user()
    db_records = checkup_store.list_checkups_for_user(request_user.storage_key)
    if not db_records:
        return {"records": 0, "created": 0, "failed": 0}

    existing_entries = await kb_service.get_all_entries(
        category="daily_checkup",
        entry_type=KnowledgeEntryType.INSIGHT,
    )
    existing_identities = {
        identity
        for identity in (
            _extract_checkup_identity_from_metadata(entry.metadata or {})
            for entry in existing_entries
        )
        if identity
    }

    created_count = 0
    failed_count = 0

    for record in db_records:
        payload = dict(record.payload or {})
        payload.setdefault("checkup_type", record.checkup_type)
        payload.setdefault("date", record.checkup_date.isoformat())
        payload.setdefault("checkup_date", record.checkup_date.isoformat())

        identity = _extract_checkup_identity_from_metadata(payload)
        if not identity or identity in existing_identities:
            continue

        checkup_type = str(payload.get("checkup_type", "")).strip().lower()
        checkup_date_text = str(payload.get("checkup_date") or payload.get("date") or "").strip()[:10]
        if checkup_type not in {"morning", "evening"} or not checkup_date_text:
            continue

        try:
            checkup_date = date.fromisoformat(checkup_date_text)
        except ValueError:
            logger.warning(
                "Skipping DB checkup with invalid date user=%s type=%s date=%s",
                request_user.storage_key,
                checkup_type,
                checkup_date_text,
            )
            continue

        title_prefix = "Morning Checkup" if checkup_type == "morning" else "Evening Checkup"
        # Add sync_event_key for deterministic ID (prevents duplicates on re-sync)
        checkup_sync_key = f"checkup:{checkup_type}:{checkup_date.isoformat()}"
        payload["sync_event_key"] = checkup_sync_key
        try:
            await kb_service.create_entry(
                entry_type=KnowledgeEntryType.INSIGHT,
                entry_sub_type=KnowledgeEntrySubType.MISC_INSIGHT,
                category="daily_checkup",
                title=f"{title_prefix} - {checkup_date.isoformat()}",
                content=_build_checkup_content_from_payload(payload),
                metadata=payload,
                tags=sorted({"daily_checkup", checkup_type, "insight"}),
            )
            existing_identities.add(identity)
            created_count += 1
        except Exception as sync_error:
            failed_count += 1
            logger.warning(
                "Failed to materialize DB checkup into KB user=%s type=%s date=%s: %s",
                request_user.storage_key,
                checkup_type,
                checkup_date.isoformat(),
                sync_error,
            )

    return {
        "records": len(db_records),
        "created": created_count,
        "failed": failed_count,
    }


async def _upsert_checkup_insight(
    kb_service,
    checkup_type: str,
    checkup_date: date,
    title: str,
    content: str,
    metadata: Dict[str, Any],
    tags: List[str],
) -> None:
    """Persist a single daily checkup insight entry with date/type upsert semantics."""
    existing_entries = await kb_service.get_all_entries(
        category="daily_checkup",
        entry_type=KnowledgeEntryType.INSIGHT,
    )

    checkup_date_iso = checkup_date.isoformat()
    target_entry = None
    for entry in existing_entries:
        entry_metadata = entry.metadata or {}
        if (
            str(entry_metadata.get("checkup_type", "")).strip().lower() == checkup_type
            and str(entry_metadata.get("checkup_date", "")).strip() == checkup_date_iso
        ):
            target_entry = entry
            break

    merged_metadata = dict(metadata or {})
    merged_metadata["checkup_type"] = checkup_type
    merged_metadata["checkup_date"] = checkup_date_iso
    # Add sync_event_key for deterministic ID (prevents duplicates on re-sync)
    merged_metadata["context"] = {"sync_event_key": f"checkup:{checkup_type}:{checkup_date_iso}"}

    normalized_tags = sorted(set([*tags, "daily_checkup", checkup_type, "insight"]))

    if target_entry:
        updated_entry = await kb_service.update_entry(
            entry_id=target_entry.entry_id,
            title=title,
            content=content,
            metadata=merged_metadata,
            tags=normalized_tags,
        )
        if updated_entry:
            return

    await kb_service.create_entry(
        entry_type=KnowledgeEntryType.INSIGHT,
        entry_sub_type=KnowledgeEntrySubType.MISC_INSIGHT,
        category="daily_checkup",
        title=title,
        content=content,
        metadata=merged_metadata,
        tags=normalized_tags,
    )


@router.get("/analytics")
async def get_knowledge_analytics(
    time_range: str = Query("30d", alias="range", regex="^(7d|30d|90d)$", description="Analytics range")
):
    """Return live analytics data computed from persisted knowledge entries."""
    try:
        kb_service = get_knowledge_base_service()
        await _sync_missing_db_checkup_insights(kb_service)
        all_entries = await kb_service.get_all_entries()
        all_entries = _merge_db_checkup_entries(all_entries, category=None, entry_type=None)
        all_entries = _dedupe_external_sync_entries(all_entries)
        all_entries = _prune_legacy_system_preference_entries(all_entries)

        preferences = await kb_service.get_user_preferences()
        analytics_timezone_name = _extract_preferences_timezone(preferences)
        analytics_timezone = _resolve_timezone(analytics_timezone_name)

        days = _resolve_date_range(time_range)
        now = datetime.now(analytics_timezone)
        start = now - timedelta(days=days - 1)

        # Normalize datetimes and sort once for deterministic analytics output.
        entries_with_ts = []
        for entry in all_entries:
            entry_ts = _to_timezone(_resolve_entry_event_timestamp(entry), analytics_timezone)
            entries_with_ts.append((entry, entry_ts))

        entries_with_ts.sort(key=lambda item: item[1])

        in_range_entries = [(entry, ts) for entry, ts in entries_with_ts if ts >= start]
        interaction_entries = [
            (entry, ts)
            for entry, ts in in_range_entries
            if entry.entry_type == KnowledgeEntryType.INTERACTION
        ]

        # Daily interaction counts with dominant agent for the day.
        daily_counts: Dict[str, int] = defaultdict(int)
        daily_agent_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        daily_category_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        hourly_counts: Dict[int, int] = defaultdict(int)
        agent_counts: Dict[str, int] = defaultdict(int)
        category_counts: Dict[str, int] = defaultdict(int)
        time_entry_count = 0
        time_entry_billable_count = 0
        time_entry_total_minutes = 0.0

        for entry, ts in interaction_entries:
            day_key = _format_iso_date(ts)
            agent_name = (
                (entry.metadata or {}).get("agent_type")
                or entry.category
                or "unknown"
            )
            normalized_agent = str(agent_name).strip().lower() or "unknown"
            daily_counts[day_key] += 1
            daily_agent_counts[day_key][normalized_agent] += 1
            hourly_counts[ts.hour] += 1
            agent_counts[normalized_agent] += 1

        for entry, ts in in_range_entries:
            day_key = _format_iso_date(ts)
            normalized_category = _normalize_entry_category(entry)
            daily_category_counts[day_key][normalized_category] += 1
            category_counts[normalized_category] += 1

            if normalized_category == "time_entry":
                time_entry_count += 1
                metadata = entry.metadata or {}
                context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

                duration_minutes = context.get("duration_minutes")
                if duration_minutes is not None:
                    time_entry_total_minutes += max(0.0, _safe_float(duration_minutes, default=0.0))

                raw_billable = context.get("billable", False)
                if isinstance(raw_billable, bool):
                    is_billable = raw_billable
                else:
                    is_billable = str(raw_billable).strip().lower() in {"1", "true", "yes"}

                if is_billable:
                    time_entry_billable_count += 1

        daily_interactions = []
        knowledge_growth = []
        preference_changes = []
        category_focus = []

        total_up_to_day = 0
        cursor = start
        end_date = now.date()

        # Pre-compute counts by day for all entries and preference-like entries.
        entries_by_day: Dict[str, int] = defaultdict(int)
        pref_by_day: Dict[str, int] = defaultdict(int)
        pref_category_by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for entry, ts in entries_with_ts:
            day_key = _format_iso_date(ts)
            entries_by_day[day_key] += 1

            if entry.entry_type in {KnowledgeEntryType.PREFERENCE, KnowledgeEntryType.USER_PREFERENCE}:
                pref_by_day[day_key] += 1
                pref_category_by_day[day_key][entry.category] += 1

        # Compute cumulative total through the day before the window start.
        for _, ts in entries_with_ts:
            if ts.date() < start.date():
                total_up_to_day += 1

        while cursor.date() <= end_date:
            day_key = cursor.date().isoformat()

            dominant_agent = "none"
            if daily_agent_counts.get(day_key):
                dominant_agent = max(
                    daily_agent_counts[day_key].items(),
                    key=lambda item: item[1],
                )[0]

            daily_interactions.append({
                "date": day_key,
                "count": daily_counts.get(day_key, 0),
                "agent": dominant_agent,
            })

            new_entries_today = entries_by_day.get(day_key, 0)
            total_up_to_day += new_entries_today
            knowledge_growth.append({
                "date": day_key,
                "total_entries": total_up_to_day,
                "new_entries": new_entries_today,
            })

            top_pref_category = "none"
            if pref_category_by_day.get(day_key):
                top_pref_category = max(
                    pref_category_by_day[day_key].items(),
                    key=lambda item: item[1],
                )[0]

            preference_changes.append({
                "date": day_key,
                "category": top_pref_category,
                "changes": pref_by_day.get(day_key, 0),
            })

            dominant_category = "none"
            dominant_category_count = 0
            if daily_category_counts.get(day_key):
                dominant_category, dominant_category_count = max(
                    daily_category_counts[day_key].items(),
                    key=lambda item: item[1],
                )

            category_focus.append({
                "date": day_key,
                "category": dominant_category,
                "count": dominant_category_count,
            })

            cursor += timedelta(days=1)

        weekly_map: Dict[str, int] = defaultdict(int)
        for item in daily_interactions:
            dt = datetime.fromisoformat(item["date"])
            weekly_map[_week_bucket_label(dt)] += item["count"]

        weekly_interactions = [
            {"week": week_label, "count": count}
            for week_label, count in sorted(weekly_map.items())
        ]

        agent_palette = [
            "#3b82f6",
            "#f59e0b",
            "#10b981",
            "#06b6d4",
            "#8b5cf6",
            "#ec4899",
            "#ef4444",
            "#22c55e",
        ]

        sorted_agent_counts = sorted(agent_counts.items(), key=lambda item: item[1], reverse=True)
        by_agent = [
            {
                "agent": agent.replace("_", " ").title(),
                "count": count,
                "color": agent_palette[idx % len(agent_palette)],
            }
            for idx, (agent, count) in enumerate(sorted_agent_counts)
        ]

        category_palette = [
            "#06b6d4",
            "#10b981",
            "#f59e0b",
            "#8b5cf6",
            "#ec4899",
            "#3b82f6",
            "#f97316",
            "#22c55e",
        ]
        sorted_category_counts = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
        by_category = [
            {
                "category": category.replace("_", " ").title(),
                "raw_category": category,
                "count": count,
                "color": category_palette[idx % len(category_palette)],
            }
            for idx, (category, count) in enumerate(sorted_category_counts)
        ]

        most_used_agent = by_agent[0]["agent"] if by_agent else "N/A"
        top_knowledge_category = by_category[0]["category"] if by_category else "N/A"
        total_interactions = len(interaction_entries)
        avg_daily_interactions = total_interactions / max(days, 1)

        total_pref_changes = sum(item["changes"] for item in preference_changes)
        change_frequency = total_pref_changes / max(days, 1)
        preference_stability = max(0.0, min(1.0, 1.0 - min(change_frequency / 2.0, 1.0)))

        new_entries_in_range = len(in_range_entries)
        learning_velocity = new_entries_in_range / max(days, 1)
        avg_time_entry_minutes = time_entry_total_minutes / time_entry_count if time_entry_count else 0

        most_active_hours = [
            {
                "hour": hour,
                "interactions": hourly_counts.get(hour, 0),
            }
            for hour in range(24)
        ]

        return {
            "interactions": {
                "daily": daily_interactions,
                "weekly": weekly_interactions,
                "by_agent": by_agent,
                "by_category": by_category,
            },
            "patterns": {
                "most_active_hours": most_active_hours,
                "preference_changes": preference_changes,
                "knowledge_growth": knowledge_growth,
                "category_focus": category_focus,
            },
            "insights": {
                "total_interactions": total_interactions,
                "most_used_agent": most_used_agent,
                "avg_daily_interactions": round(avg_daily_interactions, 2),
                "knowledge_base_size": len(all_entries),
                "preference_stability": round(preference_stability, 2),
                "learning_velocity": round(learning_velocity, 2),
                "top_knowledge_category": top_knowledge_category,
                "time_entry_records": time_entry_count,
                "time_entry_billable_records": time_entry_billable_count,
                "avg_time_entry_minutes": round(avg_time_entry_minutes, 1),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics data: {str(e)}")


@router.post("/notifications/refresh")
async def refresh_ai_notifications(
    limit: int = Query(40, ge=1, le=200, description="Maximum notifications to return"),
):
    """Recompute and persist AI notifications from the latest behavioral signals."""
    try:
        kb_service = get_knowledge_base_service()
        return await _refresh_ai_notifications(kb_service, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh AI notifications: {str(e)}")


@router.get("/notifications")
async def list_ai_notifications(
    limit: int = Query(40, ge=1, le=200, description="Maximum notifications to return"),
    include_resolved: bool = Query(False, description="Include resolved notifications in response"),
):
    """List persisted AI notifications for the current user."""
    try:
        notification_store = get_ai_notification_store()
        if not notification_store:
            kb_service = get_knowledge_base_service()
            return await _refresh_ai_notifications(kb_service, limit=limit)

        request_user = get_current_user()
        records = notification_store.list_notifications(
            user_id=request_user.storage_key,
            limit=limit,
            include_resolved=include_resolved,
        )
        return {
            "persistence_enabled": True,
            "notifications": [_notification_response_from_record(record).model_dump() for record in records],
            "generated": len(records),
            "upserted": 0,
            "resolved": 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list AI notifications: {str(e)}")


@router.patch("/notifications/{notification_id}/ack", response_model=AINotificationResponse)
async def acknowledge_ai_notification(notification_id: int, request: NotificationAcknowledgeRequest):
    """Acknowledge or unacknowledge a notification without deleting its insight history."""
    try:
        notification_store = get_ai_notification_store()
        if not notification_store:
            raise HTTPException(status_code=503, detail="AI notification persistence is not configured")

        request_user = get_current_user()
        record = notification_store.set_acknowledged(
            user_id=request_user.storage_key,
            notification_id=notification_id,
            acknowledged=bool(request.acknowledged),
        )
        if not record:
            raise HTTPException(status_code=404, detail="Notification not found")

        return _notification_response_from_record(record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update notification acknowledgement: {str(e)}")


@router.post("/checkups/morning")
async def run_morning_checkup(request: DailyCheckupRequest):
    """Generate a morning planning checkup using persisted knowledge and time-entry context."""
    try:
        kb_service = get_knowledge_base_service()
        note = (request.note or "").strip()
        perspective = request.perspective if isinstance(request.perspective, dict) else {}
        context_snapshot = request.context_snapshot if isinstance(request.context_snapshot, dict) else {}
        if not context_snapshot and isinstance(request.contextSnapshot, dict):
            context_snapshot = request.contextSnapshot

        preferences = await kb_service.get_user_preferences()
        checkup_timezone_name = _resolve_checkup_timezone(request.timezone, context_snapshot, preferences)
        checkup_timezone = _resolve_timezone(checkup_timezone_name)
        checkup_date = _parse_requested_date(request.date, checkup_timezone_name)
        checkup_now_local = datetime.now(checkup_timezone)

        all_entries = await kb_service.get_all_entries()
        all_entries = _dedupe_external_sync_entries(all_entries)
        time_entries: List[Dict[str, Any]] = []

        for entry in all_entries:
            if _normalize_entry_category(entry) != "time_entry":
                continue

            event_ts = _resolve_entry_event_timestamp(entry)
            event_date = _to_timezone(event_ts, checkup_timezone).date()
            if event_date > checkup_date:
                continue

            context = _entry_context(entry)
            duration_minutes = _safe_float(context.get("duration_minutes"), default=0.0)
            if duration_minutes <= 0 and context.get("duration_seconds") is not None:
                duration_minutes = _safe_float(context.get("duration_seconds"), default=0.0) / 60.0

            time_entries.append({
                "event_date": event_date,
                "event_ts": event_ts,
                "project_name": str(context.get("project_name") or "").strip() or "Unassigned",
                "description": str(context.get("description") or entry.title or "Untitled task").strip(),
                "duration_minutes": max(0.0, duration_minutes),
                "focus_score": _safe_float(context.get("focus_score"), default=0.0),
            })

        lookback_start = checkup_date - timedelta(days=6)
        week_entries = [item for item in time_entries if lookback_start <= item["event_date"] <= checkup_date]
        today_entries = [item for item in time_entries if item["event_date"] == checkup_date]

        total_week_minutes = sum(item["duration_minutes"] for item in week_entries)
        avg_daily_minutes = total_week_minutes / 7.0

        project_minutes: Dict[str, float] = defaultdict(float)
        for item in week_entries:
            project_minutes[item["project_name"]] += item["duration_minutes"]

        top_projects = [
            project
            for project, _ in sorted(project_minutes.items(), key=lambda pair: pair[1], reverse=True)[:3]
        ]

        valid_focus_scores = [item["focus_score"] for item in week_entries if item["focus_score"] > 0]
        avg_focus_score = round(sum(valid_focus_scores) / len(valid_focus_scores), 2) if valid_focus_scores else None

        priorities = preferences.general.get("priorities", []) if isinstance(preferences.general, dict) else []
        priorities = priorities if isinstance(priorities, list) else []
        communication_profile = _extract_communication_profile(all_entries)
        style_directive = _build_style_directive(communication_profile, "morning")

        work_hours = (
            (preferences.general.get("work_hours") if isinstance(preferences.general, dict) else None)
            or (preferences.productivity.get("work_hours") if isinstance(preferences.productivity, dict) else None)
            or "09:00-17:00"
        )
        check_in_time = (
            preferences.journal.get("check_in_time")
            if isinstance(preferences.journal, dict)
            else "09:00"
        ) or "09:00"
        evening_check_time = (
            (preferences.journal.get("evening_check_time") if isinstance(preferences.journal, dict) else None)
            or (preferences.journal.get("evening_check_in_time") if isinstance(preferences.journal, dict) else None)
            or (preferences.journal.get("check_out_time") if isinstance(preferences.journal, dict) else None)
            or "21:00"
        )

        run_anchor_time = (
            _minutes_to_hhmm((checkup_now_local.hour * 60) + checkup_now_local.minute)
            if checkup_now_local.date() == checkup_date
            else None
        )

        priority_focus = str(context_snapshot.get("priorityFocus") or "").strip()
        if not priority_focus:
            priority_focus = str(priorities[0]).strip() if priorities else ""
        project_focus = top_projects[0] if top_projects else ""
        focus_target = note or priority_focus or project_focus or "Most important task"

        deadline_tasks = context_snapshot.get("deadlineTasks", {}) if isinstance(context_snapshot.get("deadlineTasks"), dict) else {}
        overdue_tasks = int(_safe_float(deadline_tasks.get("overdue"), default=0.0))
        due_today_tasks = int(_safe_float(deadline_tasks.get("dueToday"), default=0.0))
        planned_deep_work_minutes = _safe_float(perspective.get("plannedDeepWorkMinutes"), default=0.0)
        confidence_score = _safe_float(perspective.get("confidence"), default=0.0)

        top_goals = context_snapshot.get("topGoals", []) if isinstance(context_snapshot.get("topGoals"), list) else []
        top_goals_text = ", ".join(str(goal) for goal in top_goals[:3]) if top_goals else "none"

        focus_tasks_raw = context_snapshot.get("focusTasks") if isinstance(context_snapshot.get("focusTasks"), list) else []
        focus_tasks = [item for item in focus_tasks_raw if isinstance(item, dict)]
        focus_task_titles = [
            str(item.get("title") or "").strip()
            for item in focus_tasks
            if str(item.get("title") or "").strip()
        ][:3]

        upcoming_deadlines_raw = (
            context_snapshot.get("upcomingDeadlines")
            if isinstance(context_snapshot.get("upcomingDeadlines"), list)
            else []
        )
        upcoming_deadlines = [item for item in upcoming_deadlines_raw if isinstance(item, dict)]
        upcoming_deadlines_count = len(upcoming_deadlines)

        habit_metrics = context_snapshot.get("habitMetrics", {}) if isinstance(context_snapshot.get("habitMetrics"), dict) else {}
        habits_total = int(_safe_float(habit_metrics.get("totalHabits"), default=0.0))
        habits_completed_today = int(_safe_float(habit_metrics.get("completedToday"), default=0.0))
        habits_avg_streak = _safe_float(habit_metrics.get("avgStreak"), default=0.0)
        habits_completion_rate_7d = _safe_float(habit_metrics.get("completionRate7d"), default=0.0)

        time_metrics = context_snapshot.get("timeMetrics", {}) if isinstance(context_snapshot.get("timeMetrics"), dict) else {}
        tracked_time_spent_minutes = _safe_float(time_metrics.get("totalTimeSpentMinutes"), default=0.0)
        tracked_estimated_minutes = _safe_float(time_metrics.get("totalEstimatedMinutes"), default=0.0)
        deep_work_coverage_ratio = _safe_float(time_metrics.get("deepWorkCoverageRatio"), default=0.0)

        fallback_lines = [
            f"Primary focus: {focus_target}",
            f"Anchor check-in time: {check_in_time}",
            f"Schedule horizon (to evening check): {evening_check_time}",
            f"Suggested work window baseline: {work_hours}",
            f"Last 7-day average logged work: {_format_minutes(avg_daily_minutes)}",
        ]
        if focus_task_titles:
            fallback_lines.append(f"Focus tasks: {', '.join(focus_task_titles)}")
        if overdue_tasks > 0:
            fallback_lines.append(f"Overdue tasks to clear first: {overdue_tasks}")
        if due_today_tasks > 0:
            fallback_lines.append(f"Tasks due today: {due_today_tasks}")
        if upcoming_deadlines_count > 0:
            fallback_lines.append(f"Upcoming deadlines (7d): {upcoming_deadlines_count}")
        if habits_total > 0:
            fallback_lines.append(
                f"Habits today: {habits_completed_today}/{habits_total} complete ({round(habits_completion_rate_7d, 1)}% over 7d)"
            )
        if planned_deep_work_minutes > 0:
            fallback_lines.append(f"Planned deep work: {_format_minutes(planned_deep_work_minutes)}")
        if tracked_time_spent_minutes > 0:
            fallback_lines.append(f"Tracked task time so far: {_format_minutes(tracked_time_spent_minutes)}")
        if confidence_score > 0:
            fallback_lines.append(f"Self-reported confidence: {round(confidence_score, 1)}/10")
        if top_projects:
            fallback_lines.append(f"Keep momentum on: {', '.join(top_projects)}")
        if avg_focus_score is not None:
            fallback_lines.append(f"Recent focus baseline: {avg_focus_score}/10")

        schedule_blocks = _build_morning_schedule_blocks(
            focus_target=focus_target,
            focus_task_titles=focus_task_titles,
            work_hours=work_hours,
            check_in_time=check_in_time,
            schedule_end_time=str(evening_check_time),
            run_anchor_time=run_anchor_time,
            planned_deep_work_minutes=planned_deep_work_minutes,
            overdue_tasks=overdue_tasks,
            due_today_tasks=due_today_tasks,
            habits_total=habits_total,
            habits_completed_today=habits_completed_today,
            avg_daily_minutes=avg_daily_minutes,
        )

        schedule_seed = " | ".join(
            [
                f"{block['start_label']}-{block['end_label']}: {block['title']} ({block['reason']})"
                for block in schedule_blocks
            ]
        )

        fallback_html = _build_morning_schedule_html(
            checkup_date=checkup_date,
            focus_target=focus_target,
            fallback_lines=fallback_lines,
            schedule_blocks=schedule_blocks,
        )
        fallback_text = _build_fallback_checkup_message(fallback_lines, communication_profile, "morning")

        llm_prompt = (
            f"Date: {checkup_date.isoformat()}\n"
            f"Intent note: {note or 'none'}\n"
            f"Communication profile: {style_directive}\n"
            f"Focus target: {focus_target}\n"
            f"Work hours: {work_hours}\n"
            f"Check-in time: {check_in_time}\n"
            f"Evening check boundary: {evening_check_time}\n"
            f"Runtime anchor time: {run_anchor_time or 'n/a'}\n"
            f"Last 7 days logged minutes: {round(total_week_minutes, 1)}\n"
            f"Average daily logged minutes: {round(avg_daily_minutes, 1)}\n"
            f"Top projects: {', '.join(top_projects) if top_projects else 'none'}\n"
            f"Top goals from context: {top_goals_text}\n"
            f"Focus tasks from context: {', '.join(focus_task_titles) if focus_task_titles else 'none'}\n"
            f"Overdue tasks: {overdue_tasks}\n"
            f"Due today tasks: {due_today_tasks}\n"
            f"Upcoming deadlines (7d): {upcoming_deadlines_count}\n"
            f"Habit completion today: {habits_completed_today}/{habits_total}\n"
            f"Habit avg streak: {round(habits_avg_streak, 1)}\n"
            f"Habit completion rate 7d: {round(habits_completion_rate_7d, 1)}\n"
            f"Tracked estimated task minutes: {round(tracked_estimated_minutes, 1)}\n"
            f"Tracked spent task minutes: {round(tracked_time_spent_minutes, 1)}\n"
            f"Deep work coverage ratio from context: {round(deep_work_coverage_ratio, 2)}\n"
            f"Planned deep work minutes: {planned_deep_work_minutes}\n"
            f"User confidence: {confidence_score if confidence_score > 0 else 'n/a'}\n"
            f"Today existing entries: {len(today_entries)}\n"
            f"Schedule seed blocks: {schedule_seed or 'none'}\n"
            "Use the schedule seed block times exactly as provided; do not invent or shift block start/end times. "
            "Reason over task priorities, deadlines, habits, and tracked time to produce a practical daily schedule. "
            "Return ONLY valid HTML that can be rendered directly. "
            "Use this exact semantic structure and class names: "
            "<section class='daily-checkup'>"
            "<header class='dc-header'><div class='dc-badge-row'><span class='dc-kicker'>Morning Checkup</span><span class='dc-date'>date label</span></div><h3 class='dc-focus'>one focus sentence</h3><p class='dc-subtitle'>short strategic context sentence</p></header>"
            "<section class='dc-metrics'><div class='dc-metric'><p class='dc-metric-label'>Scheduled Blocks</p><p class='dc-metric-value'>numeric value</p></div><div class='dc-metric'><p class='dc-metric-label'>High Priority</p><p class='dc-metric-value'>numeric value</p></div><div class='dc-metric'><p class='dc-metric-label'>Planned Duration</p><p class='dc-metric-value'>duration label</p></div></section>"
            "<section class='daily-schedule dc-panel'><div class='dc-panel-head'><p class='dc-panel-title'>Time-Blocked Plan</p><p class='dc-panel-subtitle'>short subtitle</p></div><ol class='dc-timeline'><li class='dc-block dc-block--high|medium|low'><div class='dc-time-wrap'><span class='dc-time'>start - end</span><span class='dc-priority'>Priority label</span></div><div class='dc-block-copy'><p class='dc-block-title'>block title</p><p class='dc-block-reason'>why this block matters</p></div></li></ol></section>"
            "<section class='execution-notes dc-panel'><p class='dc-panel-title'>Execution Notes</p><ul class='dc-notes'><li>three concise action bullets tied to deadlines/habits/priorities</li></ul></section>"
            "<section class='journal dc-panel dc-journal'><p class='dc-panel-title'>Accountability + Journal</p><p class='dc-journal-q'>Accountability: one direct accountability question.</p><p class='dc-journal-q'>Journal prompt: one reflective prompt.</p></section>"
            "</section>. "
            "Requirements: 4-7 timeline blocks with explicit start/end times, concise copy, no markdown fences, no scripts, and no inline styles."
        )

        llm_message = await _generate_checkup_message(
            llm_prompt,
            max_tokens=720,
            style_directive=style_directive,
            force_html=True,
        )
        llm_candidate = llm_message.strip() if llm_message and _looks_like_html(llm_message) else None
        llm_html = llm_candidate if llm_candidate and _is_structured_morning_checkup_html(llm_candidate) else None
        coach_message_html = llm_html or fallback_html
        coach_message = _strip_html_tags(coach_message_html) or fallback_text

        response_payload = {
            "date": checkup_date.isoformat(),
            "checkup_type": "morning",
            "intent_note": note or None,
            "focus_target": focus_target,
            "recommended_projects": top_projects,
            "stats": {
                "today_entries": len(today_entries),
                "last_7_days_entries": len(week_entries),
                "last_7_days_minutes": round(total_week_minutes, 1),
                "avg_daily_minutes": round(avg_daily_minutes, 1),
                "avg_focus_score": avg_focus_score,
            },
            "decision_metrics": {
                "overdue_tasks": overdue_tasks,
                "due_today_tasks": due_today_tasks,
                "upcoming_deadlines_7d": upcoming_deadlines_count,
                "planned_deep_work_minutes": round(planned_deep_work_minutes, 1),
                "tracked_spent_task_minutes": round(tracked_time_spent_minutes, 1),
                "tracked_estimated_task_minutes": round(tracked_estimated_minutes, 1),
                "deep_work_coverage_ratio": round(deep_work_coverage_ratio, 2),
                "habits_total": habits_total,
                "habits_completed_today": habits_completed_today,
                "habits_avg_streak": round(habits_avg_streak, 1),
                "habits_completion_rate_7d": round(habits_completion_rate_7d, 1),
                "confidence": round(confidence_score, 1) if confidence_score > 0 else None,
            },
            "journaling": {
                "intent_prompt": (
                    "What one behavior will make today a win, even if everything else changes?"
                ),
                "accountability_prompt": "What will you protect first if your schedule compresses?",
                "focus_commitment": {
                    "priority_focus": focus_target,
                    "focus_tasks": focus_task_titles,
                    "goal_anchors": top_goals[:3],
                    "deadline_pressure": {
                        "overdue": overdue_tasks,
                        "due_today": due_today_tasks,
                        "upcoming_7d": upcoming_deadlines_count,
                    },
                    "habit_anchor": {
                        "completed_today": habits_completed_today,
                        "total": habits_total,
                        "avg_streak": round(habits_avg_streak, 1),
                    },
                },
            },
            "perspective": perspective,
            "context_snapshot": context_snapshot,
            "style_profile": _public_style_profile(communication_profile),
            "coach_message": coach_message,
            "coach_message_html": coach_message_html,
            "daily_schedule": schedule_blocks,
            "generated_with": "llm_html" if llm_html else "fallback_html",
        }

        insight_content = (
            f"Morning checkup for {checkup_date.isoformat()}\n"
            f"Focus: {focus_target}\n"
            f"Work Hours: {work_hours}\n"
            f"Check-in: {check_in_time}\n\n"
            f"Coach Guidance:\n{coach_message}"
        )

        await _upsert_checkup_insight(
            kb_service=kb_service,
            checkup_type="morning",
            checkup_date=checkup_date,
            title=f"Morning Checkup - {checkup_date.isoformat()}",
            content=insight_content,
            metadata=response_payload,
            tags=["planning"],
        )
        _persist_checkup_payload_to_db("morning", checkup_date, response_payload)

        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate morning checkup: {str(e)}")


@router.post("/checkups/evening")
async def run_evening_checkup(request: DailyCheckupRequest):
    """Generate an evening reflection checkup based on a single day of logged context."""
    try:
        kb_service = get_knowledge_base_service()
        note = (request.note or "").strip()
        perspective = request.perspective if isinstance(request.perspective, dict) else {}
        context_snapshot = request.context_snapshot if isinstance(request.context_snapshot, dict) else {}
        if not context_snapshot and isinstance(request.contextSnapshot, dict):
            context_snapshot = request.contextSnapshot

        preferences = await kb_service.get_user_preferences()
        checkup_timezone_name = _resolve_checkup_timezone(request.timezone, context_snapshot, preferences)
        checkup_timezone = _resolve_timezone(checkup_timezone_name)
        checkup_date = _parse_requested_date(request.date, checkup_timezone_name)

        all_entries = await kb_service.get_all_entries()
        all_entries = _dedupe_external_sync_entries(all_entries)
        today_entries: List[Dict[str, Any]] = []

        for entry in all_entries:
            if _normalize_entry_category(entry) != "time_entry":
                continue

            event_ts = _resolve_entry_event_timestamp(entry)
            if _to_timezone(event_ts, checkup_timezone).date() != checkup_date:
                continue

            context = _entry_context(entry)

            duration_minutes = _safe_float(context.get("duration_minutes"), default=0.0)
            if duration_minutes <= 0 and context.get("duration_seconds") is not None:
                duration_minutes = _safe_float(context.get("duration_seconds"), default=0.0) / 60.0

            raw_billable = context.get("billable", False)
            if isinstance(raw_billable, bool):
                is_billable = raw_billable
            else:
                is_billable = str(raw_billable).strip().lower() in {"1", "true", "yes"}

            blocker_text = str(context.get("blockers") or "").strip()

            today_entries.append({
                "event_ts": event_ts,
                "project_name": str(context.get("project_name") or "").strip() or "Unassigned",
                "description": str(context.get("description") or entry.title or "Untitled task").strip(),
                "duration_minutes": max(0.0, duration_minutes),
                "billable": is_billable,
                "focus_score": _safe_float(context.get("focus_score"), default=0.0),
                "energy_score": _safe_float(context.get("energy_score"), default=0.0),
                "blockers": blocker_text,
            })

        today_entries.sort(key=lambda item: item["event_ts"])

        total_minutes = sum(item["duration_minutes"] for item in today_entries)
        billable_minutes = sum(item["duration_minutes"] for item in today_entries if item["billable"])

        project_minutes: Dict[str, float] = defaultdict(float)
        for item in today_entries:
            project_minutes[item["project_name"]] += item["duration_minutes"]

        top_projects = [
            project
            for project, _ in sorted(project_minutes.items(), key=lambda pair: pair[1], reverse=True)[:3]
        ]

        focus_scores = [item["focus_score"] for item in today_entries if item["focus_score"] > 0]
        energy_scores = [item["energy_score"] for item in today_entries if item["energy_score"] > 0]
        avg_focus = round(sum(focus_scores) / len(focus_scores), 2) if focus_scores else None
        avg_energy = round(sum(energy_scores) / len(energy_scores), 2) if energy_scores else None

        deadline_tasks = context_snapshot.get("deadlineTasks", {}) if isinstance(context_snapshot.get("deadlineTasks"), dict) else {}
        overdue_tasks = int(_safe_float(deadline_tasks.get("overdue"), default=0.0))
        due_today_tasks = int(_safe_float(deadline_tasks.get("dueToday"), default=0.0))
        completed_tasks_today = int(_safe_float(context_snapshot.get("completedTasksToday"), default=0.0))

        focus_tasks_raw = context_snapshot.get("focusTasks") if isinstance(context_snapshot.get("focusTasks"), list) else []
        focus_tasks = [item for item in focus_tasks_raw if isinstance(item, dict)]
        focus_task_titles = [
            str(item.get("title") or "").strip()
            for item in focus_tasks
            if str(item.get("title") or "").strip()
        ][:3]

        upcoming_deadlines_raw = (
            context_snapshot.get("upcomingDeadlines")
            if isinstance(context_snapshot.get("upcomingDeadlines"), list)
            else []
        )
        upcoming_deadlines = [item for item in upcoming_deadlines_raw if isinstance(item, dict)]
        upcoming_deadlines_count = len(upcoming_deadlines)

        habit_metrics = context_snapshot.get("habitMetrics", {}) if isinstance(context_snapshot.get("habitMetrics"), dict) else {}
        habits_total = int(_safe_float(habit_metrics.get("totalHabits"), default=0.0))
        habits_completed_today = int(_safe_float(habit_metrics.get("completedToday"), default=0.0))
        habits_avg_streak = _safe_float(habit_metrics.get("avgStreak"), default=0.0)
        habits_completion_rate_7d = _safe_float(habit_metrics.get("completionRate7d"), default=0.0)

        time_metrics = context_snapshot.get("timeMetrics", {}) if isinstance(context_snapshot.get("timeMetrics"), dict) else {}
        tracked_time_spent_minutes = _safe_float(time_metrics.get("totalTimeSpentMinutes"), default=0.0)
        tracked_estimated_minutes = _safe_float(time_metrics.get("totalEstimatedMinutes"), default=0.0)
        deep_work_coverage_ratio = _safe_float(time_metrics.get("deepWorkCoverageRatio"), default=0.0)

        planned_deep_work_minutes = _safe_float(
            perspective.get("plannedDeepWorkMinutes"),
            default=_safe_float(context_snapshot.get("plannedDeepWorkMinutes"), default=0.0),
        )
        self_rating = _safe_float(perspective.get("selfRating"), default=0.0)

        top_priority_raw = perspective.get("topPriorityCompleted")
        if isinstance(top_priority_raw, bool):
            top_priority_completed = top_priority_raw
        else:
            top_priority_completed = str(top_priority_raw).strip().lower() in {"1", "true", "yes", "done"}

        baseline_minutes = planned_deep_work_minutes if planned_deep_work_minutes > 0 else 120.0
        minutes_score = min(4.0, (total_minutes / max(1.0, baseline_minutes)) * 4.0)
        focus_component = min(3.0, ((avg_focus or 0.0) / 10.0) * 3.0)
        task_denominator = max(1, due_today_tasks + overdue_tasks)
        tasks_component = min(2.0, (completed_tasks_today / task_denominator) * 2.0) if task_denominator > 0 else 0.0
        priority_component = 1.0 if top_priority_completed else 0.0

        objective_score = round(min(10.0, minutes_score + focus_component + tasks_component + priority_component), 2)
        subjective_score = round(max(0.0, min(10.0, self_rating)), 2) if self_rating > 0 else None
        performance_score = round(
            (objective_score * 0.65)
            + ((subjective_score if subjective_score is not None else objective_score) * 0.35),
            2,
        )

        blockers = sorted({item["blockers"] for item in today_entries if item["blockers"]})
        longest_tasks = sorted(today_entries, key=lambda item: item["duration_minutes"], reverse=True)[:3]

        wins: List[str] = []
        if total_minutes >= 180:
            wins.append(f"You protected {_format_minutes(total_minutes)} of focused work today.")
        if avg_focus is not None and avg_focus >= 4:
            wins.append(f"Focus quality was strong at {avg_focus}/10.")
        if top_projects:
            wins.append(f"You made measurable progress in {', '.join(top_projects[:2])}.")
        if not wins and today_entries:
            wins.append("You maintained momentum by logging and reflecting on your work.")

        tomorrow_focus = []
        if longest_tasks:
            tomorrow_focus.append(f"Continue momentum on: {longest_tasks[0]['description']}")
        if blockers:
            tomorrow_focus.append(f"Address blocker first: {blockers[0]}")
        if not tomorrow_focus:
            tomorrow_focus.append("Define your top 1 task before starting tomorrow.")

        fallback_lines = [
            f"Total logged today: {_format_minutes(total_minutes)}",
            f"Billable portion: {_format_minutes(billable_minutes)}",
            f"Sessions captured: {len(today_entries)}",
            f"Estimated performance score: {performance_score}/10",
        ]
        if avg_focus is not None:
            fallback_lines.append(f"Average focus: {avg_focus}/10")
        if avg_energy is not None:
            fallback_lines.append(f"Average energy: {avg_energy}/10")
        if subjective_score is not None:
            fallback_lines.append(f"Self-assessment: {subjective_score}/10")
        if focus_task_titles:
            fallback_lines.append(f"Focus tasks reviewed: {', '.join(focus_task_titles)}")
        if overdue_tasks > 0:
            fallback_lines.append(f"Overdue tasks carried: {overdue_tasks}")
        if upcoming_deadlines_count > 0:
            fallback_lines.append(f"Upcoming deadlines (7d): {upcoming_deadlines_count}")
        if habits_total > 0:
            fallback_lines.append(
                f"Habits completed: {habits_completed_today}/{habits_total} ({round(habits_completion_rate_7d, 1)}% over 7d)"
            )
        if tracked_time_spent_minutes > 0:
            fallback_lines.append(f"Tracked task effort: {_format_minutes(tracked_time_spent_minutes)}")
        fallback_lines.extend([f"Tomorrow: {item}" for item in tomorrow_focus])

        communication_profile = _extract_communication_profile(all_entries)
        style_directive = _build_style_directive(communication_profile, "evening")

        recap_line = (
            f"You logged {_format_minutes(total_minutes)} across {len(today_entries)} sessions with an estimated score of {performance_score}/10."
        )
        fallback_html = _build_evening_reflection_html(
            checkup_date=checkup_date,
            recap_line=recap_line,
            total_minutes=total_minutes,
            billable_minutes=billable_minutes,
            performance_score=performance_score,
            avg_focus=avg_focus,
            avg_energy=avg_energy,
            wins=wins,
            blockers=blockers,
            tomorrow_focus=tomorrow_focus,
            focus_task_titles=focus_task_titles,
            top_projects=top_projects,
        )
        fallback_text = _build_fallback_checkup_message(fallback_lines, communication_profile, "evening")

        llm_prompt = (
            f"Date: {checkup_date.isoformat()}\n"
            f"Reflection note: {note or 'none'}\n"
            f"Communication profile: {style_directive}\n"
            f"Total minutes: {round(total_minutes, 1)}\n"
            f"Billable minutes: {round(billable_minutes, 1)}\n"
            f"Sessions: {len(today_entries)}\n"
            f"Top projects: {', '.join(top_projects) if top_projects else 'none'}\n"
            f"Avg focus: {avg_focus if avg_focus is not None else 'n/a'}\n"
            f"Avg energy: {avg_energy if avg_energy is not None else 'n/a'}\n"
            f"Blockers: {', '.join(blockers) if blockers else 'none'}\n"
            f"Focus tasks from context: {', '.join(focus_task_titles) if focus_task_titles else 'none'}\n"
            f"Overdue tasks: {overdue_tasks}\n"
            f"Due today tasks: {due_today_tasks}\n"
            f"Upcoming deadlines (7d): {upcoming_deadlines_count}\n"
            f"Completed tasks today: {completed_tasks_today}\n"
            f"Habits completed today: {habits_completed_today}/{habits_total}\n"
            f"Habit avg streak: {round(habits_avg_streak, 1)}\n"
            f"Habit completion rate 7d: {round(habits_completion_rate_7d, 1)}\n"
            f"Tracked estimated task minutes: {round(tracked_estimated_minutes, 1)}\n"
            f"Tracked spent task minutes: {round(tracked_time_spent_minutes, 1)}\n"
            f"Deep work coverage ratio from context: {round(deep_work_coverage_ratio, 2)}\n"
            f"Planned deep work minutes: {planned_deep_work_minutes}\n"
            f"Self-rating: {subjective_score if subjective_score is not None else 'n/a'}\n"
            f"Objective score: {objective_score}/10\n"
            f"Precomputed wins: {', '.join(wins) if wins else 'none'}\n"
            f"Precomputed tomorrow focus: {', '.join(tomorrow_focus) if tomorrow_focus else 'none'}\n"
            "Return ONLY valid HTML that can be rendered directly. "
            "Use this exact semantic structure and class names: "
            "<section class='daily-checkup evening-checkup'>"
            "<header class='dc-header'><div class='dc-badge-row'><span class='dc-kicker'>Evening Checkup</span><span class='dc-date'>date label</span></div><h3 class='dc-focus'>one recap sentence</h3><p class='dc-subtitle'>one concise strategic summary</p></header>"
            "<section class='dc-metrics'><div class='dc-metric'><p class='dc-metric-label'>Logged Time</p><p class='dc-metric-value'>duration</p></div><div class='dc-metric'><p class='dc-metric-label'>Billable Time</p><p class='dc-metric-value'>duration</p></div><div class='dc-metric'><p class='dc-metric-label'>Performance</p><p class='dc-metric-value'>score/10</p></div></section>"
            "<section class='daily-schedule dc-panel'><div class='dc-panel-head'><p class='dc-panel-title'>Tomorrow Commitments</p><p class='dc-panel-subtitle'>focus/energy signal</p></div><ol class='dc-timeline'><li class='dc-block dc-block--high|medium|low'><div class='dc-time-wrap'><span class='dc-time'>Action 1</span><span class='dc-priority'>priority label</span></div><div class='dc-block-copy'><p class='dc-block-title'>concrete action</p><p class='dc-block-reason'>why this action matters</p></div></li></ol></section>"
            "<section class='execution-notes dc-panel'><p class='dc-panel-title'>Wins + Friction</p><ul class='dc-notes'><li>two evidence-based wins and up to two friction points</li></ul></section>"
            "<section class='journal dc-panel dc-journal'><p class='dc-panel-title'>Reflection + Accountability</p><p class='dc-journal-q'>reflection question</p><p class='dc-journal-q'>accountability question</p></section>"
            "</section>. "
            "Requirements: include 2-4 tomorrow commitments, tie actions to deadlines/habits/deep-work context, no markdown fences, no scripts, and no inline styles."
        )

        llm_message = await _generate_checkup_message(
            llm_prompt,
            max_tokens=760,
            style_directive=style_directive,
            force_html=True,
        )
        llm_candidate = llm_message.strip() if llm_message and _looks_like_html(llm_message) else None
        llm_html = llm_candidate if llm_candidate and _is_structured_evening_checkup_html(llm_candidate) else None
        coach_message_html = llm_html or fallback_html
        coach_message = _strip_html_tags(coach_message_html) or fallback_text

        timeline = [
            {
                "time": item["event_ts"].isoformat(),
                "project": item["project_name"],
                "description": item["description"],
                "duration_minutes": round(item["duration_minutes"], 1),
                "billable": item["billable"],
            }
            for item in today_entries
        ]

        response_payload = {
            "date": checkup_date.isoformat(),
            "checkup_type": "evening",
            "reflection_note": note or None,
            "stats": {
                "total_minutes": round(total_minutes, 1),
                "billable_minutes": round(billable_minutes, 1),
                "sessions": len(today_entries),
                "avg_focus": avg_focus,
                "avg_energy": avg_energy,
                "top_projects": top_projects,
            },
            "performance": {
                "score": performance_score,
                "objective_score": objective_score,
                "subjective_score": subjective_score,
                "minutes_score_component": round(minutes_score, 2),
                "focus_score_component": round(focus_component, 2),
                "tasks_score_component": round(tasks_component, 2),
                "top_priority_component": round(priority_component, 2),
                "planned_deep_work_minutes": round(planned_deep_work_minutes, 1),
                "overdue_tasks": overdue_tasks,
                "due_today_tasks": due_today_tasks,
                "completed_tasks_today": completed_tasks_today,
                "top_priority_completed": top_priority_completed,
            },
            "decision_metrics": {
                "overdue_tasks": overdue_tasks,
                "due_today_tasks": due_today_tasks,
                "upcoming_deadlines_7d": upcoming_deadlines_count,
                "focus_tasks_count": len(focus_task_titles),
                "habits_total": habits_total,
                "habits_completed_today": habits_completed_today,
                "habits_avg_streak": round(habits_avg_streak, 1),
                "habits_completion_rate_7d": round(habits_completion_rate_7d, 1),
                "tracked_spent_task_minutes": round(tracked_time_spent_minutes, 1),
                "tracked_estimated_task_minutes": round(tracked_estimated_minutes, 1),
                "deep_work_coverage_ratio": round(deep_work_coverage_ratio, 2),
            },
            "perspective": perspective,
            "context_snapshot": context_snapshot,
            "wins": wins,
            "blockers": blockers,
            "tomorrow_focus": tomorrow_focus,
            "reflection_journal": {
                "recap_prompt": "What moved your day forward the most, and why?",
                "wins": wins,
                "friction_points": blockers,
                "tomorrow_commitments": tomorrow_focus,
                "evidence": {
                    "top_projects": top_projects,
                    "focus_tasks": focus_task_titles,
                    "deadline_pressure": {
                        "overdue": overdue_tasks,
                        "due_today": due_today_tasks,
                        "upcoming_7d": upcoming_deadlines_count,
                    },
                    "habit_state": {
                        "completed_today": habits_completed_today,
                        "total": habits_total,
                        "avg_streak": round(habits_avg_streak, 1),
                    },
                    "deep_work_coverage_ratio": round(deep_work_coverage_ratio, 2),
                },
                "tomorrow_prompt": "What will you do differently in the first 30 minutes tomorrow?",
            },
            "timeline": timeline,
            "style_profile": _public_style_profile(communication_profile),
            "coach_message": coach_message,
            "coach_message_html": coach_message_html,
            "generated_with": "llm_html" if llm_html else "fallback_html",
        }

        insight_content = (
            f"Evening checkup for {checkup_date.isoformat()}\n"
            f"Total Logged: {_format_minutes(total_minutes)}\n"
            f"Billable: {_format_minutes(billable_minutes)}\n"
            f"Performance Score: {performance_score}/10\n"
            f"Sessions: {len(today_entries)}\n\n"
            "Wins:\n" + "\n".join(f"- {item}" for item in wins) + "\n\n"
            "Tomorrow Focus:\n" + "\n".join(f"- {item}" for item in tomorrow_focus) + "\n\n"
            f"Coach Reflection:\n{coach_message}"
        )

        await _upsert_checkup_insight(
            kb_service=kb_service,
            checkup_type="evening",
            checkup_date=checkup_date,
            title=f"Evening Checkup - {checkup_date.isoformat()}",
            content=insight_content,
            metadata=response_payload,
            tags=["reflection"],
        )
        _persist_checkup_payload_to_db("evening", checkup_date, response_payload)

        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate evening checkup: {str(e)}")


@router.get("/checkups/latest")
async def get_latest_checkups():
    """Get latest morning/evening checkups with database-first retrieval."""
    try:
        kb_service = get_knowledge_base_service()
        latest_checkups = await _resolve_latest_checkups_with_fallback(kb_service)
        latest_from_db = _load_latest_checkups_from_db()

        source = "knowledge_base"
        if latest_from_db.get("morning") or latest_from_db.get("evening"):
            source = "database"

        return {
            "morning": latest_checkups.get("morning"),
            "evening": latest_checkups.get("evening"),
            "source": source,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load latest checkups: {str(e)}")


@router.post("/onboarding")
async def save_onboarding_data(data: OnboardingData):
    """Save user onboarding data to knowledge base."""
    try:
        kb_service = get_knowledge_base_service()

        coach_preferences = data.coach_preferences or data.coachPreferences or {}
        if not isinstance(coach_preferences, dict):
            coach_preferences = {}

        domain_preferences = data.domain_preferences or data.domainPreferences or {}
        if not isinstance(domain_preferences, dict):
            domain_preferences = {}

        preference_profile = data.preference_profile or data.preferenceProfile or {}
        if not isinstance(preference_profile, dict):
            preference_profile = {}
        
        # First, delete existing onboarding entries in bulk to avoid repeated index rebuilds.
        all_entries = await kb_service.get_all_entries()
        user_pref_entries = [e for e in all_entries if e.entry_type == KnowledgeEntryType.USER_PREFERENCE]

        if user_pref_entries:
            try:
                await kb_service.delete_entries([entry.entry_id for entry in user_pref_entries])
            except Exception as e:
                logger.warning("Failed to bulk delete onboarding entries: %s", e)
        
        # Generate deterministic sync key for profile (prevents duplicates on re-sync)
        profile_sync_key = f"onboarding:profile:{data.role}"
        
        # Now save new user profile
        profile_entry = await kb_service.create_entry(
            entry_type=KnowledgeEntryType.USER_PREFERENCE,
            entry_sub_type=KnowledgeEntrySubType.USER_PROFILE,
            category="user_profile",
            title="User Profile",
            content=f"Role: {data.role}\nPreferences: {', '.join(data.preferences)}\nMentor: {data.mentor.get('name', 'AI Assistant')}",
            metadata={
                "role": data.role,
                "preferences": data.preferences,
                "mentor": data.mentor,
                "preferredTone": data.preferredTone,
                "coach_preferences": coach_preferences,
                "domain_preferences": domain_preferences,
                "preference_profile": preference_profile,
                "onboarding_completed": True,
                "context": {"sync_event_key": profile_sync_key}
            },
            tags=["profile", "onboarding", data.role.lower()]
        )
        
        # Save each goal with deterministic sync key
        goal_entries = []
        for goal in data.goals:
            goal_id = goal.get('id') or goal.get('title', 'untitled').lower().replace(' ', '_')
            goal_sync_key = f"onboarding:goal:{goal_id}"
            goal_entry = await kb_service.create_entry(
                entry_type=KnowledgeEntryType.USER_PREFERENCE,
                entry_sub_type=KnowledgeEntrySubType.GOAL,
                category="goals",
                title=goal.get('title', 'Untitled Goal'),
                content=f"{goal.get('title', 'Untitled Goal')}: {goal.get('description', '')}",
                metadata={
                    "priority": goal.get('priority', 'Medium'),
                    "category": goal.get('category', data.role),
                    "milestones": goal.get('milestones', []),
                    "smart_criteria": goal.get('smartCriteria', {}),
                    "context": {"sync_event_key": goal_sync_key}
                },
                tags=["goal", data.role.lower(), goal.get('priority', 'medium').lower()]
            )
            goal_entries.append(goal_entry)
        
        # Save planner configuration with deterministic sync key
        planner_sync_key = f"onboarding:planner:{data.role}"
        planner_entry = await kb_service.create_entry(
            entry_type=KnowledgeEntryType.USER_PREFERENCE,
            entry_sub_type=KnowledgeEntrySubType.SCHEDULE,
            category="planner",
            title="Planner Configuration",
            content=f"Work Hours: {data.planner.get('availability', {}).get('workHours', {}).get('start', '09:00')} - {data.planner.get('availability', {}).get('workHours', {}).get('end', '17:00')}\nTimezone: {data.planner.get('availability', {}).get('timezone', 'UTC')}",
            metadata={
                "availability": data.planner.get('availability', {}),
                "notifications": data.planner.get('notifications', {}),
                "integrations": data.planner.get('integrations', {}),
                "context": {"sync_event_key": planner_sync_key}
            },
            tags=["planner", "schedule", "configuration"]
        )

        preferences_synced = False
        try:
            resolved_profile: Dict[str, Dict[str, Any]] = {
                section: values
                for section, values in preference_profile.items()
                if isinstance(values, dict)
            }

            if not resolved_profile:
                availability = data.planner.get("availability", {}) if isinstance(data.planner, dict) else {}
                work_hours = availability.get("workHours", {}) if isinstance(availability, dict) else {}
                check_in = availability.get("checkIn", {}) if isinstance(availability, dict) else {}
                timezone_name = availability.get("timezone", "UTC") if isinstance(availability, dict) else "UTC"

                work_hours_value = f"{work_hours.get('start', '09:00')}-{work_hours.get('end', '17:00')}"
                check_in_time = check_in.get("preferredTime", "09:00")
                check_in_frequency = check_in.get("frequency", "daily")

                resolved_profile = {
                    "productivity": {
                        "work_hours": work_hours_value,
                        "check_in_time": check_in_time,
                        "check_in_frequency": check_in_frequency,
                        "priority_signals": data.preferences,
                    },
                    "health": {
                        "wellness_focus": coach_preferences.get("wellnessFocus", "balanced"),
                    },
                    "finance": {
                        "planning_priority": coach_preferences.get("financialFocus", "budgeting"),
                    },
                    "journal": {
                        "reflection_frequency": check_in_frequency,
                        "check_in_time": check_in_time,
                    },
                    "general": {
                        "role": data.role,
                        "timezone": timezone_name,
                        "priorities": data.preferences,
                        "mentor": data.mentor,
                        "preferred_tone": data.preferredTone,
                        "coach_preferences": coach_preferences,
                    },
                }

            for section in ["productivity", "health", "finance", "journal", "general"]:
                incoming_section = domain_preferences.get(section)
                if isinstance(incoming_section, dict):
                    section_base = resolved_profile.get(section, {})
                    if not isinstance(section_base, dict):
                        section_base = {}
                    section_base.update(incoming_section)
                    resolved_profile[section] = section_base

            current_preferences = await kb_service.get_user_preferences()
            preferences_payload = current_preferences.model_dump()

            for section, values in resolved_profile.items():
                if not isinstance(values, dict):
                    continue
                existing_section = preferences_payload.get(section, {})
                if not isinstance(existing_section, dict):
                    existing_section = {}
                existing_section.update(values)
                preferences_payload[section] = existing_section

            updated_preferences = UserPreferences(**preferences_payload)
            preferences_synced = await kb_service.update_user_preferences(updated_preferences)
        except Exception as pref_sync_error:
            logger.warning("Failed to sync structured onboarding preferences: %s", pref_sync_error)
        
        return {
            "success": True,
            "message": "Onboarding data saved successfully",
            "profile_id": profile_entry.entry_id,
            "goals_count": len(goal_entries),
            "planner_id": planner_entry.entry_id,
            "preferences_synced": preferences_synced
        }
    except Exception as e:
        import traceback
        error_detail = f"Failed to save onboarding data: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@router.get("/onboarding/profile")
async def get_onboarding_profile():
    """Retrieve user's onboarding profile from knowledge base."""
    try:
        kb_service = get_knowledge_base_service()
        
        # Get all user preference entries
        all_entries = await kb_service.get_all_entries()
        user_entries = [e for e in all_entries if e.entry_type == KnowledgeEntryType.USER_PREFERENCE]
        
        if not user_entries:
            return {
                "role": None,
                "goals": [],
                "answers": [],
                "mentor": {},
                "planner": {},
                "preferences": [],
                "preferredTone": None,
                "coachAvatar": None,
                "schedule": None,
                "onboardingCompleted": False,
            }
        
        # Reconstruct profile from entries
        profile_data = {
            "role": None,
            "goals": [],
            "answers": [],
            "mentor": {},
            "planner": {},
            "preferences": [],
            "preferredTone": None,
            "coachAvatar": None,
            "schedule": None,
            "coachPreferences": {},
            "domainPreferences": {},
            "preferenceProfile": {}
        }
        
        for entry in user_entries:
            if entry.entry_sub_type == KnowledgeEntrySubType.USER_PROFILE:
                metadata = entry.metadata or {}
                profile_data["role"] = metadata.get("role")
                profile_data["preferences"] = metadata.get("preferences", [])
                profile_data["mentor"] = metadata.get("mentor", {})
                profile_data["preferredTone"] = metadata.get("preferredTone") or metadata.get("preferred_tone")
                profile_data["coachAvatar"] = metadata.get("mentor", {}).get("avatar")
                profile_data["coachPreferences"] = metadata.get("coach_preferences", {})
                profile_data["domainPreferences"] = metadata.get("domain_preferences", {})
                profile_data["preferenceProfile"] = metadata.get("preference_profile", {})
                # Convert preferences to Answer format
                for pref in metadata.get("preferences", []):
                    profile_data["answers"].append({
                        "id": f"pref-{len(profile_data['answers'])}",
                        "answer": pref,
                        "description": f"Priority: {pref}"
                    })
                
            elif entry.entry_sub_type == KnowledgeEntrySubType.GOAL:
                metadata = entry.metadata or {}
                profile_data["goals"].append({
                    "id": entry.entry_id,
                    "title": entry.title,
                    "description": entry.content.split(": ", 1)[1] if ": " in entry.content else entry.content,
                    "category": metadata.get("category", "General"),
                    "priority": metadata.get("priority", "Medium"),
                    "milestones": metadata.get("milestones", []),
                    "smartCriteria": metadata.get("smart_criteria", {}),
                    "linkedPriorities": metadata.get("linkedPriorities", [])
                })
                
            elif entry.entry_sub_type == KnowledgeEntrySubType.SCHEDULE:
                metadata = entry.metadata or {}
                profile_data["planner"] = {
                    "goals": profile_data["goals"],  # Will be populated after processing all entries
                    "availability": metadata.get("availability", {}),
                    "notifications": metadata.get("notifications", {}),
                    "integrations": metadata.get("integrations", {})
                }
                profile_data["schedule"] = metadata.get("availability", {})
        
        # Update planner goals after all goals are collected
        if profile_data["planner"]:
            profile_data["planner"]["goals"] = profile_data["goals"]

        if not profile_data["preferenceProfile"]:
            try:
                preferences = await kb_service.get_user_preferences()
                profile_data["preferenceProfile"] = {
                    "productivity": preferences.productivity,
                    "health": preferences.health,
                    "finance": preferences.finance,
                    "journal": preferences.journal,
                    "general": preferences.general,
                }
            except Exception:
                profile_data["preferenceProfile"] = {}
        
        profile_data["onboardingCompleted"] = profile_data["role"] is not None
        return profile_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error retrieving profile: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)
