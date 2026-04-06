"""
API endpoints for knowledge base operations.
"""

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
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
        entries = await kb_service.get_all_entries(category=category, entry_type=entry_type)
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


def _parse_requested_date(date_token: Optional[str]) -> date:
    """Parse YYYY-MM-DD dates from API payloads."""
    if not date_token:
        return datetime.now(timezone.utc).date()

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
) -> Optional[str]:
    """Generate optional LLM-enhanced coaching text, if provider is initialized."""
    try:
        from app.llm import service as llm_service_module
        from app.llm.base import CompletionRequest, ChatMessage

        llm_service = llm_service_module._llm_service
        if not llm_service or not llm_service._initialized:
            return None

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

    normalized_tags = sorted(set([*tags, "daily_checkup", checkup_type]))

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
        all_entries = await kb_service.get_all_entries()

        days = _resolve_date_range(time_range)
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days - 1)

        # Normalize datetimes and sort once for deterministic analytics output.
        entries_with_ts = []
        for entry in all_entries:
            entry_ts = _resolve_entry_event_timestamp(entry)
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


@router.post("/checkups/morning")
async def run_morning_checkup(request: DailyCheckupRequest):
    """Generate a morning planning checkup using persisted knowledge and time-entry context."""
    try:
        kb_service = get_knowledge_base_service()
        checkup_date = _parse_requested_date(request.date)
        note = (request.note or "").strip()
        perspective = request.perspective if isinstance(request.perspective, dict) else {}
        context_snapshot = request.context_snapshot if isinstance(request.context_snapshot, dict) else {}
        if not context_snapshot and isinstance(request.contextSnapshot, dict):
            context_snapshot = request.contextSnapshot

        all_entries = await kb_service.get_all_entries()
        time_entries: List[Dict[str, Any]] = []

        for entry in all_entries:
            if _normalize_entry_category(entry) != "time_entry":
                continue

            event_ts = _resolve_entry_event_timestamp(entry)
            event_date = event_ts.date()
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

        preferences = await kb_service.get_user_preferences()
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

        fallback_lines = [
            f"Primary focus: {focus_target}",
            f"Anchor check-in time: {check_in_time}",
            f"Suggested work window: {work_hours}",
            f"Last 7-day average logged work: {_format_minutes(avg_daily_minutes)}",
        ]
        if overdue_tasks > 0:
            fallback_lines.append(f"Overdue tasks to clear first: {overdue_tasks}")
        if due_today_tasks > 0:
            fallback_lines.append(f"Tasks due today: {due_today_tasks}")
        if planned_deep_work_minutes > 0:
            fallback_lines.append(f"Planned deep work: {_format_minutes(planned_deep_work_minutes)}")
        if confidence_score > 0:
            fallback_lines.append(f"Self-reported confidence: {round(confidence_score, 1)}/10")
        if top_projects:
            fallback_lines.append(f"Keep momentum on: {', '.join(top_projects)}")
        if avg_focus_score is not None:
            fallback_lines.append(f"Recent focus baseline: {avg_focus_score}/10")

        llm_prompt = (
            f"Date: {checkup_date.isoformat()}\n"
            f"Intent note: {note or 'none'}\n"
            f"Communication profile: {style_directive}\n"
            f"Focus target: {focus_target}\n"
            f"Work hours: {work_hours}\n"
            f"Check-in time: {check_in_time}\n"
            f"Last 7 days logged minutes: {round(total_week_minutes, 1)}\n"
            f"Average daily logged minutes: {round(avg_daily_minutes, 1)}\n"
            f"Top projects: {', '.join(top_projects) if top_projects else 'none'}\n"
            f"Top goals from context: {top_goals_text}\n"
            f"Overdue tasks: {overdue_tasks}\n"
            f"Due today tasks: {due_today_tasks}\n"
            f"Planned deep work minutes: {planned_deep_work_minutes}\n"
            f"User confidence: {confidence_score if confidence_score > 0 else 'n/a'}\n"
            f"Today existing entries: {len(today_entries)}\n"
            "Create a practical morning checkup with: 1) one focus sentence, "
            "2) three action bullets, 3) one accountability question."
        )

        llm_message = await _generate_checkup_message(llm_prompt, style_directive=style_directive)
        coach_message = llm_message or _build_fallback_checkup_message(fallback_lines, communication_profile, "morning")

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
                "planned_deep_work_minutes": round(planned_deep_work_minutes, 1),
                "confidence": round(confidence_score, 1) if confidence_score > 0 else None,
            },
            "perspective": perspective,
            "context_snapshot": context_snapshot,
            "style_profile": _public_style_profile(communication_profile),
            "coach_message": coach_message,
            "generated_with": "llm" if llm_message else "fallback",
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
            tags=["planning", "time_entry"],
        )

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
        checkup_date = _parse_requested_date(request.date)
        note = (request.note or "").strip()
        perspective = request.perspective if isinstance(request.perspective, dict) else {}
        context_snapshot = request.context_snapshot if isinstance(request.context_snapshot, dict) else {}
        if not context_snapshot and isinstance(request.contextSnapshot, dict):
            context_snapshot = request.contextSnapshot

        all_entries = await kb_service.get_all_entries()
        today_entries: List[Dict[str, Any]] = []

        for entry in all_entries:
            if _normalize_entry_category(entry) != "time_entry":
                continue

            event_ts = _resolve_entry_event_timestamp(entry)
            if event_ts.date() != checkup_date:
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
        if overdue_tasks > 0:
            fallback_lines.append(f"Overdue tasks carried: {overdue_tasks}")
        fallback_lines.extend([f"Tomorrow: {item}" for item in tomorrow_focus])

        communication_profile = _extract_communication_profile(all_entries)
        style_directive = _build_style_directive(communication_profile, "evening")

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
            f"Overdue tasks: {overdue_tasks}\n"
            f"Due today tasks: {due_today_tasks}\n"
            f"Completed tasks today: {completed_tasks_today}\n"
            f"Planned deep work minutes: {planned_deep_work_minutes}\n"
            f"Self-rating: {subjective_score if subjective_score is not None else 'n/a'}\n"
            f"Objective score: {objective_score}/10\n"
            "Provide an evening checkup with: 1) one recap sentence, "
            "2) two wins, 3) two concrete tomorrow actions."
        )

        llm_message = await _generate_checkup_message(llm_prompt, style_directive=style_directive)
        coach_message = llm_message or _build_fallback_checkup_message(fallback_lines, communication_profile, "evening")

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
            "perspective": perspective,
            "context_snapshot": context_snapshot,
            "wins": wins,
            "blockers": blockers,
            "tomorrow_focus": tomorrow_focus,
            "timeline": timeline,
            "style_profile": _public_style_profile(communication_profile),
            "coach_message": coach_message,
            "generated_with": "llm" if llm_message else "fallback",
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
            tags=["reflection", "time_entry"],
        )

        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate evening checkup: {str(e)}")


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
                "onboarding_completed": True
            },
            tags=["profile", "onboarding", data.role.lower()]
        )
        
        # Save each goal
        goal_entries = []
        for goal in data.goals:
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
                    "smart_criteria": goal.get('smartCriteria', {})
                },
                tags=["goal", data.role.lower(), goal.get('priority', 'medium').lower()]
            )
            goal_entries.append(goal_entry)
        
        # Save planner configuration
        planner_entry = await kb_service.create_entry(
            entry_type=KnowledgeEntryType.USER_PREFERENCE,
            entry_sub_type=KnowledgeEntrySubType.SCHEDULE,
            category="planner",
            title="Planner Configuration",
            content=f"Work Hours: {data.planner.get('availability', {}).get('workHours', {}).get('start', '09:00')} - {data.planner.get('availability', {}).get('workHours', {}).get('end', '17:00')}\nTimezone: {data.planner.get('availability', {}).get('timezone', 'UTC')}",
            metadata={
                "availability": data.planner.get('availability', {}),
                "notifications": data.planner.get('notifications', {}),
                "integrations": data.planner.get('integrations', {})
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
