"""
API endpoints for knowledge base operations.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
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
            entry_ts = entry.created_at
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
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
                if isinstance(duration_minutes, (int, float)):
                    time_entry_total_minutes += float(duration_minutes)

                if bool(context.get("billable", False)):
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


@router.post("/onboarding")
async def save_onboarding_data(data: OnboardingData):
    """Save user onboarding data to knowledge base."""
    try:
        kb_service = get_knowledge_base_service()
        
        # First, delete all existing onboarding entries to avoid duplicates
        all_entries = await kb_service.get_all_entries()
        user_pref_entries = [e for e in all_entries if e.entry_type == KnowledgeEntryType.USER_PREFERENCE]
        
        for entry in user_pref_entries:
            try:
                await kb_service.delete_entry(entry.entry_id)
            except Exception as e:
                logger.warning(f"Failed to delete entry {entry.entry_id}: {str(e)}")
        
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
        
        return {
            "success": True,
            "message": "Onboarding data saved successfully",
            "profile_id": profile_entry.entry_id,
            "goals_count": len(goal_entries),
            "planner_id": planner_entry.entry_id
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
            raise HTTPException(status_code=404, detail="No onboarding profile found")
        
        # Reconstruct profile from entries
        profile_data = {
            "role": None,
            "goals": [],
            "answers": [],
            "mentor": {},
            "planner": {},
            "preferences": [],
            "coachAvatar": None,
            "schedule": None
        }
        
        for entry in user_entries:
            if entry.entry_sub_type == KnowledgeEntrySubType.USER_PROFILE:
                metadata = entry.metadata or {}
                profile_data["role"] = metadata.get("role")
                profile_data["preferences"] = metadata.get("preferences", [])
                profile_data["mentor"] = metadata.get("mentor", {})
                profile_data["coachAvatar"] = metadata.get("mentor", {}).get("avatar")
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
        
        return profile_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error retrieving profile: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)
