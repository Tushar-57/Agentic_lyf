"""
Tool Registry — per-agent arsenals backed by the knowledge base.

Each specialized agent (productivity, health, finance) gets its own complete
toolkit. Tools fetch pre-computed intelligence (INSIGHT/PATTERN entries +
classified time_entry_v2 vectors) rather than raw data — the LLM picks tools,
not searches against documents.

Tools follow the existing pattern from specialized_productivity_tools.py:
    @tool(parse_docstring=True) → return Command(update={"messages": [...]})
    Annotated[DeepAgentState, InjectedState] for user_id access
    Annotated[str, InjectedToolCallId] for tool message correlation

Tools are produced by factory functions that close over BrainService and KB
instances, so service references aren't smuggled through DeepAgentState.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

from app.models.knowledge import KnowledgeEntryType
from .deep_state import DeepAgentState

logger = logging.getLogger(__name__)


# ─── Helpers shared across tools ───────────────────────────────────────────

def _ok(tool_call_id: str, payload: Any, key: Optional[str] = None) -> Command:
    """Wrap a tool result as a ToolMessage Command. Compact JSON for the LLM."""
    if isinstance(payload, str):
        body = payload
    else:
        try:
            body = json.dumps(payload, default=str, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            body = str(payload)
    msg = ToolMessage(content=body, tool_call_id=tool_call_id)
    update: Dict[str, Any] = {"messages": [msg]}
    if key:
        update["agent_contexts"] = {key: payload}
    return Command(update=update)


def _err(tool_call_id: str, message: str) -> Command:
    return Command(
        update={
            "messages": [ToolMessage(content=f"⚠️ {message}", tool_call_id=tool_call_id)]
        }
    )


def _meta(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, dict):
        return entry.get("metadata") or {}
    md = getattr(entry, "metadata", None) or {}
    return md if isinstance(md, dict) else {}


def _content(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("content") or "")
    return str(getattr(entry, "content", "") or "")


def _entry_dt(entry: Any) -> Optional[datetime]:
    md = _meta(entry)
    candidate = md.get("start_time") or md.get("captured_at") or md.get("timestamp") or md.get("generated_at")
    if not candidate:
        return None
    try:
        text = str(candidate).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    # Naive timestamps (no offset, no Z) crash the _filter_recent comparison
    # against datetime.now(timezone.utc). Treat as UTC so all entry datetimes
    # round-trip through the same tz.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _filter_recent(entries: List[Any], days: int) -> List[Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for e in entries:
        dt = _entry_dt(e)
        if dt is None or dt >= cutoff:
            out.append(e)
    return out


# ─── Productivity arsenal (16 tools) ───────────────────────────────────────

def make_productivity_tools(brain, kb) -> List[Callable]:
    """Build the productivity agent's complete arsenal.

    Tools cover time/focus, goals, tasks, behavioral signals, and memory recall.
    """

    @tool(parse_docstring=True)
    async def get_classified_time_entries(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 7,
        work_type: Optional[str] = None,
    ) -> Command:
        """Retrieve recent time entries with their pre-computed work_type, focus, energy and productivity classification.

        Use when you need to know what kind of work the user has been doing — deep_work vs meetings vs admin — and at what quality.

        Args:
            days: How many days back to fetch (default 7).
            work_type: Optional filter — one of deep_work, shallow_work, meetings, planning, learning, context_switching.

        Returns:
            List of classified entries with project, duration, focus, energy, productivity, goal alignment.
        """
        entries = await kb.get_all_entries(category="time_entry_v2")
        recent = _filter_recent(entries or [], days)
        out = []
        for e in recent:
            md = _meta(e)
            if work_type and str(md.get("work_type", "")).lower() != work_type.lower():
                continue
            out.append({
                "content": _content(e)[:300],
                "work_type": md.get("work_type"),
                "focus_quality": md.get("focus_quality"),
                "energy_pattern": md.get("energy_pattern"),
                "productivity_score": md.get("productivity_score"),
                "duration_minutes": md.get("duration_minutes"),
                "linked_goal": md.get("linked_goal"),
                "weekday": md.get("weekday"),
                "hour_of_day": md.get("hour_of_day"),
                "start_time": md.get("start_time"),
            })
        return _ok(tool_call_id, {"days": days, "work_type": work_type, "count": len(out), "entries": out[:50]}, "classified_entries")

    @tool(parse_docstring=True)
    async def get_productivity_trend(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 14,
    ) -> Command:
        """Aggregate daily average productivity score and report the trend direction.

        Args:
            days: Window in days (default 14).

        Returns:
            Per-day averages and overall trend (improving/declining/flat).
        """
        entries = await kb.get_all_entries(category="time_entry_v2")
        recent = _filter_recent(entries or [], days)
        by_day: Dict[str, List[float]] = defaultdict(list)
        for e in recent:
            md = _meta(e)
            dt = _entry_dt(e)
            score = md.get("productivity_score")
            if dt and score is not None:
                by_day[dt.date().isoformat()].append(float(score))
        daily = {d: round(sum(v) / len(v), 2) for d, v in sorted(by_day.items())}
        if len(daily) >= 4:
            half = len(daily) // 2
            keys = list(daily.keys())
            first_half = sum(daily[k] for k in keys[:half]) / max(half, 1)
            second_half = sum(daily[k] for k in keys[half:]) / max(len(keys) - half, 1)
            delta = second_half - first_half
            trend = "improving" if delta > 0.3 else "declining" if delta < -0.3 else "flat"
        else:
            trend = "insufficient_data"
        return _ok(tool_call_id, {"days": days, "daily_averages": daily, "trend": trend}, "productivity_trend")

    @tool(parse_docstring=True)
    async def get_peak_performance_hours(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Return the hours of the day where the user achieves deep, productive work most consistently.

        Aggregates from classified time entries where focus_quality and productivity_score are both high.

        Returns:
            Ranked list of hours with average productivity and frequency count.
        """
        entries = await kb.get_all_entries(category="time_entry_v2")
        recent = _filter_recent(entries or [], 30)
        bucket: Dict[int, List[float]] = defaultdict(list)
        for e in recent:
            md = _meta(e)
            hr = md.get("hour_of_day")
            prod = md.get("productivity_score")
            focus = md.get("focus_quality")
            if hr is None or prod is None or focus is None:
                continue
            if float(prod) >= 7.0 and float(focus) >= 7.0:
                bucket[int(hr)].append(float(prod))
        ranked = sorted(
            ({"hour": h, "avg_productivity": round(sum(v) / len(v), 2), "count": len(v)} for h, v in bucket.items()),
            key=lambda r: (-r["count"], -r["avg_productivity"]),
        )
        return _ok(tool_call_id, {"peak_hours": ranked[:10]}, "peak_performance_hours")

    @tool(parse_docstring=True)
    async def get_deep_work_profile(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Fetch the latest computed productivity profile insight (deep work capacity, peak windows, focus consistency).

        Returns:
            Most recent productivity_profile INSIGHT entry content + metadata.
        """
        entries = await kb.get_all_entries(category="insight_v2")
        profile_entries = [e for e in entries or [] if "productivity_profile" in (e.tags or [])] if entries else []
        if not profile_entries:
            # Fall back to any productivity_profile by content
            profile_entries = [e for e in entries or [] if "Deep work capacity" in _content(e)]
        if not profile_entries:
            return _ok(tool_call_id, {"profile": None, "note": "no productivity_profile generated yet"}, "deep_work_profile")
        latest = max(profile_entries, key=lambda e: _entry_dt(e) or datetime.min.replace(tzinfo=timezone.utc))
        return _ok(tool_call_id, {"profile": _content(latest), "metadata": _meta(latest)}, "deep_work_profile")

    @tool(parse_docstring=True)
    async def get_focus_target_vs_actual(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        date: Optional[str] = None,
    ) -> Command:
        """Show whether the user followed through on a given day's morning focus target.

        Joins morning_intent + same-day time_entry_v2 + evening_reflection via the commitment chain pattern.

        Args:
            date: ISO date (YYYY-MM-DD). Defaults to today.

        Returns:
            Focus target, related entries logged, follow-through verdict.
        """
        target_date = date or datetime.now(timezone.utc).date().isoformat()
        patterns = await kb.get_all_entries(category="pattern_v2")
        match = None
        for p in patterns or []:
            md = _meta(p)
            if md.get("pattern_type") == "commitment_chain" and md.get("date") == target_date:
                match = p
                break
        if not match:
            return _ok(tool_call_id, {"date": target_date, "result": None, "note": "no commitment chain stored for that date"}, "focus_target_vs_actual")
        return _ok(tool_call_id, {"date": target_date, "summary": _content(match), "metadata": _meta(match)}, "focus_target_vs_actual")

    @tool(parse_docstring=True)
    async def get_active_goals_summary(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """List all goals with computed status (ghost / at_risk / on_track / in_progress) and investment hours.

        Returns:
            Array of goals with title, status, invested_hours, planned hours, days_remaining.
        """
        entries = await kb.get_all_entries(category="goal_v2")
        out = []
        for e in entries or []:
            md = _meta(e)
            out.append({
                "title": getattr(e, "title", ""),
                "goal_id": md.get("goal_id"),
                "status": md.get("status"),
                "invested_hours": md.get("invested_hours"),
                "hours_this_month": md.get("hours_this_month"),
                "days_remaining": md.get("days_remaining"),
                "content": _content(e)[:400],
            })
        return _ok(tool_call_id, {"goals": out}, "active_goals_summary")

    @tool(parse_docstring=True)
    async def get_goal_investment_summary(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        goal_id: Optional[str] = None,
    ) -> Command:
        """Show invested hours vs planned hours for a goal (or for all goals if omitted), plus recent activity.

        Args:
            goal_id: Optional specific goal_id. If omitted, returns summary across all goals.

        Returns:
            Invested vs planned per goal, optionally with last 5 related time entries.
        """
        goals = await kb.get_all_entries(category="goal_v2")
        time_entries = await kb.get_all_entries(category="time_entry_v2")
        goal_results = []
        for g in goals or []:
            md = _meta(g)
            gid = md.get("goal_id")
            if goal_id and str(gid) != str(goal_id):
                continue
            related = []
            for te in time_entries or []:
                temd = _meta(te)
                if str(temd.get("linked_goal")) == str(gid):
                    related.append({
                        "duration_minutes": temd.get("duration_minutes"),
                        "start_time": temd.get("start_time"),
                        "work_type": temd.get("work_type"),
                        "content": _content(te)[:200],
                    })
            goal_results.append({
                "goal_id": gid,
                "title": getattr(g, "title", ""),
                "status": md.get("status"),
                "invested_hours": md.get("invested_hours"),
                "hours_this_month": md.get("hours_this_month"),
                "recent_entries": sorted(related, key=lambda r: r.get("start_time") or "", reverse=True)[:5],
            })
        return _ok(tool_call_id, {"goals": goal_results}, "goal_investment_summary")

    @tool(parse_docstring=True)
    async def get_ghost_goals(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """List goals flagged as ghost (>30 days old, <5h invested) by the intelligence pre-compute layer.

        Returns:
            Ghost goal INSIGHT entries — title, days old, invested hours.
        """
        insights = await kb.get_all_entries(category="insight_v2")
        ghosts = [e for e in insights or [] if _meta(e).get("insight_type") == "ghost_goal"]
        out = [{"summary": _content(e), "metadata": _meta(e)} for e in ghosts]
        return _ok(tool_call_id, {"ghost_goals": out}, "ghost_goals")

    @tool(parse_docstring=True)
    async def get_tasks_summary(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Command:
        """Summary of the user's task pipeline. Filterable by status and priority.

        Args:
            status: Optional filter — todo, in_progress, done, blocked.
            priority: Optional filter — low, medium, high, urgent.

        Returns:
            Counts by status and priority, plus recent tasks.
        """
        entries = await kb.get_all_entries(category="task_entry")
        all_tasks = []
        for e in entries or []:
            md = _meta(e)
            all_tasks.append({
                "title": getattr(e, "title", ""),
                "status": md.get("status"),
                "priority": md.get("priority"),
                "due_date": md.get("due_date"),
                "linked_goal": md.get("linked_goal"),
                "content": _content(e)[:200],
            })
        filtered = [
            t for t in all_tasks
            if (not status or str(t["status"] or "").lower() == status.lower())
            and (not priority or str(t["priority"] or "").lower() == priority.lower())
        ]
        by_status = Counter(str(t["status"] or "unknown") for t in all_tasks)
        by_priority = Counter(str(t["priority"] or "unknown") for t in all_tasks)
        return _ok(tool_call_id, {
            "filter": {"status": status, "priority": priority},
            "matched_count": len(filtered),
            "by_status": dict(by_status),
            "by_priority": dict(by_priority),
            "tasks": filtered[:30],
        }, "tasks_summary")

    @tool(parse_docstring=True)
    async def get_priority_inflation_alert(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Check whether priority inflation has been detected (>70% of recent tasks marked high/urgent).

        Returns:
            Priority inflation INSIGHT if present, else null.
        """
        insights = await kb.get_all_entries(category="insight_v2")
        hits = [e for e in insights or [] if _meta(e).get("insight_type") == "priority_inflation"]
        if not hits:
            return _ok(tool_call_id, {"alert": None}, "priority_inflation")
        latest = max(hits, key=lambda e: _entry_dt(e) or datetime.min.replace(tzinfo=timezone.utc))
        return _ok(tool_call_id, {"alert": _content(latest), "metadata": _meta(latest)}, "priority_inflation")

    @tool(parse_docstring=True)
    async def get_recurring_blockers(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Return blockers that have appeared 3+ times in the last 14 days (from PATTERN entries).

        Returns:
            List of recurring blocker patterns with frequency.
        """
        patterns = await kb.get_all_entries(category="pattern_v2")
        blockers = [e for e in patterns or [] if _meta(e).get("pattern_type") == "recurring_blocker"]
        out = [{"summary": _content(e), "metadata": _meta(e)} for e in blockers]
        return _ok(tool_call_id, {"recurring_blockers": out}, "recurring_blockers")

    @tool(parse_docstring=True)
    async def get_behavioral_drift(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Check whether productivity has drifted week-over-week (current 7d vs prior 7d).

        Returns:
            Drift INSIGHT entries (most recent first).
        """
        insights = await kb.get_all_entries(category="insight_v2")
        drift = [e for e in insights or [] if _meta(e).get("insight_type") == "behavioral_drift"]
        drift_sorted = sorted(drift, key=lambda e: _entry_dt(e) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        out = [{"summary": _content(e), "metadata": _meta(e)} for e in drift_sorted[:5]]
        return _ok(tool_call_id, {"drift_insights": out}, "behavioral_drift")

    @tool(parse_docstring=True)
    async def get_commitment_chain(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        date: Optional[str] = None,
    ) -> Command:
        """Retrieve the morning intent → time entries → evening reflection chain for a date.

        Args:
            date: ISO date (defaults to today).

        Returns:
            Joined commitment chain pattern entry.
        """
        target = date or datetime.now(timezone.utc).date().isoformat()
        patterns = await kb.get_all_entries(category="pattern_v2")
        for p in patterns or []:
            md = _meta(p)
            if md.get("pattern_type") == "commitment_chain" and md.get("date") == target:
                return _ok(tool_call_id, {"date": target, "chain": _content(p), "metadata": md}, "commitment_chain")
        return _ok(tool_call_id, {"date": target, "chain": None}, "commitment_chain")

    @tool(parse_docstring=True)
    async def search_behavioral_insights(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        query: str,
    ) -> Command:
        """Semantic search over INSIGHT and PATTERN entries.

        Args:
            query: What you want to know about the user's behavior.

        Returns:
            Top matching insights/patterns.
        """
        results = await brain.recall_semantic(state.get("user_id", "single_user"), query, k=8)
        out = [{"content": _content(r), "metadata": _meta(r), "tags": getattr(r, "tags", [])} for r in results]
        return _ok(tool_call_id, {"query": query, "results": out}, "behavioral_insights")

    @tool(parse_docstring=True)
    async def recall_past_conversation(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        topic: str,
    ) -> Command:
        """Search the user's past chat turns for prior discussion of a topic.

        Args:
            topic: What you want to recall from prior conversations.

        Returns:
            Matching past chat turns with the agent's prior responses in metadata.
        """
        results = await brain.recall_episodic(state.get("user_id", "single_user"), topic, k=5)
        out = [{
            "user_said": _content(r),
            "agent_response": _meta(r).get("agent_response"),
            "timestamp": _meta(r).get("timestamp"),
            "agent_type": _meta(r).get("agent_type"),
        } for r in results]
        return _ok(tool_call_id, {"topic": topic, "turns": out}, "past_conversations")

    @tool(parse_docstring=True)
    async def get_user_priorities(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Fetch the user's stated priorities from preference entries.

        Returns:
            Per-question preference content (Q + A + detail).
        """
        entries = await kb.get_all_entries(category="user_preference")
        out = [{"content": _content(e), "metadata": _meta(e)} for e in entries or []]
        return _ok(tool_call_id, {"preferences": out}, "user_priorities")

    return [
        get_classified_time_entries,
        get_productivity_trend,
        get_peak_performance_hours,
        get_deep_work_profile,
        get_focus_target_vs_actual,
        get_active_goals_summary,
        get_goal_investment_summary,
        get_ghost_goals,
        get_tasks_summary,
        get_priority_inflation_alert,
        get_recurring_blockers,
        get_behavioral_drift,
        get_commitment_chain,
        search_behavioral_insights,
        recall_past_conversation,
        get_user_priorities,
    ]


# ─── Health arsenal (13 tools) ─────────────────────────────────────────────

def make_health_tools(brain, kb) -> List[Callable]:
    """Build the health agent's complete arsenal."""

    @tool(parse_docstring=True)
    async def get_habit_trend(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        habit_name: str,
    ) -> Command:
        """Fetch a single habit's per-day vectors and recent activity.

        Args:
            habit_name: Habit name (case-insensitive partial match).

        Returns:
            Per-habit entries with completion_7d, completion_30d, streak, last activity.
        """
        entries = await kb.get_all_entries(category="habit_entry")
        target = habit_name.lower()
        matches = []
        for e in entries or []:
            md = _meta(e)
            name = str(md.get("habit_name") or "").lower()
            if target in name:
                matches.append({"content": _content(e), "metadata": md})
        return _ok(tool_call_id, {"habit_name": habit_name, "entries": matches}, "habit_trend")

    @tool(parse_docstring=True)
    async def get_habit_completion_summary(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 7,
    ) -> Command:
        """Aggregate completion rates across all habits over the last N days.

        Args:
            days: Window in days.

        Returns:
            Per-habit completion percentage and overall summary.
        """
        entries = await kb.get_all_entries(category="habit_entry")
        recent = _filter_recent(entries or [], days)
        per_habit: Dict[str, List[float]] = defaultdict(list)
        for e in recent:
            md = _meta(e)
            name = md.get("habit_name") or "unknown"
            comp = md.get("completion_7d") if days <= 7 else md.get("completion_30d")
            if comp is not None:
                per_habit[name].append(float(comp))
        summary = {h: round(sum(v) / len(v), 1) for h, v in per_habit.items()}
        return _ok(tool_call_id, {"days": days, "habits": summary}, "habit_completion_summary")

    @tool(parse_docstring=True)
    async def get_zombie_habits(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """List habits where 30-day completion is below 20% (precomputed insights).

        Returns:
            Zombie habit INSIGHT entries.
        """
        insights = await kb.get_all_entries(category="insight_v2")
        zombies = [e for e in insights or [] if _meta(e).get("insight_type") == "zombie_habit"]
        out = [{"summary": _content(e), "metadata": _meta(e)} for e in zombies]
        return _ok(tool_call_id, {"zombie_habits": out}, "zombie_habits")

    @tool(parse_docstring=True)
    async def get_streak_overview(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Show all active habit streaks ranked by length.

        Returns:
            Habits sorted by current streak (descending).
        """
        entries = await kb.get_all_entries(category="habit_entry")
        latest_per_habit: Dict[str, Any] = {}
        for e in entries or []:
            md = _meta(e)
            name = md.get("habit_name")
            if not name:
                continue
            current_dt = _entry_dt(e) or datetime.min.replace(tzinfo=timezone.utc)
            existing = latest_per_habit.get(name)
            if not existing or (_entry_dt(existing) or datetime.min.replace(tzinfo=timezone.utc)) < current_dt:
                latest_per_habit[name] = e
        ranked = sorted(
            ({"habit_name": n, "streak": _meta(e).get("streak", 0), "completion_7d": _meta(e).get("completion_7d")} for n, e in latest_per_habit.items()),
            key=lambda r: -int(r["streak"] or 0),
        )
        return _ok(tool_call_id, {"streaks": ranked}, "streak_overview")

    @tool(parse_docstring=True)
    async def get_habit_pattern(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        habit_name: str,
    ) -> Command:
        """Identify when in the day/week the user actually performs a habit.

        Args:
            habit_name: Habit name.

        Returns:
            Pattern summary based on per-habit vectors.
        """
        entries = await kb.get_all_entries(category="habit_entry")
        target = habit_name.lower()
        days_active = []
        for e in entries or []:
            md = _meta(e)
            name = str(md.get("habit_name") or "").lower()
            if target in name and md.get("last_completed"):
                try:
                    raw = str(md["last_completed"]).strip()
                    if raw.endswith("Z"):
                        raw = raw[:-1] + "+00:00"
                    parsed = datetime.fromisoformat(raw)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    days_active.append(parsed.strftime("%A"))
                except (ValueError, TypeError):
                    continue
        weekday_counts = Counter(days_active)
        return _ok(tool_call_id, {"habit_name": habit_name, "weekday_distribution": dict(weekday_counts)}, "habit_pattern")

    @tool(parse_docstring=True)
    async def get_energy_pattern(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 14,
    ) -> Command:
        """Distribution of energy_pattern (high_focus / low_energy / etc.) across recent time entries.

        Args:
            days: Window.

        Returns:
            Energy pattern → minutes, ranked.
        """
        entries = await kb.get_all_entries(category="time_entry_v2")
        recent = _filter_recent(entries or [], days)
        bucket: Dict[str, float] = defaultdict(float)
        for e in recent:
            md = _meta(e)
            ep = md.get("energy_pattern") or "unknown"
            bucket[ep] += float(md.get("duration_minutes") or 0)
        ranked = sorted(bucket.items(), key=lambda kv: -kv[1])
        return _ok(tool_call_id, {"days": days, "energy_pattern_minutes": dict(ranked)}, "energy_pattern")

    @tool(parse_docstring=True)
    async def get_morning_energy_trend(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 14,
    ) -> Command:
        """Morning energy scores from morning_intent vectors over the last N days.

        Args:
            days: Window.

        Returns:
            Per-day morning energy.
        """
        entries = await kb.get_all_entries(category="morning_intent")
        recent = _filter_recent(entries or [], days)
        per_day = []
        for e in recent:
            md = _meta(e)
            per_day.append({
                "date": md.get("checkup_date"),
                "energy": md.get("morning_energy") or md.get("energy"),
                "focus_target": md.get("focus_target"),
            })
        per_day.sort(key=lambda r: r.get("date") or "")
        return _ok(tool_call_id, {"days": days, "entries": per_day}, "morning_energy_trend")

    @tool(parse_docstring=True)
    async def get_evening_mood_summary(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 14,
    ) -> Command:
        """Wins / blockers / day rating from evening_reflection vectors.

        Args:
            days: Window.

        Returns:
            Per-day evening reflection summary.
        """
        entries = await kb.get_all_entries(category="evening_reflection")
        recent = _filter_recent(entries or [], days)
        out = []
        for e in recent:
            md = _meta(e)
            out.append({
                "date": md.get("checkup_date"),
                "wins": md.get("wins"),
                "blockers": md.get("blockers"),
                "follow_through": md.get("follow_through"),
                "content": _content(e),
            })
        return _ok(tool_call_id, {"days": days, "entries": out}, "evening_mood")

    @tool(parse_docstring=True)
    async def get_mood_shifts(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 14,
    ) -> Command:
        """Morning→evening mood/energy deltas over the last N days.

        Args:
            days: Window.

        Returns:
            Per-day deltas.
        """
        morn = await kb.get_all_entries(category="morning_intent")
        eve = await kb.get_all_entries(category="evening_reflection")
        morn_recent = {_meta(e).get("checkup_date"): _meta(e) for e in _filter_recent(morn or [], days)}
        eve_recent = {_meta(e).get("checkup_date"): _meta(e) for e in _filter_recent(eve or [], days)}
        deltas = []
        for date in sorted(set(morn_recent.keys()) & set(eve_recent.keys())):
            m = morn_recent[date]
            v = eve_recent[date]
            try:
                m_score = float(m.get("morning_energy") or m.get("energy") or 0)
                e_score = float(v.get("evening_energy") or v.get("evening_score") or 0)
                deltas.append({"date": date, "morning": m_score, "evening": e_score, "delta": round(e_score - m_score, 1)})
            except (ValueError, TypeError):
                continue
        return _ok(tool_call_id, {"days": days, "deltas": deltas}, "mood_shifts")

    @tool(parse_docstring=True)
    async def get_wellness_blockers(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Recurring blocker patterns tagged with health/wellness keywords.

        Returns:
            Wellness-related blocker patterns.
        """
        patterns = await kb.get_all_entries(category="pattern_v2")
        wellness_keywords = ("sleep", "tired", "stress", "anxiety", "energy", "exhaustion", "headache", "pain")
        out = []
        for p in patterns or []:
            md = _meta(p)
            if md.get("pattern_type") != "recurring_blocker":
                continue
            text = (md.get("blocker_text") or "").lower()
            if any(kw in text for kw in wellness_keywords):
                out.append({"summary": _content(p), "metadata": md})
        return _ok(tool_call_id, {"wellness_blockers": out}, "wellness_blockers")

    @tool(parse_docstring=True)
    async def search_behavioral_insights(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        query: str,
    ) -> Command:
        """Semantic search over INSIGHT/PATTERN entries.

        Args:
            query: What to search for.

        Returns:
            Top matches.
        """
        results = await brain.recall_semantic(state.get("user_id", "single_user"), query, k=8)
        out = [{"content": _content(r), "metadata": _meta(r)} for r in results]
        return _ok(tool_call_id, {"query": query, "results": out}, "behavioral_insights")

    @tool(parse_docstring=True)
    async def recall_past_conversation(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        topic: str,
    ) -> Command:
        """Search past chat turns for prior discussion of a topic.

        Args:
            topic: What to recall.

        Returns:
            Past matching turns with agent responses in metadata.
        """
        results = await brain.recall_episodic(state.get("user_id", "single_user"), topic, k=5)
        out = [{"user_said": _content(r), "agent_response": _meta(r).get("agent_response"), "timestamp": _meta(r).get("timestamp")} for r in results]
        return _ok(tool_call_id, {"topic": topic, "turns": out}, "past_conversations")

    @tool(parse_docstring=True)
    async def get_user_priorities(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Fetch user's stated priorities from preference entries.

        Returns:
            Preferences relevant to wellbeing.
        """
        entries = await kb.get_all_entries(category="user_preference")
        out = [{"content": _content(e), "metadata": _meta(e)} for e in entries or []]
        return _ok(tool_call_id, {"preferences": out}, "user_priorities")

    return [
        get_habit_trend,
        get_habit_completion_summary,
        get_zombie_habits,
        get_streak_overview,
        get_habit_pattern,
        get_energy_pattern,
        get_morning_energy_trend,
        get_evening_mood_summary,
        get_mood_shifts,
        get_wellness_blockers,
        search_behavioral_insights,
        recall_past_conversation,
        get_user_priorities,
    ]


# ─── Finance arsenal (10 tools — scaffolded) ───────────────────────────────

def make_finance_tools(brain, kb) -> List[Callable]:
    """Build the finance agent's complete arsenal (scaffolded — activates when finance data syncs)."""

    @tool(parse_docstring=True)
    async def get_spending_trend(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 30,
    ) -> Command:
        """Spending trend over the last N days. Empty until finance data is synced.

        Args:
            days: Window.

        Returns:
            Per-day spending summary.
        """
        entries = await kb.get_all_entries(category="finance_transaction")
        recent = _filter_recent(entries or [], days)
        return _ok(tool_call_id, {"days": days, "transactions": len(recent), "note": "Finance sync pending."}, "spending_trend")

    @tool(parse_docstring=True)
    async def get_budget_status(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Current month budget vs actual.

        Returns:
            Per-category budget snapshot.
        """
        entries = await kb.get_all_entries(category="finance_budget")
        out = [{"content": _content(e), "metadata": _meta(e)} for e in entries or []]
        return _ok(tool_call_id, {"budgets": out, "note": "Finance sync pending." if not out else None}, "budget_status")

    @tool(parse_docstring=True)
    async def get_recurring_expenses(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Recurring expense patterns.

        Returns:
            Identified recurring expenses.
        """
        patterns = await kb.get_all_entries(category="pattern_v2")
        recurring = [e for e in patterns or [] if _meta(e).get("pattern_type") == "recurring_expense"]
        out = [{"summary": _content(e), "metadata": _meta(e)} for e in recurring]
        return _ok(tool_call_id, {"recurring": out}, "recurring_expenses")

    @tool(parse_docstring=True)
    async def get_income_summary(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        days: int = 30,
    ) -> Command:
        """Income summary over the window.

        Args:
            days: Window.

        Returns:
            Income totals (empty until sync exists).
        """
        return _ok(tool_call_id, {"days": days, "note": "Finance sync pending."}, "income_summary")

    @tool(parse_docstring=True)
    async def get_savings_progress(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Progress toward savings goals.

        Returns:
            Savings progress per goal.
        """
        goals = await kb.get_all_entries(category="goal_v2")
        savings = [g for g in goals or [] if "savings" in (str(_meta(g).get("category", "")).lower() + str(getattr(g, "title", "")).lower())]
        out = [{"title": getattr(g, "title", ""), "metadata": _meta(g), "content": _content(g)} for g in savings]
        return _ok(tool_call_id, {"savings_goals": out}, "savings_progress")

    @tool(parse_docstring=True)
    async def get_financial_goals(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """All goals tagged finance category.

        Returns:
            Finance-category goals.
        """
        goals = await kb.get_all_entries(category="goal_v2")
        finance = [g for g in goals or [] if str(_meta(g).get("category", "")).lower() in {"finance", "money", "wealth"}]
        out = [{"title": getattr(g, "title", ""), "metadata": _meta(g)} for g in finance]
        return _ok(tool_call_id, {"finance_goals": out}, "financial_goals")

    @tool(parse_docstring=True)
    async def get_financial_blockers(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Recurring blockers tagged with finance keywords.

        Returns:
            Finance-related blockers.
        """
        patterns = await kb.get_all_entries(category="pattern_v2")
        finance_kw = ("money", "budget", "spending", "income", "expense", "debt", "savings")
        out = []
        for p in patterns or []:
            md = _meta(p)
            if md.get("pattern_type") != "recurring_blocker":
                continue
            text = (md.get("blocker_text") or "").lower()
            if any(kw in text for kw in finance_kw):
                out.append({"summary": _content(p), "metadata": md})
        return _ok(tool_call_id, {"financial_blockers": out}, "financial_blockers")

    @tool(parse_docstring=True)
    async def search_behavioral_insights(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        query: str,
    ) -> Command:
        """Semantic search over INSIGHT/PATTERN.

        Args:
            query: Search query.

        Returns:
            Top matches.
        """
        results = await brain.recall_semantic(state.get("user_id", "single_user"), query, k=8)
        out = [{"content": _content(r), "metadata": _meta(r)} for r in results]
        return _ok(tool_call_id, {"query": query, "results": out}, "behavioral_insights")

    @tool(parse_docstring=True)
    async def recall_past_conversation(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        topic: str,
    ) -> Command:
        """Search past chat turns.

        Args:
            topic: Topic to recall.

        Returns:
            Past matching turns.
        """
        results = await brain.recall_episodic(state.get("user_id", "single_user"), topic, k=5)
        out = [{"user_said": _content(r), "agent_response": _meta(r).get("agent_response")} for r in results]
        return _ok(tool_call_id, {"topic": topic, "turns": out}, "past_conversations")

    @tool(parse_docstring=True)
    async def get_user_priorities(
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Fetch user's stated finance priorities.

        Returns:
            Preference entries.
        """
        entries = await kb.get_all_entries(category="user_preference")
        out = [{"content": _content(e), "metadata": _meta(e)} for e in entries or []]
        return _ok(tool_call_id, {"preferences": out}, "user_priorities")

    return [
        get_spending_trend,
        get_budget_status,
        get_recurring_expenses,
        get_income_summary,
        get_savings_progress,
        get_financial_goals,
        get_financial_blockers,
        search_behavioral_insights,
        recall_past_conversation,
        get_user_priorities,
    ]
