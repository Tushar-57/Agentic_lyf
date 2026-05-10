"""
Intelligence pre-computation service.

Runs after every AlterEgo sync. Transforms raw incoming data into the
intelligence-engine retrieval surface:

  1. Classify each new time entry via SmartTimeContextAnalyzer and persist
     the rich classification string as the embedding content (`time_entry_v2`).
  2. Roll up 7d / 30d windows into PATTERN entries (productivity profile).
  3. Detect anomalies → INSIGHT / PATTERN entries with deterministic IDs:
       ghost goals, zombie habits, behavioral drift, recurring blockers,
       priority inflation, deep work peak windows, commitment chains.
  4. Re-embed every goal touched by new time entries (status + invested).
  5. Close the commitment chain for the current day.

Detection logic is intentionally conservative — only writes an INSIGHT when
thresholds clearly cross. All entries use deterministic IDs so re-syncs upsert
in place rather than producing duplicates.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from app.models.knowledge import (
    KnowledgeEntry,
    KnowledgeEntrySubType,
    KnowledgeEntryType,
)

logger = logging.getLogger(__name__)


def _safe_dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _safe_dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe_dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_dataclass_to_dict(v) for v in obj]
    if hasattr(obj, "value"):
        return obj.value
    return obj


class IntelligencePrecomputeService:
    """Post-sync background job. See module docstring."""

    GHOST_GOAL_AGE_DAYS = 30
    GHOST_GOAL_INVESTED_HOURS = 5
    ZOMBIE_HABIT_THRESHOLD = 20.0
    DRIFT_PRODUCTIVITY_DELTA = 0.2
    RECURRING_BLOCKER_MIN_COUNT = 3
    RECURRING_BLOCKER_WINDOW_DAYS = 14
    PRIORITY_INFLATION_PCT = 0.7
    PEAK_WINDOW_MIN_ENTRIES = 5
    PEAK_WINDOW_PROD_THRESHOLD = 7.0
    PEAK_WINDOW_FOCUS_THRESHOLD = 7.0

    def __init__(self, kb, analyzer):
        """
        Args:
            kb: KnowledgeBaseService instance
            analyzer: SmartTimeContextAnalyzer instance
        """
        self.kb = kb
        self.analyzer = analyzer

    # ─── Trigger entry point ─────────────────────────────────────────────

    async def run_post_sync(
        self,
        user_id: str,
        new_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Top-level orchestration. Errors logged, never raised — sync flow
        must not fail on intelligence layer issues."""
        new_entries = new_entries or []
        try:
            user_priorities = await self._load_user_priorities(user_id)

            # 1. Classify new time entries → time_entry_v2 vectors
            await self._classify_and_persist_time_entries(user_id, new_entries, user_priorities)

            # 2. Rolling window analysis → PATTERN entries
            await self._persist_window_patterns(user_id, user_priorities)

            # 3. Anomaly detection → INSIGHT / PATTERN entries
            await self._detect_ghost_goals(user_id)
            await self._detect_zombie_habits(user_id)
            await self._detect_behavioral_drift(user_id)
            await self._detect_recurring_blockers(user_id)
            await self._detect_priority_inflation(user_id)
            await self._detect_deep_work_peak_window(user_id)

            # 4. Re-embed goals touched by new time entries
            touched_goals = {
                e.get("linked_goal")
                for e in new_entries
                if e.get("category") == "time_entry" and e.get("linked_goal")
            }
            for goal_id in touched_goals:
                await self._refresh_goal_embedding(user_id, goal_id)

            # 5. Close commitment chain for today
            today = datetime.now(timezone.utc).date()
            await self._close_commitment_chain(user_id, today.isoformat())

        except Exception as exc:  # noqa: BLE001
            logger.exception("IntelligencePrecomputeService.run_post_sync failed: %s", exc)

    # ─── Step 1: classify and persist time entries ───────────────────────

    async def _classify_and_persist_time_entries(
        self,
        user_id: str,
        new_entries: List[Dict[str, Any]],
        user_priorities: List[str],
    ) -> None:
        for entry in new_entries:
            if entry.get("category") != "time_entry":
                continue
            try:
                classification = self.analyzer.classify_entry(entry, user_priorities)
                classification_dict = _safe_dataclass_to_dict(classification)
                content = self.kb._build_time_entry_embedding(entry, classification_dict)
                metadata = self._build_time_entry_metadata(entry, classification_dict)
                sync_event_key = f"time_entry_v2:{entry.get('entry_id') or entry.get('alterego_entry_id')}"
                await self.kb.create_entry(
                    entry_type=KnowledgeEntryType.INTERACTION,
                    entry_sub_type=KnowledgeEntrySubType.WORK_INTERACTION,
                    category="time_entry_v2",
                    title=content[:140],
                    content=content,
                    metadata={**metadata, "context": {"sync_event_key": sync_event_key}},
                    tags=entry.get("tags") or [],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("classify_and_persist failed for entry %s: %s", entry.get("entry_id"), exc)

    def _build_time_entry_metadata(
        self,
        entry: Dict[str, Any],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_time = entry.get("start_time")
        weekday, hour = None, None
        dt = self._parse_dt_aware(start_time) if start_time else None
        if dt is not None:
            weekday = dt.strftime("%A").lower()
            hour = dt.hour
        return {
            "entry_id": entry.get("entry_id"),
            "alterego_entry_id": entry.get("alterego_entry_id") or entry.get("entry_id"),
            "start_time": entry.get("start_time"),
            "end_time": entry.get("end_time"),
            "duration_minutes": entry.get("duration_minutes"),
            "focus_score": entry.get("focus_score"),
            "energy_score": entry.get("energy_score"),
            "linked_goal": entry.get("linked_goal"),
            "project_id": entry.get("project_id"),
            "tags": entry.get("tags") or [],
            "weekday": weekday,
            "hour_of_day": hour,
            "is_first_entry_of_day": entry.get("is_first_entry_of_day", False),
            "is_last_entry_of_day": entry.get("is_last_entry_of_day", False),
            "work_type": classification.get("work_type"),
            "energy_pattern": classification.get("energy_pattern"),
            "focus_quality": classification.get("focus_quality"),
            "productivity_score": classification.get("productivity_score"),
            "goal_alignment": classification.get("goal_alignment"),
        }

    # ─── Step 2: rolling window patterns ─────────────────────────────────

    async def _persist_window_patterns(self, user_id: str, user_priorities: List[str]) -> None:
        """Roll up 7d and 30d analyses, persist productivity profile + pattern insights."""
        try:
            recent_7d = await self._fetch_recent_time_entries(user_id, days=7)
            if recent_7d:
                window = self.analyzer.analyze_time_window(recent_7d, "7d", user_priorities)
                for idx, insight_text in enumerate(window.pattern_insights or []):
                    await self._upsert_pattern(
                        user_id=user_id,
                        pattern_type="7d_window",
                        det_key_parts=("7d_window", self._iso_week(), str(idx)),
                        text=insight_text,
                        metadata={"window": "7d", "index": idx},
                    )

            recent_30d = await self._fetch_recent_time_entries(user_id, days=30)
            if recent_30d:
                profile = self.analyzer.generate_productivity_profile(recent_30d, user_priorities)
                profile_text = self._serialize_productivity_profile(profile)
                if profile_text:
                    await self._upsert_insight(
                        user_id=user_id,
                        insight_type="productivity_profile",
                        det_key_parts=("productivity_profile", self._iso_week()),
                        text=profile_text,
                        metadata={
                            "deep_work_capacity": profile.deep_work_capacity,
                            "context_switch_frequency": profile.context_switch_frequency,
                            "focus_consistency_score": profile.focus_consistency_score,
                            "peak_performance_windows": [
                                f"{s}-{e}" for s, e in profile.peak_performance_windows or []
                            ],
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("persist_window_patterns failed: %s", exc)

    @staticmethod
    def _serialize_productivity_profile(profile) -> str:
        windows = profile.peak_performance_windows or []
        windows_str = ", ".join(f"{s}-{e}h" for s, e in windows) if windows else "none yet"
        adjustments = "; ".join(
            adj.get("recommendation", "") for adj in (profile.recommended_adjustments or [])
        ) or "no adjustments suggested"
        return (
            f"Deep work capacity: {profile.deep_work_capacity:.0f} min avg sustained. "
            f"Peak performance windows: {windows_str}. "
            f"Context switching: {profile.context_switch_frequency:.1f}/hr. "
            f"Focus consistency: {profile.focus_consistency_score:.0f}/100. "
            f"Learning ratio: {profile.learning_investment_ratio*100:.0f}%, "
            f"shallow ratio: {profile.shallow_work_ratio*100:.0f}%. "
            f"Adjustments: {adjustments}."
        )

    # ─── Step 3: anomaly detection ──────────────────────────────────────

    async def _detect_ghost_goals(self, user_id: str) -> None:
        goals = await self._fetch_goals(user_id)
        for goal in goals:
            age = self._goal_age_days(goal)
            invested = float(goal.get("invested_hours") or 0)
            if age > self.GHOST_GOAL_AGE_DAYS and invested < self.GHOST_GOAL_INVESTED_HOURS:
                text = (
                    f"Goal '{goal.get('title','')}' may be abandoned — {age} days old, "
                    f"only {invested}h invested."
                )
                await self._upsert_insight(
                    user_id=user_id,
                    insight_type="ghost_goal",
                    det_key_parts=("ghost_goal", str(goal.get("goal_id") or goal.get("id"))),
                    text=text,
                    metadata={
                        "insight_type": "ghost_goal",
                        "goal_id": goal.get("goal_id") or goal.get("id"),
                        "severity": "medium",
                    },
                )

    async def _detect_zombie_habits(self, user_id: str) -> None:
        habits = await self._fetch_habit_entries(user_id)
        iso_week = self._iso_week()
        for habit in habits:
            comp_30d = float(habit.get("completion_30d") or habit.get("completionRate30d") or 0)
            if comp_30d and comp_30d < self.ZOMBIE_HABIT_THRESHOLD:
                name = habit.get("habit_name") or habit.get("name") or "habit"
                text = f"Habit '{name}' completion rate critical: {comp_30d:.0f}% over last 30 days."
                await self._upsert_insight(
                    user_id=user_id,
                    insight_type="zombie_habit",
                    det_key_parts=("zombie_habit", str(habit.get("habit_id") or name), iso_week),
                    text=text,
                    metadata={
                        "insight_type": "zombie_habit",
                        "habit_id": habit.get("habit_id"),
                        "habit_name": name,
                        "severity": "high",
                    },
                )

    async def _detect_behavioral_drift(self, user_id: str) -> None:
        recent_7d = await self._fetch_recent_time_entries(user_id, days=7)
        prior_7d = await self._fetch_recent_time_entries(user_id, days=14, exclude_recent_days=7)
        if not recent_7d or not prior_7d:
            return
        cur = self._avg_productivity(recent_7d)
        pri = self._avg_productivity(prior_7d)
        if cur is None or pri is None:
            return
        delta = cur - pri
        if delta < -self.DRIFT_PRODUCTIVITY_DELTA:
            text = (
                f"Productivity trend declining: {cur:.1f} this week vs {pri:.1f} last week "
                f"(Δ {delta:+.1f})."
            )
            await self._upsert_insight(
                user_id=user_id,
                insight_type="behavioral_drift",
                det_key_parts=("drift", self._iso_week()),
                text=text,
                metadata={
                    "insight_type": "behavioral_drift",
                    "direction": "down",
                    "delta": round(delta, 2),
                    "current_avg": round(cur, 2),
                    "prior_avg": round(pri, 2),
                },
            )

    async def _detect_recurring_blockers(self, user_id: str) -> None:
        entries = await self._fetch_recent_time_entries(
            user_id, days=self.RECURRING_BLOCKER_WINDOW_DAYS
        )
        counter: Counter = Counter()
        for e in entries:
            blocker = (e.get("blockers") or "").strip()
            if blocker:
                counter[blocker.lower()] += 1
        iso_week = self._iso_week()
        for text_key, count in counter.items():
            if count >= self.RECURRING_BLOCKER_MIN_COUNT:
                blocker_hash = hashlib.sha256(text_key.encode("utf-8")).hexdigest()[:12]
                pattern_text = (
                    f"Recurring blocker: '{text_key}' appeared {count}x in last "
                    f"{self.RECURRING_BLOCKER_WINDOW_DAYS} days."
                )
                await self._upsert_pattern(
                    user_id=user_id,
                    pattern_type="recurring_blocker",
                    det_key_parts=("recurring_blocker", blocker_hash, iso_week),
                    text=pattern_text,
                    metadata={
                        "pattern_type": "recurring_blocker",
                        "blocker_text": text_key,
                        "frequency": count,
                    },
                )

    async def _detect_priority_inflation(self, user_id: str) -> None:
        tasks = await self._fetch_recent_tasks(user_id, days=7)
        if not tasks:
            return
        total = len(tasks)
        high = sum(1 for t in tasks if str(t.get("priority", "")).lower() in {"high", "urgent"})
        if total >= 5 and (high / total) > self.PRIORITY_INFLATION_PCT:
            pct = round(high / total * 100, 1)
            text = (
                f"{high} of {total} tasks ({pct}%) created in last 7 days marked high priority — "
                "priority signal may be diluted."
            )
            await self._upsert_insight(
                user_id=user_id,
                insight_type="priority_inflation",
                det_key_parts=("priority_inflation", self._iso_week()),
                text=text,
                metadata={
                    "insight_type": "priority_inflation",
                    "high_count": high,
                    "total": total,
                    "pct": pct,
                },
            )

    async def _detect_deep_work_peak_window(self, user_id: str) -> None:
        entries = await self._fetch_recent_time_entries(user_id, days=30)
        # Group by hour, keep entries that meet deep+productive bar
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        weekday_buckets: Dict[int, Counter] = defaultdict(Counter)
        for e in entries:
            hour = e.get("hour_of_day")
            prod = float(e.get("productivity_score") or 0)
            focus = float(e.get("focus_quality") or 0)
            if hour is None or prod < self.PEAK_WINDOW_PROD_THRESHOLD or focus < self.PEAK_WINDOW_FOCUS_THRESHOLD:
                continue
            buckets[int(hour)].append(e)
            wd = e.get("weekday")
            if wd:
                weekday_buckets[int(hour)][wd] += 1

        peak_hours = sorted([h for h, lst in buckets.items() if len(lst) >= self.PEAK_WINDOW_MIN_ENTRIES])
        if not peak_hours:
            return

        # Collapse contiguous hours into windows
        windows: List[List[int]] = []
        current = [peak_hours[0]]
        for h in peak_hours[1:]:
            if h == current[-1] + 1:
                current.append(h)
            else:
                windows.append(current)
                current = [h]
        windows.append(current)

        for win in windows:
            start, end = win[0], win[-1] + 1
            weekdays = sorted({wd for h in win for wd in weekday_buckets[h]})
            text = (
                f"Deep work consistently achieved {start}-{end}h"
                + (f" on {', '.join(weekdays)}." if weekdays else ".")
            )
            await self._upsert_pattern(
                user_id=user_id,
                pattern_type="peak_performance_window",
                det_key_parts=("peak_window", self._iso_week(), f"{start}-{end}"),
                text=text,
                metadata={
                    "pattern_type": "peak_performance_window",
                    "start_hour": start,
                    "end_hour": end,
                    "weekdays": weekdays,
                },
            )

    # ─── Step 4: refresh goal embeddings ─────────────────────────────────

    async def _refresh_goal_embedding(self, user_id: str, goal_id: str) -> None:
        try:
            goal = await self._fetch_goal_by_id(user_id, goal_id)
            if not goal:
                return
            goal["invested_hours"] = await self._aggregate_goal_invested_hours(user_id, goal_id)
            goal["hours_this_month"] = await self._aggregate_goal_invested_hours(
                user_id, goal_id, days=self._days_this_month()
            )
            goal["status"] = self._compute_goal_status(goal)
            goal["days_remaining"] = self._goal_days_remaining(goal)

            content = self.kb._build_goal_embedding(goal)
            sync_event_key = f"goal_v2:{goal_id}"
            await self.kb.create_entry(
                entry_type=KnowledgeEntryType.INSIGHT,
                entry_sub_type=KnowledgeEntrySubType.GOAL,
                category="goal_v2",
                title=goal.get("title", "")[:140],
                content=content,
                metadata={
                    "goal_id": goal_id,
                    "status": goal["status"],
                    "invested_hours": goal["invested_hours"],
                    "hours_this_month": goal["hours_this_month"],
                    "days_remaining": goal["days_remaining"],
                    "context": {"sync_event_key": sync_event_key},
                },
                tags=["goal"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("refresh_goal_embedding failed for %s: %s", goal_id, exc)

    def _compute_goal_status(self, goal: Dict[str, Any]) -> str:
        age = self._goal_age_days(goal)
        invested = float(goal.get("invested_hours") or 0)
        completion_pct = float(goal.get("completion_percent") or 0)
        days_remaining = self._goal_days_remaining(goal) or 0

        if age > self.GHOST_GOAL_AGE_DAYS and invested < self.GHOST_GOAL_INVESTED_HOURS:
            return "ghost"
        if days_remaining and days_remaining < 14 and completion_pct < 50:
            return "at_risk"
        # on_track: completion_pct >= 80% of elapsed_pct
        total_duration = goal.get("total_duration_days") or (age + max(days_remaining, 0))
        if total_duration:
            elapsed_pct = (age / total_duration) * 100 if total_duration else 0
            if elapsed_pct and (completion_pct / max(elapsed_pct, 1)) >= 0.8:
                return "on_track"
        return "in_progress"

    # ─── Step 5: close commitment chain ─────────────────────────────────

    async def _close_commitment_chain(self, user_id: str, date_iso: str) -> None:
        morning = await self._fetch_morning_intent(user_id, date_iso)
        if not morning:
            return
        focus_target = (morning.get("focus_target") or "").lower()
        if not focus_target:
            return
        day_entries = await self._fetch_time_entries_for_date(user_id, date_iso)
        evening = await self._fetch_evening_reflection(user_id, date_iso)

        related_minutes = 0.0
        related_count = 0
        for e in day_entries:
            text = f"{e.get('description','')} {e.get('project_name','')}".lower()
            if focus_target and focus_target in text:
                related_minutes += float(e.get("duration_minutes") or 0)
                related_count += 1

        if related_count == 0:
            verdict = "no"
        elif related_minutes >= 60:
            verdict = "yes"
        else:
            verdict = "partial"

        text = (
            f"On {date_iso}, committed to '{morning.get('focus_target','')}': "
            f"logged {related_minutes/60:.1f}h across {related_count} entries. "
            f"Follow-through: {verdict}."
        )
        morning_id = morning.get("entry_id")
        evening_id = evening.get("entry_id") if evening else None
        await self._upsert_pattern(
            user_id=user_id,
            pattern_type="commitment_chain",
            det_key_parts=("commitment_chain", date_iso),
            text=text,
            metadata={
                "pattern_type": "commitment_chain",
                "date": date_iso,
                "morning_entry_id": morning_id,
                "evening_entry_id": evening_id,
                "follow_through": verdict,
                "related_minutes": related_minutes,
                "related_count": related_count,
            },
        )

    # ─── Helpers: persistence wrappers ──────────────────────────────────

    async def _upsert_insight(
        self,
        *,
        user_id: str,
        insight_type: str,
        det_key_parts: Iterable[str],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        sync_event_key = "insight_v2:" + ":".join(str(p) for p in det_key_parts)
        meta = {
            **metadata,
            "generated_by": "intelligence_precompute",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "context": {"sync_event_key": sync_event_key},
        }
        try:
            await self.kb.create_entry(
                entry_type=KnowledgeEntryType.INSIGHT,
                entry_sub_type=KnowledgeEntrySubType.IMPORTANT_INSIGHT,
                category="insight_v2",
                title=text[:140],
                content=text,
                metadata=meta,
                tags=["intelligence", insight_type],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("upsert_insight failed: %s", exc)

    async def _upsert_pattern(
        self,
        *,
        user_id: str,
        pattern_type: str,
        det_key_parts: Iterable[str],
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        sync_event_key = "pattern_v2:" + ":".join(str(p) for p in det_key_parts)
        meta = {
            **metadata,
            "generated_by": "intelligence_precompute",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "context": {"sync_event_key": sync_event_key},
        }
        try:
            await self.kb.create_entry(
                entry_type=KnowledgeEntryType.PATTERN,
                entry_sub_type=KnowledgeEntrySubType.CONSCIOUS_PATTERNS,
                category="pattern_v2",
                title=text[:140],
                content=text,
                metadata=meta,
                tags=["intelligence", pattern_type],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("upsert_pattern failed: %s", exc)

    # ─── Helpers: data fetching (lean wrappers around KB) ────────────────

    async def _load_user_priorities(self, user_id: str) -> List[str]:
        try:
            entries = await self.kb.get_all_entries(category="user_preference")
            return [getattr(e, "content", "") for e in entries or [] if getattr(e, "content", "")]
        except Exception:
            return []

    async def _fetch_entries_by_category(self, category: str) -> List[Any]:
        try:
            return await self.kb.get_all_entries(category=category)
        except Exception:
            return []

    async def _fetch_recent_time_entries(
        self,
        user_id: str,
        days: int,
        exclude_recent_days: int = 0,
    ) -> List[Dict[str, Any]]:
        entries = await self._fetch_entries_by_category("time_entry_v2")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        exclude_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=exclude_recent_days)
            if exclude_recent_days
            else None
        )
        results: List[Dict[str, Any]] = []
        for e in entries or []:
            meta = self._entry_meta(e)
            dt = self._parse_dt_aware(meta.get("start_time"))
            if dt and dt >= cutoff:
                if exclude_cutoff and dt >= exclude_cutoff:
                    continue
                results.append({**meta, "content": self._entry_content(e)})
        return results

    async def _fetch_time_entries_for_date(self, user_id: str, date_iso: str) -> List[Dict[str, Any]]:
        all_recent = await self._fetch_recent_time_entries(user_id, days=2)
        out: List[Dict[str, Any]] = []
        for e in all_recent:
            start = str(e.get("start_time") or "")
            if start.startswith(date_iso):
                out.append(e)
        return out

    async def _fetch_goals(self, user_id: str) -> List[Dict[str, Any]]:
        entries = await self._fetch_entries_by_category("goal_v2")
        return [{**self._entry_meta(e), "content": self._entry_content(e)} for e in entries or []]

    async def _fetch_goal_by_id(self, user_id: str, goal_id: str) -> Optional[Dict[str, Any]]:
        goals = await self._fetch_goals(user_id)
        for g in goals:
            if str(g.get("goal_id")) == str(goal_id):
                return g
        return None

    async def _aggregate_goal_invested_hours(
        self,
        user_id: str,
        goal_id: str,
        days: Optional[int] = None,
    ) -> float:
        entries = await self._fetch_recent_time_entries(user_id, days=days or 365)
        total_min = 0.0
        for e in entries:
            if str(e.get("linked_goal")) == str(goal_id):
                total_min += float(e.get("duration_minutes") or 0)
        return round(total_min / 60.0, 2)

    async def _fetch_habit_entries(self, user_id: str) -> List[Dict[str, Any]]:
        entries = await self._fetch_entries_by_category("habit_entry")
        return [{**self._entry_meta(e), "content": self._entry_content(e)} for e in entries or []]

    async def _fetch_recent_tasks(self, user_id: str, days: int) -> List[Dict[str, Any]]:
        entries = await self._fetch_entries_by_category("task_entry")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out: List[Dict[str, Any]] = []
        for e in entries or []:
            meta = self._entry_meta(e)
            created = meta.get("created_at") or meta.get("start_time") or getattr(e, "created_at", None)
            dt = self._parse_dt_aware(created)
            if dt and dt >= cutoff:
                out.append(meta)
        return out

    async def _fetch_morning_intent(self, user_id: str, date_iso: str) -> Optional[Dict[str, Any]]:
        entries = await self._fetch_entries_by_category("morning_intent")
        for e in entries or []:
            meta = self._entry_meta(e)
            content = self._entry_content(e)
            if date_iso in content or str(meta.get("checkup_date") or "") == date_iso:
                return {**meta, "content": content, "entry_id": getattr(e, "entry_id", None)}
        return None

    async def _fetch_evening_reflection(self, user_id: str, date_iso: str) -> Optional[Dict[str, Any]]:
        entries = await self._fetch_entries_by_category("evening_reflection")
        for e in entries or []:
            meta = self._entry_meta(e)
            content = self._entry_content(e)
            if date_iso in content or str(meta.get("checkup_date") or "") == date_iso:
                return {**meta, "content": content, "entry_id": getattr(e, "entry_id", None)}
        return None

    # ─── Misc helpers ───────────────────────────────────────────────────

    @staticmethod
    def _entry_meta(entry: Any) -> Dict[str, Any]:
        if isinstance(entry, dict):
            return entry.get("metadata") or {}
        meta = getattr(entry, "metadata", None) or {}
        return meta if isinstance(meta, dict) else {}

    @staticmethod
    def _entry_content(entry: Any) -> str:
        if isinstance(entry, dict):
            return str(entry.get("content") or "")
        return str(getattr(entry, "content", "") or "")

    @staticmethod
    def _parse_dt_aware(value: Any) -> Optional[datetime]:
        """Parse an ISO datetime string into a tz-aware UTC datetime.

        AlterEgo sends timestamps both as 'Z'-suffixed UTC ('2026-05-07T19:35:36Z')
        and as naive local strings ('2026-05-07T19:35:36'). datetime.fromisoformat
        produces a naive datetime for the latter, which crashed comparisons against
        datetime.now(timezone.utc) with "can't compare offset-naive and offset-aware".
        Naive inputs are assumed to already be UTC and tagged as such so all
        downstream arithmetic stays consistent.
        """
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            text = str(value).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _avg_productivity(entries: List[Dict[str, Any]]) -> Optional[float]:
        scores = [float(e.get("productivity_score") or 0) for e in entries if e.get("productivity_score") is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @classmethod
    def _goal_age_days(cls, goal: Dict[str, Any]) -> int:
        dt = cls._parse_dt_aware(goal.get("created_at") or goal.get("start_date"))
        if dt is None:
            return 0
        return (datetime.now(timezone.utc) - dt).days

    @classmethod
    def _goal_days_remaining(cls, goal: Dict[str, Any]) -> Optional[int]:
        dt = cls._parse_dt_aware(
            goal.get("endDate") or goal.get("end_date") or goal.get("target_date")
        )
        if dt is None:
            return None
        return max((dt - datetime.now(timezone.utc)).days, 0)

    @staticmethod
    def _iso_week() -> str:
        now = datetime.now(timezone.utc)
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"

    @staticmethod
    def _days_this_month() -> int:
        return datetime.now(timezone.utc).day
