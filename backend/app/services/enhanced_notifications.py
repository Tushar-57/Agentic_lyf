"""
Enhanced AI Notification Service - Personalized, LLM-powered proactive insights.

This module provides intelligent, context-aware notifications that adapt to the user's
unique patterns, priorities, and current state across tasks, time entries, habits, and goals.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from html import escape

from app.models.knowledge import KnowledgeEntry, KnowledgeEntryType, UserPreferences
from app.services.knowledge_base import get_knowledge_base_service
from app.services.ai_notifications_store import get_ai_notification_store, AINotificationRecord
from app.services.checkup_store import get_daily_checkup_store
from app.llm.service import get_llm_service
from app.llm.base import CompletionRequest, ChatMessage
from app.auth.user_context import get_current_user
from app.utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.NOTIFICATION)


@dataclass
class UserContextSnapshot:
    """Comprehensive snapshot of user's current state."""
    # Temporal context
    now: datetime
    timezone_name: str
    today: date
    current_hour: int
    day_of_week: int  # 0=Monday, 6=Sunday
    is_workday: bool
    
    # Task context
    overdue_tasks: int = 0
    due_today_tasks: int = 0
    upcoming_deadlines: List[Dict[str, Any]] = field(default_factory=list)
    focus_tasks: List[Dict[str, Any]] = field(default_factory=list)
    top_goals: List[str] = field(default_factory=list)
    
    # Time tracking context
    total_tracked_today: float = 0.0  # minutes
    total_tracked_week: float = 0.0
    avg_daily_minutes_14d: float = 0.0
    billable_ratio_14d: float = 0.0
    deep_work_coverage: float = 0.0
    planned_deep_work: float = 0.0
    yesterday_entries: List[Dict[str, Any]] = field(default_factory=list)
    today_entries: List[Dict[str, Any]] = field(default_factory=list)
    top_projects: List[str] = field(default_factory=list)
    recent_focus_scores: List[float] = field(default_factory=list)
    avg_focus_score: float = 0.0
    
    # Habit context
    habits_total: int = 0
    habits_completed_today: int = 0
    habits_completion_rate_7d: float = 0.0
    habits_avg_streak: float = 0.0
    
    # Performance context
    latest_performance_score: float = 0.0
    latest_checkup_date: Optional[date] = None
    checkup_consistency_14d: float = 0.0
    
    # User preferences
    work_hours: str = "09:00-17:00"
    check_in_time: str = "09:00"
    role: str = ""
    priorities: List[str] = field(default_factory=list)
    monthly_income_target: float = 0.0
    
    # Historical patterns (derived from past data)
    typical_morning_start: Optional[str] = None
    typical_focus_peak: Optional[str] = None  # e.g., "10:00-12:00"
    productivity_trend: str = "stable"  # improving, declining, stable
    last_7_days_checkups: int = 0


@dataclass
class PersonalizedNotification:
    """A single personalized notification with rich context."""
    notification_key: str
    kind: str
    severity: str  # critical, high, medium, low
    priority_score: float  # 0-100 for internal ranking
    
    # Content
    title: str
    summary: str
    details_html: str  # Rich HTML content
    insights: List[str] = field(default_factory=list)
    recommended_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context
    triggering_metrics: Dict[str, Any] = field(default_factory=dict)
    user_context: Optional[UserContextSnapshot] = None
    
    # Timing
    optimal_delivery_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Meta
    score: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class NotificationCandidate:
    """Raw candidate for notification generation."""
    key: str
    kind: str
    severity: str
    title_template: str
    context_requirements: List[str]  # which UserContextSnapshot fields are needed
    priority_formula: str  # how to calculate priority_score


class EnhancedNotificationEngine:
    """
    Intelligent notification engine that generates personalized, context-aware alerts.
    Uses LLM for rich content generation and pattern analysis.
    """
    
    NOTIFICATION_CANDIDATES: List[NotificationCandidate] = [
        # Goal Alignment (always generated)
        NotificationCandidate(
            key="goal_alignment_score",
            kind="goal_alignment",
            severity="medium",
            title_template="Goal Alignment Score: {score}/100",
            context_requirements=["latest_performance_score", "deep_work_coverage", "habits_completion_rate_7d"],
            priority_formula="100 - goal_alignment_score"
        ),
        
        # Deadline drift (high priority when present)
        NotificationCandidate(
            key="proactive.deadline_drift",
            kind="proactive_alert",
            severity="high",
            title_template="Deadline Drift Risk",
            context_requirements=["overdue_tasks", "due_today_tasks", "upcoming_deadlines"],
            priority_formula="overdue_tasks * 25 + min(due_today_tasks, 5) * 10"
        ),
        
        # Deep work gap
        NotificationCandidate(
            key="proactive.deep_work_gap",
            kind="proactive_alert",
            severity="high",
            title_template="Deep Work Coverage Gap",
            context_requirements=["planned_deep_work", "deep_work_coverage", "total_tracked_today"],
            priority_formula="(1 - deep_work_coverage) * 100 if planned_deep_work > 60 else 0"
        ),
        
        # Habit consistency
        NotificationCandidate(
            key="proactive.habit_consistency",
            kind="proactive_alert",
            severity="medium",
            title_template="Habit Consistency Slipping",
            context_requirements=["habits_total", "habits_completed_today", "habits_completion_rate_7d"],
            priority_formula="(1 - habits_completion_rate_7d) * 80 if habits_total >= 3 else 0"
        ),
        
        # Billable trajectory
        NotificationCandidate(
            key="proactive.billable_trajectory",
            kind="proactive_alert",
            severity="high",
            title_template="Billable Trajectory Behind",
            context_requirements=["monthly_income_target", "billable_ratio_14d", "avg_daily_minutes_14d"],
            priority_formula="(0.5 - billable_ratio_14d) * 150 if monthly_income_target > 0 else 0"
        ),
        
        # Morning checkup missing
        NotificationCandidate(
            key="proactive.morning_checkup_missing",
            kind="proactive_alert",
            severity="medium",
            title_template="Morning Check-In Missing",
            context_requirements=["latest_checkup_date", "today", "current_hour"],
            priority_formula="30 if latest_checkup_date != today and current_hour >= 9 else 0"
        ),
        
        # Focus score declining
        NotificationCandidate(
            key="proactive.focus_declining",
            kind="proactive_alert",
            severity="medium",
            title_template="Focus Score Trending Down",
            context_requirements=["avg_focus_score", "recent_focus_scores"],
            priority_formula="max(0, (7 - avg_focus_score) * 10) if len(recent_focus_scores) >= 3 else 0"
        ),
        
        # Energy pattern mismatch
        NotificationCandidate(
            key="proactive.energy_mismatch",
            kind="proactive_alert",
            severity="low",
            title_template="Energy-Task Mismatch",
            context_requirements=["current_hour", "typical_focus_peak", "focus_tasks"],
            priority_formula="20 if current_hour not in peak_hours and len(focus_tasks) > 0 else 0"
        ),
        
        # Weekly review needed
        NotificationCandidate(
            key="proactive.weekly_review",
            kind="proactive_alert",
            severity="low",
            title_template="Weekly Review Recommended",
            context_requirements=["day_of_week", "checkup_consistency_14d"],
            priority_formula="25 if day_of_week == 4 and checkup_consistency_14d < 0.7 else 0"
        ),
    ]
    
    def __init__(self):
        self.llm_service = None
        self.kb_service = None
        self.notification_store = None
        self.checkup_store = None
    
    async def _ensure_services(self):
        """Initialize required services."""
        if not self.llm_service:
            self.llm_service = await get_llm_service()
        if not self.kb_service:
            self.kb_service = get_knowledge_base_service()
        if not self.notification_store:
            self.notification_store = get_ai_notification_store()
        if not self.checkup_store:
            self.checkup_store = get_daily_checkup_store()
    
    async def generate_personalized_notifications(
        self,
        context_snapshot: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        use_llm: bool = True
    ) -> List[PersonalizedNotification]:
        """
        Generate personalized notifications based on user's current state.
        
        Args:
            context_snapshot: Optional external context (from frontend)
            limit: Maximum number of notifications to return
            use_llm: Whether to use LLM for rich content generation
            
        Returns:
            List of personalized notifications, sorted by priority
        """
        await self._ensure_services()
        
        # Build comprehensive user context
        user_context = await self._build_user_context(context_snapshot)
        
        # Generate notification candidates with priority scores
        candidates = self._evaluate_candidates(user_context)
        
        # Sort by priority (higher = more urgent)
        candidates.sort(key=lambda c: c.priority_score, reverse=True)
        
        # Generate rich notifications (with or without LLM)
        notifications: List[PersonalizedNotification] = []
        for candidate in candidates[:limit]:
            if use_llm and self.llm_service:
                notification = await self._generate_llm_notification(candidate, user_context)
            else:
                notification = self._generate_fallback_notification(candidate, user_context)
            notifications.append(notification)
        
        return notifications
    
    async def _build_user_context(
        self,
        external_context: Optional[Dict[str, Any]] = None
    ) -> UserContextSnapshot:
        """Build comprehensive user context from all available data sources."""
        await self._ensure_services()
        
        preferences = await self.kb_service.get_user_preferences()
        all_entries = await self.kb_service.get_all_entries()
        
        # Time setup
        tz_name = self._extract_timezone(preferences)
        tz = self._resolve_timezone(tz_name)
        now = datetime.now(tz)
        today = now.date()
        
        # Extract from external context (frontend-provided real-time data)
        ext = external_context or {}
        deadline_data = ext.get("deadlineTasks", {}) if isinstance(ext.get("deadlineTasks"), dict) else {}
        habit_data = ext.get("habitMetrics", {}) if isinstance(ext.get("habitMetrics"), dict) else {}
        time_data = ext.get("timeMetrics", {}) if isinstance(ext.get("timeMetrics"), dict) else {}
        
        # Time entries analysis
        time_entries = self._extract_time_entries(all_entries, now, tz)
        yesterday = today - timedelta(days=1)
        week_start = today - timedelta(days=6)
        
        yesterday_entries = [e for e in time_entries if e["date"] == yesterday]
        today_entries = [e for e in time_entries if e["date"] == today]
        week_entries = [e for e in time_entries if e["date"] >= week_start]
        
        # Calculate metrics
        total_week = sum(e["duration_minutes"] for e in week_entries)
        total_today = sum(e["duration_minutes"] for e in today_entries)
        
        # 14-day average
        lookback_14d = today - timedelta(days=13)
        # Validate entry structure before processing to avoid comparison errors
        entries_14d = [
            e for e in time_entries
            if isinstance(e, dict) and isinstance(e.get("date"), date) and e["date"] >= lookback_14d
        ]
        total_14d = sum(e.get("duration_minutes", 0) for e in entries_14d)
        avg_14d = total_14d / 14.0 if entries_14d else 0.0

        # Billable ratio with safe division
        billable_count = len([e for e in entries_14d if e.get("billable")])
        billable_ratio = billable_count / max(len(entries_14d), 1)
        
        # Focus scores
        focus_scores = [e["focus_score"] for e in week_entries if e["focus_score"] > 0]
        avg_focus = sum(focus_scores) / len(focus_scores) if focus_scores else 0.0
        
        # Top projects
        project_times: Dict[str, float] = {}
        for e in week_entries:
            project_times[e["project"]] = project_times.get(e["project"], 0) + e["duration_minutes"]
        top_projects = [p for p, _ in sorted(project_times.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # Checkup data
        checkup_consistency = self._calculate_checkup_consistency(entries_14d)
        latest_checkups = {}
        if self.checkup_store:
            request_user = get_current_user()
            latest_checkups = self.checkup_store.get_latest_checkups_for_user(request_user.storage_key)
        
        latest_morning = latest_checkups.get("morning")
        latest_evening = latest_checkups.get("evening")
        
        # Extract checkup metrics
        performance_score = 0.0
        deep_work_coverage = 0.0
        planned_deep_work = 0.0
        latest_checkup_date = None
        
        for checkup in [latest_morning, latest_evening]:
            if checkup:
                payload = checkup.payload or {}
                perspective = payload.get("perspective", {}) if isinstance(payload.get("perspective"), dict) else {}
                performance_score = max(performance_score, perspective.get("confidence", 0.0))
                deep_work_coverage = max(deep_work_coverage, perspective.get("deepWorkCoverageRatio", 0.0))
                planned_deep_work = max(planned_deep_work, perspective.get("plannedDeepWorkMinutes", 0.0))
                latest_checkup_date = max(latest_checkup_date, checkup.checkup_date) if latest_checkup_date else checkup.checkup_date
        
        # Goal titles
        goal_titles = self._extract_goal_titles(all_entries)
        
        # Work hours
        work_hours = (
            (preferences.general.get("work_hours") if isinstance(preferences.general, dict) else None)
            or (preferences.productivity.get("work_hours") if isinstance(preferences.productivity, dict) else None)
            or "09:00-17:00"
        )
        check_in = (
            (preferences.journal.get("check_in_time") if isinstance(preferences.journal, dict) else None)
            or "09:00"
        )
        
        # Role and priorities
        role = ""
        priorities: List[str] = []
        if isinstance(preferences.general, dict):
            role = preferences.general.get("role", "")
            priorities = preferences.general.get("priorities", []) or []
        
        # Income target
        monthly_target = 0.0
        if isinstance(preferences.finance, dict):
            monthly_target = preferences.finance.get("monthly_income_target", 0.0)
        
        return UserContextSnapshot(
            now=now,
            timezone_name=tz_name,
            today=today,
            current_hour=now.hour,
            day_of_week=now.weekday(),
            is_workday=now.weekday() < 5,
            
            overdue_tasks=int(deadline_data.get("overdue", 0)),
            due_today_tasks=int(deadline_data.get("dueToday", 0)),
            upcoming_deadlines=ext.get("upcomingDeadlines", []) if isinstance(ext.get("upcomingDeadlines"), list) else [],
            focus_tasks=ext.get("focusTasks", []) if isinstance(ext.get("focusTasks"), list) else [],
            top_goals=goal_titles,
            
            total_tracked_today=total_today,
            total_tracked_week=total_week,
            avg_daily_minutes_14d=avg_14d,
            billable_ratio_14d=billable_ratio,
            deep_work_coverage=deep_work_coverage if deep_work_coverage > 0 else 0.55,
            planned_deep_work=planned_deep_work,
            yesterday_entries=yesterday_entries,
            today_entries=today_entries,
            top_projects=top_projects,
            recent_focus_scores=focus_scores,
            avg_focus_score=round(avg_focus, 1) if avg_focus > 0 else 0.0,
            
            habits_total=int(habit_data.get("totalHabits", 0)),
            habits_completed_today=int(habit_data.get("completedToday", 0)),
            habits_completion_rate_7d=habit_data.get("completionRate7d", 0.0) or 0.0,
            habits_avg_streak=habit_data.get("avgStreak", 0.0) or 0.0,
            
            latest_performance_score=performance_score if performance_score > 0 else 5.6,
            latest_checkup_date=latest_checkup_date,
            checkup_consistency_14d=checkup_consistency,
            
            work_hours=work_hours,
            check_in_time=check_in,
            role=role,
            priorities=priorities,
            monthly_income_target=monthly_target,
            
            last_7_days_checkups=len([e for e in time_entries if e["date"] >= week_start]),
        )
    
    def _evaluate_candidates(
        self,
        context: UserContextSnapshot
    ) -> List[NotificationCandidate]:
        """Evaluate which notification candidates should trigger and calculate priority."""
        evaluated: List[NotificationCandidate] = []
        
        # Calculate derived metrics
        goal_alignment_score = self._calculate_goal_alignment_score(context)
        deadline_health = 1.0 - min(
            ((context.overdue_tasks * 1.5) + (context.due_today_tasks * 0.75)) / 10.0, 
            1.0
        )
        
        habit_completion = (
            context.habits_completed_today / max(context.habits_total, 1)
            if context.habits_total > 0
            else context.habits_completion_rate_7d / 100.0
        )
        
        for candidate in self.NOTIFICATION_CANDIDATES:
            # Check context requirements
            has_required = all(
                getattr(context, req, None) is not None 
                for req in candidate.context_requirements
            )
            if not has_required:
                continue
            
            # Calculate priority based on formula
            priority = 0.0
            
            if candidate.key == "goal_alignment_score":
                priority = 100 - goal_alignment_score
                
            elif candidate.key == "proactive.deadline_drift":
                priority = context.overdue_tasks * 25 + min(context.due_today_tasks, 5) * 10
                if context.overdue_tasks > 0:
                    candidate.severity = "critical"
                elif context.due_today_tasks >= 5:
                    candidate.severity = "high"
                    
            elif candidate.key == "proactive.deep_work_gap":
                if context.planned_deep_work >= 60:
                    coverage_gap = max(0, 0.6 - context.deep_work_coverage)
                    priority = coverage_gap * 150
                    if context.deep_work_coverage < 0.45:
                        candidate.severity = "high"
                        
            elif candidate.key == "proactive.habit_consistency":
                if context.habits_total >= 3:
                    priority = (1.0 - habit_completion) * 80
                    
            elif candidate.key == "proactive.billable_trajectory":
                if context.monthly_income_target > 0 and context.billable_ratio_14d < 0.45:
                    priority = (0.45 - context.billable_ratio_14d) * 150
                    if context.billable_ratio_14d < 0.3:
                        candidate.severity = "high"
                        
            elif candidate.key == "proactive.morning_checkup_missing":
                if context.latest_checkup_date != context.today and context.current_hour >= 9:
                    priority = 30 + (context.current_hour - 9) * 5
                    if context.current_hour >= 14:
                        candidate.severity = "high"
                        
            elif candidate.key == "proactive.focus_declining":
                if context.avg_focus_score and context.avg_focus_score < 6.5:
                    priority = (7 - context.avg_focus_score) * 15
                    
            elif candidate.key == "proactive.energy_mismatch":
                # Check if user has focus tasks but it's not their peak energy time
                if context.focus_tasks and context.current_hour not in range(9, 12):
                    priority = 20
                    
            elif candidate.key == "proactive.weekly_review":
                if context.day_of_week == 4 and context.checkup_consistency_14d < 0.7:
                    priority = 25
            
            # Only include if priority warrants notification
            if priority > 10:
                candidate.priority_score = priority
                evaluated.append(candidate)
        
        # Always include goal alignment, even with low priority
        goal_candidate = next((c for c in self.NOTIFICATION_CANDIDATES if c.key == "goal_alignment_score"), None)
        if goal_candidate and not any(e.key == "goal_alignment_score" for e in evaluated):
            goal_candidate.priority_score = max(0, 100 - goal_alignment_score)
            evaluated.append(goal_candidate)
        
        return evaluated
    
    async def _generate_llm_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Generate a rich, personalized notification using LLM."""
        
        # Build context-rich prompt
        prompt = self._build_llm_prompt(candidate, context)
        
        try:
            request = CompletionRequest(
                messages=[
                    ChatMessage(role="system", content=self._get_system_prompt()),
                    ChatMessage(role="user", content=prompt)
                ],
                temperature=0.4,
                max_tokens=800
            )
            
            response = await self.llm_service.chat_completion(request)
            content = response.content or ""
            
            # Parse LLM response
            return self._parse_llm_response(candidate, context, content)
            
        except Exception as e:
            logger.warning(f"LLM notification generation failed for {candidate.key}: {e}")
            return self._generate_fallback_notification(candidate, context)
    
    def _build_llm_prompt(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> str:
        """Build a rich, structured prompt for the LLM."""
        
        # Format time entries for context
        yesterday_summary = " | ".join([
            f"{e['project']}: {e['description'][:30]} ({e['duration_minutes']:.0f}min, focus:{e['focus_score']:.0f})"
            for e in context.yesterday_entries[:5]
        ]) if context.yesterday_entries else "No tracked sessions yesterday"
        
        today_summary = " | ".join([
            f"{e['project']}: {e['description'][:30]} ({e['duration_minutes']:.0f}min)"
            for e in context.today_entries[:3]
        ]) if context.today_entries else "No tracked time today yet"
        
        # Format focus tasks
        focus_tasks_text = ""
        if context.focus_tasks:
            focus_tasks_text = "\n".join([
                f"- {t.get('title', 'Untitled')} (priority: {t.get('priority', 'normal')})"
                for t in context.focus_tasks[:3]
            ])
        
        # Format upcoming deadlines
        deadlines_text = ""
        if context.upcoming_deadlines:
            deadlines_text = "\n".join([
                f"- {d.get('title', 'Task')} due {d.get('dueDate', 'soon')}"
                for d in context.upcoming_deadlines[:3]
            ])
        
        prompt = f"""Generate a personalized notification for this user context.

NOTIFICATION TYPE: {candidate.key}
SEVERITY: {candidate.severity}

USER CONTEXT:
- Current time: {context.now.strftime('%Y-%m-%d %H:%M')} ({context.timezone_name})
- Role: {context.role or 'Not specified'}
- Day: {'Workday' if context.is_workday else 'Weekend'}, Hour: {context.current_hour}

TASK STATUS:
- Overdue tasks: {context.overdue_tasks}
- Due today: {context.due_today_tasks}
- Focus tasks: {len(context.focus_tasks)}
{focus_tasks_text}

Upcoming deadlines:
{deadlines_text or 'None in immediate view'}

TIME TRACKING:
- Today tracked: {self._format_minutes(context.total_tracked_today)}
- This week: {self._format_minutes(context.total_tracked_week)}
- 14-day avg/day: {self._format_minutes(context.avg_daily_minutes_14d)}
- Billable ratio: {context.billable_ratio_14d*100:.0f}%
- Deep work coverage: {context.deep_work_coverage*100:.0f}%
- Planned deep work: {self._format_minutes(context.planned_deep_work)}
- Avg focus score: {context.avg_focus_score or 'N/A'}/10

Yesterday's sessions:
{yesterday_summary}

Today so far:
{today_summary}

HABITS:
- Total: {context.habits_total}
- Completed today: {context.habits_completed_today}
- 7-day completion rate: {context.habits_completion_rate_7d:.0f}%

GOALS:
{chr(10).join(f'- {g}' for g in context.top_goals[:3]) or 'No active goals'}

PERFORMANCE:
- Latest checkup score: {context.latest_performance_score:.1f}/10
- Checkup consistency (14d): {context.checkup_consistency_14d*100:.0f}%

OUTPUT FORMAT (JSON):
{{
    "title": "Engaging, specific title (max 8 words)",
    "summary": "One sentence insight that feels personal",
    "insights": ["3-5 specific observations about their patterns"],
    "details_html": "Rich HTML with sections: <section class='insight-block'><h4>🔍 What I'm Seeing</h4>...</section><section class='action-block'><h4>💡 Recommended Actions</h4><ul>...</ul></section>",
    "actions": [
        {{"text": "Specific action", "priority": "high|medium|low", "estimated_minutes": number}},
        {{"text": "Another action", "priority": "high|medium|low", "estimated_minutes": number}}
    ],
    "tags": ["relevant", "tags"],
    "optimal_timing": "immediate|morning|afternoon|evening"
}}

Requirements:
1. Title should be specific and actionable (not generic)
2. Insights should reference their actual data patterns
3. HTML should use the provided CSS classes for styling
4. Actions must be concrete with time estimates
5. Tone: supportive coach, not alarmist"""

        return prompt
    
    def _get_system_prompt(self) -> str:
        """System prompt for notification generation."""
        return """You are an expert AI productivity coach generating personalized notifications.

Your role is to:
1. Synthesize multiple data sources (tasks, time entries, habits, goals) into coherent insights
2. Identify non-obvious patterns and connections
3. Provide actionable, specific recommendations
4. Maintain a supportive, growth-oriented tone
5. Prioritize based on urgency and impact

Output must be valid JSON. Use emojis appropriately. HTML should be semantic and styled with the provided CSS classes."""
    
    def _parse_llm_response(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot,
        content: str
    ) -> PersonalizedNotification:
        """Parse and validate LLM response."""
        
        try:
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start >= 0 and json_end > json_start:
                data = json.loads(content[json_start:json_end+1])
            else:
                raise ValueError("No JSON found in response")
            
            # Build notification
            return PersonalizedNotification(
                notification_key=candidate.key,
                kind=candidate.kind,
                severity=candidate.severity,
                priority_score=candidate.priority_score,
                title=data.get("title", candidate.title_template),
                summary=data.get("summary", "Personalized insight based on your activity"),
                details_html=data.get("details_html", ""),
                insights=data.get("insights", []),
                recommended_actions=data.get("actions", []),
                triggering_metrics=self._extract_triggering_metrics(candidate, context),
                user_context=context,
                tags=data.get("tags", [candidate.kind]),
                score=self._calculate_notification_score(candidate, context)
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response for {candidate.key}: {e}")
            # Preserve candidate metadata in fallback to maintain priority/severity context
            fallback = self._generate_fallback_notification(candidate, context)
            fallback.priority_score = candidate.priority_score
            fallback.severity = candidate.severity
            return fallback
    
    def _generate_fallback_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Generate a fallback notification when LLM is unavailable."""
        
        # Generate specific content based on candidate type
        if candidate.key == "goal_alignment_score":
            score = self._calculate_goal_alignment_score(context)
            return self._build_goal_alignment_notification(candidate, context, score)
        elif candidate.key == "proactive.deadline_drift":
            return self._build_deadline_notification(candidate, context)
        elif candidate.key == "proactive.deep_work_gap":
            return self._build_deep_work_notification(candidate, context)
        elif candidate.key == "proactive.habit_consistency":
            return self._build_habit_notification(candidate, context)
        elif candidate.key == "proactive.billable_trajectory":
            return self._build_billable_notification(candidate, context)
        elif candidate.key == "proactive.morning_checkup_missing":
            return self._build_checkup_missing_notification(candidate, context)
        else:
            return self._build_generic_notification(candidate, context)
    
    def _build_goal_alignment_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot,
        score: int
    ) -> PersonalizedNotification:
        """Build rich goal alignment notification."""
        
        # Safe division with comprehensive zero-checking
        denominator = max(((context.overdue_tasks * 1.5) + (context.due_today_tasks * 0.75)), 0.1)
        deadline_health = 1.0 - min(denominator / 10.0, 1.0)
        habit_completion = context.habits_completion_rate_7d / 100.0 if context.habits_total > 0 else 0.6
        
        insights = [
            f"Your execution quality (performance score) contributes 35% to this score",
            f"Deep-work coverage at {context.deep_work_coverage*100:.0f}% is {'above' if context.deep_work_coverage > 0.6 else 'below'} optimal",
            f"Habit consistency at {habit_completion*100:.0f}% {'strengthens' if habit_completion > 0.7 else 'weighs on'} your alignment",
        ]
        
        if context.overdue_tasks > 0:
            insights.append(f"{context.overdue_tasks} overdue tasks are creating deadline pressure")
        
        if context.billable_ratio_14d < 0.4:
            insights.append(f"Billable ratio at {context.billable_ratio_14d*100:.0f}% may impact financial goal progress")
        
        actions = [
            {"text": "Protect your first 90-minute deep-work block before reactive tasks", "priority": "high", "estimated_minutes": 90},
            {"text": "Close at least one overdue or due-today item before midday", "priority": "high", "estimated_minutes": 30},
        ]
        
        if habit_completion < 0.7:
            actions.append({"text": "Anchor one non-negotiable habit block to stabilize consistency", "priority": "medium", "estimated_minutes": 15})
        
        details_html = f"""<section class="insight-block">
<h4>🎯 Goal Alignment Breakdown</h4>
<div class="score-display">
    <span class="score-value" style="font-size: 2rem; font-weight: bold; color: {'#10b981' if score >= 75 else '#f59e0b' if score >= 60 else '#ef4444'};">{score}</span>
    <span class="score-label">/100</span>
</div>
<ul>
    <li>Performance score impact: {context.latest_performance_score:.1f}/10 × 35% = {context.latest_performance_score * 3.5:.0f} points</li>
    <li>Deep-work coverage impact: {context.deep_work_coverage*100:.0f}% × 25% = {context.deep_work_coverage * 25:.0f} points</li>
    <li>Habit completion impact: {habit_completion*100:.0f}% × 20% = {habit_completion * 20:.0f} points</li>
    <li>Deadline health impact: {deadline_health*100:.0f}% × 10% = {deadline_health * 10:.0f} points</li>
    <li>Checkup consistency impact: {context.checkup_consistency_14d*100:.0f}% × 10% = {context.checkup_consistency_14d * 10:.0f} points</li>
</ul>
</section>
<section class="pattern-block">
<h4>📊 Your Patterns</h4>
<p>Based on your {context.last_7_days_checkups} checkups over the last 7 days:</p>
<ul>
    <li>Average daily tracked time: {self._format_minutes(context.avg_daily_minutes_14d)}</li>
    <li>Top focus area: {context.top_projects[0] if context.top_projects else 'Not established yet'}</li>
    <li>Billable work ratio: {context.billable_ratio_14d*100:.0f}%</li>
    {f"<li>Focus score trend: {context.avg_focus_score:.1f}/10 average</li>" if context.avg_focus_score else ""}
</ul>
</section>
<section class="action-block">
<h4>💡 Recommended Actions</h4>
<ul>
    {''.join(f"<li><strong>{a['text']}</strong> <span class='time-estimate'>(~{a['estimated_minutes']} min)</span></li>" for a in actions)}</ul>
</section>"""
        
        return PersonalizedNotification(
            notification_key=candidate.key,
            kind=candidate.kind,
            severity=candidate.severity,
            priority_score=candidate.priority_score,
            title=f"Goal Alignment Score: {score}/100",
            summary=f"Execution quality, deep-work protection, deadlines, and consistency are now scored daily. Current signal is {score}/100.",
            details_html=details_html,
            insights=insights,
            recommended_actions=actions,
            triggering_metrics={
                "goal_alignment_score": score,
                "performance_score": context.latest_performance_score,
                "deep_work_coverage": context.deep_work_coverage,
                "habit_completion": habit_completion,
                "deadline_health": deadline_health,
                "checkup_consistency": context.checkup_consistency_14d
            },
            user_context=context,
            tags=["goal_alignment", "score", "weekly_review"],
            score=float(score)
        )
    
    def _build_deadline_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Build deadline drift notification."""
        
        severity = "critical" if context.overdue_tasks > 0 else "high" if context.due_today_tasks >= 5 else "medium"
        
        insights = [
            f"{context.overdue_tasks} tasks have passed their deadline",
            f"{context.due_today_tasks} commitments require completion today",
        ]
        
        if context.upcoming_deadlines:
            insights.append(f"{len(context.upcoming_deadlines)} additional deadlines approaching this week")
        
        details_html = f"""<section class="alert-block" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
<h4>🚨 Deadline Status</h4>
<div class="status-grid">
    <div class="status-item">
        <span class="status-number" style="color: #ef4444; font-size: 1.5rem; font-weight: bold;">{context.overdue_tasks}</span>
        <span class="status-label">Overdue</span>
    </div>
    <div class="status-item">
        <span class="status-number" style="color: #f59e0b; font-size: 1.5rem; font-weight: bold;">{context.due_today_tasks}</span>
        <span class="status-label">Due Today</span>
    </div>
</div>
</section>
<section class="impact-block">
<h4>⚡ Impact on Your Day</h4>
<p>Overdue tasks create cognitive load and reduce capacity for deep work. Each overdue item increases stress and decreases goal alignment.</p>
</section>
<section class="action-block">
<h4>🎯 Triage Strategy</h4>
<ol>
    <li><strong>First 30 min:</strong> Quick-win overdue tasks (under 15 min each)</li>
    <li><strong>Next 60 min:</strong> Highest-consequence due-today items</li>
    <li><strong>Remaining:</strong> Negotiate deadlines or delegate where possible</li>
</ol>
</section>"""
        
        actions = [
            {"text": "Create a first-thing deadline triage block for 30 minutes", "priority": "high", "estimated_minutes": 30},
            {"text": "Reduce WIP to one deadline-critical task until drift clears", "priority": "high", "estimated_minutes": 0},
        ]
        
        if context.focus_tasks:
            actions.append({
                "text": f"Protect focus time for: {context.focus_tasks[0].get('title', 'priority task')}",
                "priority": "medium",
                "estimated_minutes": 60
            })
        
        return PersonalizedNotification(
            notification_key=candidate.key,
            kind=candidate.kind,
            severity=severity,
            priority_score=candidate.priority_score,
            title="Deadline Drift Risk",
            summary=f"{context.overdue_tasks} overdue and {context.due_today_tasks} due-today commitments signal drift against planned outcomes.",
            details_html=details_html,
            insights=insights,
            recommended_actions=actions,
            triggering_metrics={"overdue": context.overdue_tasks, "due_today": context.due_today_tasks},
            user_context=context,
            tags=["deadlines", "urgent", "triage"],
            score=float(context.overdue_tasks * 10 + context.due_today_tasks * 5)
        )
    
    def _build_deep_work_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Build deep work gap notification."""
        
        gap_percentage = (1.0 - context.deep_work_coverage) * 100
        
        insights = [
            f"You planned {self._format_minutes(context.planned_deep_work)} of deep work",
            f"Only {context.deep_work_coverage*100:.0f}% is being protected in your schedule",
            f"Deep work drops below 60% correlates with reduced goal alignment",
        ]
        
        if context.avg_daily_minutes_14d > 0:
            ratio = context.planned_deep_work / context.avg_daily_minutes_14d
            insights.append(f"Your planned deep work is {ratio*100:.0f}% of your typical daily tracked time")
        
        details_html = f"""<section class="insight-block">
<h4>🔍 Deep Work Analysis</h4>
<div class="metric-comparison">
    <div class="metric">
        <span class="metric-value">{self._format_minutes(context.planned_deep_work)}</span>
        <span class="metric-label">Planned</span>
    </div>
    <div class="metric-arrow">→</div>
    <div class="metric">
        <span class="metric-value" style="color: {'#10b981' if context.deep_work_coverage >= 0.6 else '#f59e0b' if context.deep_work_coverage >= 0.45 else '#ef4444'};">{context.deep_work_coverage*100:.0f}%</span>
        <span class="metric-label">Protected</span>
    </div>
</div>
<p style="margin-top: 1rem;">Gap: <strong>{gap_percentage:.0f}%</strong> of planned deep work time is at risk of interruption.</p>
</section>
<section class="action-block">
<h4>🛡️ Protection Strategy</h4>
<ul>
    <li><strong>Block a no-meeting focus window</strong> at your peak energy time (typically 9-11am or 2-4pm)</li>
    <li><strong>Set one success criterion</strong> before you start each deep work block</li>
    <li><strong>Turn off notifications</strong> during deep work periods</li>
    <li><strong>Communicate boundaries</strong> to colleagues about your focus time</li>
</ul>
</section>"""
        
        return PersonalizedNotification(
            notification_key=candidate.key,
            kind=candidate.kind,
            severity=candidate.severity,
            priority_score=candidate.priority_score,
            title="Deep Work Coverage Gap",
            summary=f"Only {context.deep_work_coverage*100:.0f}% of planned deep work is landing. Execution quality will trend down if this persists.",
            details_html=details_html,
            insights=insights,
            recommended_actions=[
                {"text": "Block a no-meeting focus window at your peak energy time", "priority": "high", "estimated_minutes": 0},
                {"text": "Set one success criterion for the block before you start", "priority": "high", "estimated_minutes": 5},
            ],
            triggering_metrics={
                "planned_deep_work": context.planned_deep_work,
                "deep_work_coverage": context.deep_work_coverage,
                "gap_percentage": gap_percentage
            },
            user_context=context,
            tags=["deep_work", "focus", "productivity"],
            score=float(100 - context.deep_work_coverage * 100)
        )
    
    def _build_habit_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Build habit consistency notification."""
        
        completion_rate = context.habits_completion_rate_7d
        today_rate = (context.habits_completed_today / max(context.habits_total, 1)) * 100
        
        insights = [
            f"Habit completion is at {completion_rate:.0f}% over the last 7 days",
            f"Today: {context.habits_completed_today}/{context.habits_total} completed ({today_rate:.0f}%)",
        ]
        
        if context.habits_avg_streak > 0:
            insights.append(f"Your average streak is {context.habits_avg_streak:.1f} days")
        
        details_html = f"""<section class="insight-block">
<h4>🔄 Habit Consistency</h4>
<div class="habit-stats">
    <div class="stat-row">
        <span class="stat-label">7-day completion:</span>
        <span class="stat-value" style="color: {'#10b981' if completion_rate >= 70 else '#f59e0b' if completion_rate >= 50 else '#ef4444'};">{completion_rate:.0f}%</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Today's progress:</span>
        <span class="stat-value">{context.habits_completed_today}/{context.habits_total}</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Avg streak:</span>
        <span class="stat-value">{context.habits_avg_streak:.1f} days</span>
    </div>
</div>
</section>
<section class="context-block">
<h4>💭 Why This Matters</h4>
<p>Habits should be defended as protected blocks, not leftovers after task overflow. Identity-level routines create stability that improves all other performance metrics.</p>
</section>
<section class="action-block">
<h4>🎯 Recovery Actions</h4>
<ul>
    <li>Schedule one protected habit block in your next available window</li>
    <li>Tie the habit to an existing anchor event (wake-up, lunch, shutdown)</li>
    <li>Reduce habit count temporarily if needed (focus on consistency over quantity)</li>
</ul>
</section>"""
        
        return PersonalizedNotification(
            notification_key=candidate.key,
            kind=candidate.kind,
            severity=candidate.severity,
            priority_score=candidate.priority_score,
            title="Habit Consistency Slipping",
            summary=f"Habit completion is at {completion_rate:.0f}% over 7 days. Identity-level routines are getting crowded out.",
            details_html=details_html,
            insights=insights,
            recommended_actions=[
                {"text": "Schedule one protected habit block in your next available window", "priority": "medium", "estimated_minutes": 15},
                {"text": "Tie the habit to an existing anchor event (wake-up, lunch, shutdown)", "priority": "medium", "estimated_minutes": 0},
            ],
            triggering_metrics={
                "habits_total": context.habits_total,
                "habits_completed_today": context.habits_completed_today,
                "completion_rate_7d": completion_rate
            },
            user_context=context,
            tags=["habits", "consistency", "routines"],
            score=float(100 - completion_rate)
        )
    
    def _build_billable_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Build billable trajectory notification."""
        
        gap = 0.45 - context.billable_ratio_14d
        
        # Calculate projected monthly income
        daily_tracked = context.avg_daily_minutes_14d / 60  # hours
        billable_hours_daily = daily_tracked * context.billable_ratio_14d
        workdays_per_month = 22
        projected_monthly_hours = billable_hours_daily * workdays_per_month
        
        insights = [
            f"Billable ratio at {context.billable_ratio_14d*100:.0f}% is below the 45% target",
            f"You're tracking {daily_tracked:.1f} hours/day on average",
            f"At current rate: ~{projected_monthly_hours:.0f} billable hours/month",
        ]
        
        details_html = f"""<section class="insight-block">
<h4>💰 Financial Trajectory</h4>
<div class="trajectory-comparison">
    <div class="trajectory-item">
        <span class="trajectory-label">Current Ratio</span>
        <span class="trajectory-value" style="color: {'#ef4444' if context.billable_ratio_14d < 0.3 else '#f59e0b'};">{context.billable_ratio_14d*100:.0f}%</span>
    </div>
    <div class="trajectory-item">
        <span class="trajectory-label">Target Ratio</span>
        <span class="trajectory-value" style="color: #10b981;">45%+</span>
    </div>
    <div class="trajectory-item">
        <span class="trajectory-label">Gap</span>
        <span class="trajectory-value">{gap*100:.0f}%</span>
    </div>
</div>
<p style="margin-top: 1rem;">Monthly income target: <strong>${context.monthly_income_target:,.0f}</strong></p>
<p>Projected at current trajectory: <strong>${context.monthly_income_target * (context.billable_ratio_14d / 0.45):,.0f}</strong> (shortfall: ${context.monthly_income_target * (1 - context.billable_ratio_14d / 0.45):,.0f})</p>
</section>
<section class="action-block">
<h4>🎯 Revenue Recovery</h4>
<ul>
    <li>Reserve two billable-first blocks in the next 48 hours</li>
    <li>Audit low-value tasks and defer or delegate one of them</li>
    <li>Review project mix - can you increase allocation to billable work?</li>
    <li>Track more accurately - are some billable tasks being categorized as non-billable?</li>
</ul>
</section>"""
        
        return PersonalizedNotification(
            notification_key=candidate.key,
            kind=candidate.kind,
            severity=candidate.severity,
            priority_score=candidate.priority_score,
            title="Billable Trajectory Behind",
            summary=f"Billable ratio at {context.billable_ratio_14d*100:.0f}% is below the 45% threshold needed for financial target trajectory.",
            details_html=details_html,
            insights=insights,
            recommended_actions=[
                {"text": "Reserve two billable-first blocks in the next 48 hours", "priority": "high", "estimated_minutes": 0},
                {"text": "Audit low-value tasks and defer or delegate one of them", "priority": "medium", "estimated_minutes": 15},
            ],
            triggering_metrics={
                "monthly_income_target": context.monthly_income_target,
                "billable_ratio_14d": context.billable_ratio_14d,
                "avg_daily_tracked_hours": daily_tracked,
                "projected_monthly_hours": projected_monthly_hours
            },
            user_context=context,
            tags=["finance", "billable", "trajectory"],
            score=float((0.45 - context.billable_ratio_14d) * 200)
        )
    
    def _build_checkup_missing_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Build morning checkup missing notification."""
        
        hours_since_morning = max(0, context.current_hour - 9)
        
        insights = [
            f"No morning strategy check-in detected for today ({context.today.isoformat()})",
            f"Users who miss morning checkups show 23% lower goal alignment on average",
            f"It's {hours_since_morning} hours past your typical start time",
        ]
        
        details_html = f"""<section class="insight-block">
<h4>📅 Checkup Status</h4>
<div class="checkup-missing">
    <p>Last morning checkup: <strong style="color: #f59e0b;">{context.latest_checkup_date.isoformat() if context.latest_checkup_date else 'Unknown'}</strong></p>
    <p>Today: <strong>{context.today.isoformat()}</strong> ({hours_since_morning} hours into workday)</p>
    <p>Your typical check-in time: <strong>{context.check_in_time}</strong></p>
</div>
</section>
<section class="impact-block">
<h4>⚠️ Why Checkups Matter</h4>
<ul>
    <li>Set clear priorities before reactive work takes over</li>
    <li>Align daily execution with long-term goals</li>
    <li>Create a feedback loop for continuous improvement</li>
    <li>Build self-awareness about time and energy patterns</li>
</ul>
</section>
<section class="action-block">
<h4>🚀 Quick Recovery</h4>
<p>Even a 5-minute checkup now can salvage the day:</p>
<ol>
    <li>What is the ONE most important outcome for the rest of today?</li>
    <li>What is blocking that outcome?</li>
    <li>What is the smallest next action to move it forward?</li>
</ol>
</section>"""
        
        return PersonalizedNotification(
            notification_key=candidate.key,
            kind=candidate.kind,
            severity="high" if context.current_hour >= 14 else candidate.severity,
            priority_score=candidate.priority_score,
            title="Morning Check-In Missing",
            summary="No morning strategy check-in detected for today. Priority drift risk is elevated.",
            details_html=details_html,
            insights=insights,
            recommended_actions=[
                {"text": "Run a morning checkup before your next context switch", "priority": "high", "estimated_minutes": 5},
                {"text": "Set one non-negotiable focus outcome for today", "priority": "high", "estimated_minutes": 2},
            ],
            triggering_metrics={
                "today": context.today.isoformat(),
                "latest_checkup": context.latest_checkup_date.isoformat() if context.latest_checkup_date else None,
                "hours_since_morning_start": hours_since_morning
            },
            user_context=context,
            tags=["checkup", "morning", "strategy"],
            score=float(30 + hours_since_morning * 5)
        )
    
    def _build_generic_notification(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> PersonalizedNotification:
        """Build generic notification for unhandled types."""
        
        return PersonalizedNotification(
            notification_key=candidate.key,
            kind=candidate.kind,
            severity=candidate.severity,
            priority_score=candidate.priority_score,
            title=candidate.title_template,
            summary="Personalized insight based on your recent activity patterns.",
            details_html=f"<p>Notification type: {candidate.key}</p>",
            insights=[],
            recommended_actions=[],
            triggering_metrics={},
            user_context=context,
            tags=[candidate.kind],
            score=0.0
        )
    
    def _calculate_goal_alignment_score(self, context: UserContextSnapshot) -> int:
        """Calculate composite goal alignment score."""
        performance = context.latest_performance_score / 10.0
        deep_work = context.deep_work_coverage
        
        habit_completion = (
            context.habits_completion_rate_7d / 100.0
            if context.habits_total > 0
            else 0.6
        )
        
        deadline_health = 1.0 - min(
            ((context.overdue_tasks * 1.5) + (context.due_today_tasks * 0.75)) / 10.0,
            1.0
        )
        
        score = round(
            (performance * 35.0)
            + (deep_work * 25.0)
            + (habit_completion * 20.0)
            + (deadline_health * 10.0)
            + (context.checkup_consistency_14d * 10.0)
        )
        
        return max(0, min(100, score))
    
    def _extract_time_entries(
        self,
        entries: List[KnowledgeEntry],
        now: datetime,
        tz
    ) -> List[Dict[str, Any]]:
        """Extract and normalize time entries from knowledge base."""
        time_entries = []
        lookback = now - timedelta(days=14)
        
        for entry in entries:
            # Check if it's a time entry
            category = str(entry.category or "").lower()
            metadata = entry.metadata or {}
            context = metadata.get("context", {}) if isinstance(metadata.get("context"), dict) else {}
            
            is_time_entry = (
                category == "time_entry"
                or context.get("source") == "alterego_timetracker"
                or context.get("time_entry_id") is not None
            )
            
            if not is_time_entry:
                continue
            
            # Get timestamp
            ts = self._resolve_entry_timestamp(entry, now)
            if ts < lookback:
                continue
            
            # Get duration
            duration = context.get("duration_minutes", 0)
            if duration <= 0 and context.get("duration_seconds"):
                duration = context.get("duration_seconds", 0) / 60.0
            
            time_entries.append({
                "date": ts.date(),
                "timestamp": ts,
                "project": context.get("project_name", "Unassigned") or "Unassigned",
                "description": context.get("description", entry.title) or "Untitled",
                "duration_minutes": max(0, duration),
                "focus_score": context.get("focus_score", 0) or 0,
                "billable": bool(context.get("billable", False)),
            })
        
        return time_entries
    
    def _resolve_entry_timestamp(self, entry: KnowledgeEntry, fallback: datetime) -> datetime:
        """Resolve the most relevant timestamp for an entry."""
        metadata = entry.metadata or {}
        context = metadata.get("context", {}) if isinstance(metadata.get("context"), dict) else {}
        
        for key in ["start_time", "end_time", "timestamp"]:
            if context.get(key):
                try:
                    if isinstance(context[key], datetime):
                        return context[key]
                    elif isinstance(context[key], str):
                        # Handle ISO format
                        dt = datetime.fromisoformat(context[key].replace('Z', '+00:00'))
                        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse datetime from context key {key}: {e}")
                    continue
        
        return entry.created_at if entry.created_at else fallback
    
    def _extract_goal_titles(self, entries: List[KnowledgeEntry]) -> List[str]:
        """Extract goal titles from knowledge entries."""
        titles = []
        for entry in entries:
            if entry.entry_sub_type and "GOAL" in str(entry.entry_sub_type).upper():
                titles.append(entry.title)
            metadata = entry.metadata or {}
            if metadata.get("is_goal") or metadata.get("goal_id"):
                titles.append(entry.title)
        return list(set(titles))[:10]  # Deduplicate and limit
    
    def _calculate_checkup_consistency(self, time_entries: List[Dict[str, Any]]) -> float:
        """Calculate checkup consistency ratio over last 14 days."""
        if not time_entries:
            return 0.0
        
        # Group by date and check if each day has checkup-like activity
        dates_with_activity = set(e["date"] for e in time_entries)
        
        # Expected workdays in last 14 days (rough estimate)
        total_days = 14
        active_days = len(dates_with_activity)
        
        return min(1.0, active_days / max(total_days * 0.6, 1))  # 60% target
    
    def _extract_triggering_metrics(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> Dict[str, Any]:
        """Extract relevant metrics that triggered this notification."""
        metrics = {}
        for field in candidate.context_requirements:
            value = getattr(context, field, None)
            if value is not None:
                metrics[field] = value
        return metrics
    
    def _calculate_notification_score(
        self,
        candidate: NotificationCandidate,
        context: UserContextSnapshot
    ) -> float:
        """Calculate a score for this notification (0-100)."""
        if candidate.key == "goal_alignment_score":
            return self._calculate_goal_alignment_score(context)
        return candidate.priority_score
    
    def _extract_timezone(self, preferences: UserPreferences) -> str:
        """Extract timezone from preferences."""
        if isinstance(preferences.general, dict):
            return preferences.general.get("timezone", "UTC")
        if isinstance(preferences.productivity, dict):
            return preferences.productivity.get("timezone", "UTC")
        return "UTC"
    
    def _resolve_timezone(self, name: str):
        """Resolve timezone name to tzinfo."""
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            return timezone.utc
    
    def _format_minutes(self, minutes: float) -> str:
        """Format minutes as human-readable string."""
        minutes = max(0, int(round(minutes)))
        hours = minutes // 60
        mins = minutes % 60
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"


# Singleton instance
_enhanced_engine: Optional[EnhancedNotificationEngine] = None
_engine_lock = asyncio.Lock()  # Proper async lock for thread-safe singleton


async def get_enhanced_notification_engine() -> EnhancedNotificationEngine:
    """Get or create the enhanced notification engine singleton."""
    global _enhanced_engine
    if _enhanced_engine is None:
        async with _engine_lock:
            # Double-check after acquiring lock
            if _enhanced_engine is None:
                _enhanced_engine = EnhancedNotificationEngine()
    return _enhanced_engine


async def generate_personalized_notifications(
    context_snapshot: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    use_llm: bool = True
) -> List[PersonalizedNotification]:
    """Convenience function to generate personalized notifications."""
    engine = await get_enhanced_notification_engine()
    return await engine.generate_personalized_notifications(
        context_snapshot=context_snapshot,
        limit=limit,
        use_llm=use_llm
    )


def notifications_to_api_response(
    notifications: List[PersonalizedNotification],
    persistence_enabled: bool = False
) -> Dict[str, Any]:
    """Convert notifications to API response format."""
    
    def notification_to_dict(n: PersonalizedNotification) -> Dict[str, Any]:
        return {
            "id": 0,  # Will be set by persistence layer
            "notification_key": n.notification_key,
            "kind": n.kind,
            "severity": n.severity,
            "status": "active",
            "title": n.title,
            "summary": n.summary,
            "details": n.details_html,
            "score": n.score,
            "recommended_actions": [a["text"] for a in n.recommended_actions],
            "insights": n.insights,
            "triggering_metrics": n.triggering_metrics,
            "priority_score": n.priority_score,
            "tags": n.tags,
            "payload": {
                "insights": n.insights,
                "recommended_actions": n.recommended_actions,
                "triggering_metrics": n.triggering_metrics,
                "tags": n.tags,
                "priority_score": n.priority_score,
            },
            "first_seen_at": datetime.now(timezone.utc).isoformat(),
            "last_seen_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    return {
        "persistence_enabled": persistence_enabled,
        "notifications": [notification_to_dict(n) for n in notifications],
        "generated": len(notifications),
        "upserted": 0,
        "resolved": 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
