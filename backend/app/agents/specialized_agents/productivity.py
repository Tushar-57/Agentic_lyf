"""
Productivity Agent - Specialized agent for productivity, task management, and goal tracking.
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base import BaseAgent, AgentType, AgentCapability, AgentState
from ..prompts import get_agent_prompt
from ...llm.service import get_llm_service
from ...llm.base import CompletionRequest, ChatMessage
from ...services.knowledge_base import get_knowledge_base_service
from ...models.knowledge import KnowledgeEntryType
from ...services.interaction_recorder import get_interaction_recorder
from ...utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.AGENT)


class ProductivityAgent(BaseAgent):
    """Specialized agent for productivity, task management, and goal tracking."""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="task_management",
                description="Create, organize, and track tasks and projects",
                parameters={"priority_levels": True, "deadline_tracking": True}
            ),
            AgentCapability(
                name="goal_setting",
                description="Set and track personal and professional goals",
                parameters={"smart_goals": True, "progress_tracking": True}
            ),
            AgentCapability(
                name="time_management",
                description="Optimize time usage and scheduling",
                parameters={"time_blocking": True, "productivity_analysis": True}
            ),
            AgentCapability(
                name="workflow_optimization",
                description="Improve workflows and productivity systems",
                parameters={"automation_suggestions": True, "efficiency_tips": True}
            )
        ]
        
        super().__init__(
            agent_id="productivity_specialized",
            agent_type=AgentType.PRODUCTIVITY,
            capabilities=capabilities,
            system_prompt=get_agent_prompt(AgentType.PRODUCTIVITY)
        )
        
        self.knowledge_base = get_knowledge_base_service()
    
    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute productivity-related requests with contextual knowledge."""
        try:
            user_input = state.get("user_input", "")
            state_context = state.get("context", {}) if isinstance(state, dict) else {}
            logger.info("productivity_processing", "ProductivityAgent processing", {"input_preview": user_input[:100]})
            
            # Get contextual knowledge from knowledge base
            contextual_knowledge = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="productivity",
                max_results=10
            )

            merged_context = self._merge_with_routing_context(contextual_knowledge, state_context)
            self._log_rag_trace("context_loaded", user_input=user_input, context=merged_context)

            normalized_input = user_input.lower()
            
            # Determine productivity task type
            if self._is_performance_review_request(normalized_input):
                response = await self._handle_performance_review(user_input, merged_context)
            elif self._is_activity_fact_check_request(normalized_input):
                response = self._handle_activity_fact_check(user_input, merged_context)
            elif any(keyword in normalized_input for keyword in ["task", "todo", "organize", "project"]):
                response = await self._handle_task_management(user_input, merged_context)
            elif any(keyword in normalized_input for keyword in ["goal", "objective", "target", "achieve"]):
                response = await self._handle_goal_setting(user_input, merged_context)
            elif any(keyword in normalized_input for keyword in ["time", "schedule", "productivity", "focus"]):
                response = await self._handle_time_management(user_input, merged_context)
            else:
                response = await self._handle_general_productivity(user_input, merged_context)
            
            # Create pending interaction for explicit user approval.
            recorder = get_interaction_recorder()
            if recorder:
                interaction_id = await recorder.create_pending_interaction(
                    user_input=user_input,
                    agent_response=response,
                    agent_type="productivity",
                    context=merged_context,
                )
                if interaction_id:
                    logger.info("Created pending interaction %s for productivity approval", interaction_id)
            
            return {
                "response": response,
                "reasoning": {
                    "agent_type": "productivity",
                    "context_used": len(merged_context.get("relevant_interactions", [])),
                    "time_entries_used": len(merged_context.get("recent_time_entries", [])),
                    "profile_role": (merged_context.get("profile_snapshot") or {}).get("role"),
                    "specialized_handling": True
                }
            }
            
        except Exception as e:
            logger.error("productivity_execution_failed", "ProductivityAgent execution failed", error=e)
            return {
                "response": "I'm having trouble with productivity assistance right now. Please try again later.",
                "reasoning": {"error": str(e), "agent_type": "productivity"}
            }
    
    async def _handle_task_management(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle task management requests."""
        try:
            task_context = self._build_productivity_context(context, "tasks")
            day_rundown = self._build_day_rundown(context)
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📋 I'd be happy to help you manage your tasks! What tasks would you like to organize?"
            
            prompt = f"""
            You are a productivity coach. Help the user with task management and organization.
            
            User Request: {user_input}
            Task Context: {task_context}
            Day Rundown Signals: {day_rundown}
            
            Response requirements:
            1. Keep it concise (max 6 bullets, about 140-180 words unless user asks for depth).
            2. Explicitly reference any relevant goals, priorities, or recent time-entry patterns from context.
            3. Include a quick rundown using this framing: what was completed, what may have been missed, and what to do next.
            4. Provide one immediate next action the user can do now.
            5. Ask one focused follow-up question only if critical context is missing.
            """

            self._log_rag_trace(
                "task_prompt",
                user_input=user_input,
                context=context,
                prompt=prompt,
            )
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=420
            )
            
            response = await llm_service.chat_completion(request)
            response_text = response.content or ""
            self._log_rag_trace("task_response", user_input=user_input, context=context, response=response_text)
            return response_text
            
        except Exception as e:
            logger.error("task_management_failed", "Task management failed", error=e)
            return "📋 I'd be happy to help you manage your tasks! What specific tasks would you like to organize?"
    
    async def _handle_goal_setting(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle goal setting and tracking requests."""
        try:
            goal_context = self._build_productivity_context(context, "goals")
            day_rundown = self._build_day_rundown(context)
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "🎯 I'd love to help you set and achieve your goals! What goals are you working on?"
            
            prompt = f"""
            You are a goal-setting productivity coach. Help the user create and track meaningful goals.
            
            User Request: {user_input}
            Goal Context: {goal_context}
            Day Rundown Signals: {day_rundown}
            
            Response requirements:
            1. Keep it concise (max 6 bullets, about 140-180 words unless user asks for depth).
            2. Tie recommendations to the user's stated priorities and active goals from context.
            3. Convert advice into concrete weekly actions.
            4. Explicitly mention what progress appears done vs what is still lagging.
            5. End with one measurable checkpoint.
            """

            self._log_rag_trace(
                "goal_prompt",
                user_input=user_input,
                context=context,
                prompt=prompt,
            )
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=420
            )
            
            response = await llm_service.chat_completion(request)
            response_text = response.content or ""
            self._log_rag_trace("goal_response", user_input=user_input, context=context, response=response_text)
            return response_text
            
        except Exception as e:
            logger.error("goal_setting_failed", "Goal setting failed", error=e)
            return "🎯 I'd love to help you set and achieve your goals! What specific goals would you like to work on?"
    
    async def _handle_time_management(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle time management and productivity optimization."""
        try:
            time_context = self._build_productivity_context(context, "time")
            day_rundown = self._build_day_rundown(context)
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "⏰ I'd be happy to help optimize your time! What time management challenges are you facing?"
            
            prompt = f"""
            You are a time-management productivity coach. Help the user optimize their productivity and time usage.
            
            User Request: {user_input}
            Time Management Context: {time_context}
            Day Rundown Signals: {day_rundown}
            
            Response requirements:
            1. Keep it concise (max 6 bullets, about 140-180 words unless user asks for depth).
            2. Reference relevant recent time-entry patterns and suggest targeted adjustments.
            3. Include a clear rundown section with headings: Completed Today, Missed/At Risk, Next Blocks.
            4. Offer a practical time-blocking plan for the next 24 hours.
            5. End with one immediate action.
            """

            self._log_rag_trace(
                "time_prompt",
                user_input=user_input,
                context=context,
                prompt=prompt,
            )
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=420
            )
            
            response = await llm_service.chat_completion(request)
            response_text = response.content or ""
            self._log_rag_trace("time_response", user_input=user_input, context=context, response=response_text)
            return response_text
            
        except Exception as e:
            logger.error("time_management_failed", "Time management failed", error=e)
            return "⏰ I'd be happy to help optimize your time! What specific time management areas would you like to improve?"
    
    async def _handle_general_productivity(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle general productivity queries."""
        try:
            productivity_context = self._build_productivity_context(context, "general")
            day_rundown = self._build_day_rundown(context)
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "🚀 I'm here to boost your productivity! What can I help you with?"
            
            prompt = f"""
            You are a productivity coach. Provide direct and practical advice.
            
            User Request: {user_input}
            Productivity Context: {productivity_context}
            Day Rundown Signals: {day_rundown}
            
            Response requirements:
            1. Keep response concise (max 5 bullets, about 120-160 words unless user asks for detail).
            2. Ground advice in available profile/goal/time-entry context.
            3. If enough context exists, include what seems done vs missed.
            4. Finish with one next action.
            """

            self._log_rag_trace(
                "general_prompt",
                user_input=user_input,
                context=context,
                prompt=prompt,
            )
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=320
            )
            
            response = await llm_service.chat_completion(request)
            response_text = response.content or ""
            self._log_rag_trace("general_response", user_input=user_input, context=context, response=response_text)
            return response_text
            
        except Exception as e:
            logger.error("general_productivity_failed", "General productivity failed", error=e)
            return "🚀 I'm here to boost your productivity! What specific area would you like help with?"

    def _is_performance_review_request(self, normalized_input: str) -> bool:
        return bool(
            re.search(
                r"\b(how did i do|review|highlights|performance|compare|yesterday|daily summary|day summary)\b",
                normalized_input,
            )
        )

    def _is_activity_fact_check_request(self, normalized_input: str) -> bool:
        return bool(
            re.search(r"\b(did i|have i|was i)\b", normalized_input)
            and re.search(r"\b(today|yesterday)\b", normalized_input)
        )

    async def _handle_performance_review(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle day-review questions with hard factual constraints from full window summaries."""
        try:
            productivity_context = self._build_productivity_context(context, "review")
            day_rundown = self._build_day_rundown(context)
            summary = context.get("time_window_summary") if isinstance(context, dict) else {}
            if not isinstance(summary, dict):
                summary = {}

            hard_facts = {
                "window_label": summary.get("window_label", "Requested window"),
                "entry_count": summary.get("entry_count", 0),
                "total_logged_minutes": summary.get("total_logged_minutes", 0),
                "active_minutes": summary.get("active_minutes", 0),
                "idle_minutes": summary.get("idle_minutes", 0),
                "gap_count": summary.get("gap_count", 0),
                "top_projects": summary.get("top_projects", []),
            }

            llm_service = await get_llm_service()
            if not llm_service:
                return (
                    f"Performance snapshot ({hard_facts['window_label']}): "
                    f"{hard_facts['total_logged_minutes']} tracked minutes across {hard_facts['entry_count']} entries. "
                    "Next action: choose one high-impact task and time-box it for 45 minutes."
                )

            prompt = f"""
            You are a performance review coach.

            User Request: {user_input}
            Productivity Context: {productivity_context}
            Day Rundown Signals: {day_rundown}

            Hard facts (do not alter numbers):
            {hard_facts}

            Response requirements:
            1. Use ONLY the numbers in hard facts. Do not invent totals.
            2. Structure output with headings: Highlights, Gaps, Next Action.
            3. In Gaps, explicitly mention untracked/idle time from hard facts when available.
            4. Tie advice to user priorities/goals from context.
            5. Keep response concise (120-180 words).
            """

            self._log_rag_trace(
                "review_prompt",
                user_input=user_input,
                context=context,
                prompt=prompt,
            )

            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.2,
                max_tokens=420,
            )

            response = await llm_service.chat_completion(request)
            response_text = response.content or ""
            self._log_rag_trace("review_response", user_input=user_input, context=context, response=response_text)
            return response_text
        except Exception as e:
            logger.error("performance_review_failed", "Performance review failed", error=e)
            return "I couldn't complete a reliable performance review right now. Please try again in a moment."

    def _handle_activity_fact_check(self, user_input: str, context: Dict[str, Any]) -> str:
        """Deterministically answer yes/no activity checks from available tracked entries."""
        summary = context.get("time_window_summary") if isinstance(context, dict) else {}
        if not isinstance(summary, dict):
            summary = {}

        activity_phrase = self._extract_activity_phrase(user_input)
        if not activity_phrase:
            return "I couldn't identify the exact activity to verify. Rephrase with the action you want checked."

        top_entries = summary.get("top_entries", []) if isinstance(summary.get("top_entries"), list) else []
        searchable_text = []
        for entry in top_entries:
            if not isinstance(entry, dict):
                continue
            searchable_text.append(
                " ".join(
                    [
                        str(entry.get("project_name") or ""),
                        str(entry.get("description") or ""),
                    ]
                ).strip().lower()
            )

        activity_tokens = [token for token in re.findall(r"[a-zA-Z0-9_]+", activity_phrase.lower()) if len(token) > 2]
        matched_entries = []
        for entry, text_blob in zip(top_entries, searchable_text):
            if all(token in text_blob for token in activity_tokens):
                matched_entries.append(entry)

        window_label = str(summary.get("window_label") or "requested period")
        if matched_entries:
            best_match = matched_entries[0]
            return (
                f"Yes, I found a tracked entry for {activity_phrase} in {window_label.lower()}: "
                f"{best_match.get('project_name', 'Unassigned')} - {best_match.get('description', 'activity')} "
                f"({best_match.get('duration_minutes', 0)}m)."
            )

        if summary.get("has_data"):
            return (
                f"I don't see a clear tracked entry for {activity_phrase} in {window_label.lower()}. "
                "If you did it but didn't log it, add a quick entry so future reviews stay accurate."
            )

        return (
            f"I don't have tracked entries for {window_label.lower()} yet, so I can't confirm whether {activity_phrase} happened. "
            "Log the activity once and I can verify it reliably next time."
        )

    def _extract_activity_phrase(self, user_input: str) -> str:
        normalized = str(user_input or "").strip()
        lowered = normalized.lower()

        patterns = [
            r"did i\s+(.*?)\s+(today|yesterday)",
            r"have i\s+(.*?)\s+(today|yesterday)",
            r"was i\s+(.*?)\s+(today|yesterday)",
        ]

        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                phrase = match.group(1).strip(" ?!.,")
                if phrase:
                    return phrase

        return ""
    
    def _build_productivity_context(self, context: Dict[str, Any], productivity_type: str) -> str:
        """Build productivity context from available knowledge."""
        context_parts = []

        profile_snapshot = context.get("profile_snapshot", {}) if isinstance(context, dict) else {}
        if isinstance(profile_snapshot, dict):
            role = profile_snapshot.get("role")
            priorities = profile_snapshot.get("priorities", []) if isinstance(profile_snapshot.get("priorities"), list) else []
            active_goals = profile_snapshot.get("active_goals", []) if isinstance(profile_snapshot.get("active_goals"), list) else []

            profile_bits = []
            if role:
                profile_bits.append(f"role={role}")
            if priorities:
                profile_bits.append(f"priorities={', '.join(str(item) for item in priorities[:3])}")
            if active_goals:
                profile_bits.append(f"active_goals={', '.join(str(item) for item in active_goals[:2])}")

            if profile_bits:
                context_parts.append("Profile snapshot: " + " | ".join(profile_bits))

        coach_profile = context.get("coach_profile", {}) if isinstance(context, dict) else {}
        if isinstance(coach_profile, dict) and coach_profile:
            coach_style = str(coach_profile.get("style", "")).strip()
            coach_directive = str(coach_profile.get("directive", "")).strip()
            coach_name = str(coach_profile.get("name", "Coach")).strip()
            context_parts.append(
                f"Coach style: {coach_name} ({coach_style}) | directive: {coach_directive}"
            )

        intent_blueprint = context.get("intent_blueprint", {}) if isinstance(context, dict) else {}
        if isinstance(intent_blueprint, dict) and intent_blueprint:
            context_parts.append(
                "Intent blueprint: "
                f"intent={intent_blueprint.get('primary_intent')} | "
                f"outcome={intent_blueprint.get('expected_outcome')} | "
                f"horizon={intent_blueprint.get('time_horizon')}"
            )
        
        # Add agent preferences (productivity preferences from knowledge base)
        if "agent_preferences" in context and context["agent_preferences"]:
            prefs = context["agent_preferences"]
            if isinstance(prefs, dict):
                productivity_prefs = {k: v for k, v in prefs.items() if any(term in k.lower() for term in ["work", "task", "goal", "time", "productivity", "schedule"])}
                if productivity_prefs:
                    context_parts.append(f"Productivity preferences: {productivity_prefs}")

        if context.get("user_preferences"):
            preference_snippets = []
            for pref in context.get("user_preferences", [])[:3]:
                if isinstance(pref, dict):
                    content = str(pref.get("content", "")).strip()
                    category = str(pref.get("category", "")).strip()
                    if content:
                        preference_snippets.append(f"{category}: {content}")
            if preference_snippets:
                context_parts.append("Preference memory: " + " || ".join(preference_snippets))

        if context.get("recent_time_entries"):
            entry_snippets = []
            for item in context.get("recent_time_entries", [])[:3]:
                if not isinstance(item, dict):
                    continue
                project_name = str(item.get("project_name") or "Unassigned").strip()
                description = str(item.get("description") or "work session").strip()
                duration = item.get("duration_minutes")
                duration_label = f"{duration}m" if duration is not None else "duration n/a"
                entry_snippets.append(f"{project_name} - {description} ({duration_label})")
            if entry_snippets:
                context_parts.append("Recent time entries: " + " || ".join(entry_snippets))

        if context.get("relevant_interactions"):
            interaction_snippets = []
            for item in context.get("relevant_interactions", [])[:2]:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                if content:
                    interaction_snippets.append(content[:220])
            if interaction_snippets:
                context_parts.append("Related interactions: " + " || ".join(interaction_snippets))

        time_window_summary = context.get("time_window_summary") if isinstance(context.get("time_window_summary"), dict) else {}
        if time_window_summary.get("has_data"):
            context_parts.append(
                "Time window summary: "
                f"{time_window_summary.get('window_label')} | "
                f"entries={time_window_summary.get('entry_count')} | "
                f"logged_minutes={time_window_summary.get('total_logged_minutes')} | "
                f"idle_minutes={time_window_summary.get('idle_minutes')} | "
                f"gaps={time_window_summary.get('gap_count')}"
            )
        elif time_window_summary.get("window_key") in {"today", "yesterday", "this_week"}:
            context_parts.append(
                f"Time window summary: no tracked entries for {str(time_window_summary.get('window_label')).lower()}."
            )
        
        # Add context summary
        if "context_summary" in context and context["context_summary"]:
            context_parts.append(f"Previous productivity context: {context['context_summary']}")

        if "knowledge_context_summary" in context and context["knowledge_context_summary"]:
            context_parts.append(f"Routing knowledge summary: {context['knowledge_context_summary']}")

        day_rundown = self._build_day_rundown(context)
        if day_rundown:
            context_parts.append(f"Day rundown: {day_rundown}")
        
        return " | ".join(context_parts) if context_parts else f"No specific {productivity_type} context available"

    def _build_day_rundown(self, context: Dict[str, Any]) -> str:
        """Create a compact completed-vs-missed rundown from recent tracked sessions."""
        if not isinstance(context, dict):
            return "No recent tracked sessions available."

        priorities = []
        profile_snapshot = context.get("profile_snapshot") if isinstance(context.get("profile_snapshot"), dict) else {}
        if isinstance(profile_snapshot, dict):
            raw_priorities = profile_snapshot.get("priorities")
            if isinstance(raw_priorities, list):
                priorities = [str(item).strip() for item in raw_priorities if str(item).strip()]

        summary = context.get("time_window_summary") if isinstance(context.get("time_window_summary"), dict) else {}
        if summary.get("has_data"):
            top_entries = summary.get("top_entries", []) if isinstance(summary.get("top_entries"), list) else []
            completed = []
            combined_activity_text: List[str] = []
            for item in top_entries[:4]:
                if not isinstance(item, dict):
                    continue
                description = str(item.get("description") or "work session").strip()
                project = str(item.get("project_name") or "Unassigned").strip()
                duration = item.get("duration_minutes")
                duration_label = ""
                if duration is not None:
                    try:
                        duration_label = f" ({round(float(duration))}m)"
                    except (TypeError, ValueError):
                        duration_label = ""
                completed.append(f"{project}: {description}{duration_label}")
                combined_activity_text.append(f"{project} {description}".lower())

            joined_activity = " ".join(combined_activity_text)
            missed_priorities: List[str] = []
            for priority in priorities[:4]:
                normalized_priority = priority.lower()
                if normalized_priority and normalized_priority not in joined_activity:
                    missed_priorities.append(priority)

            completed_text = "; ".join(completed) if completed else "No concrete completions detected."
            missed_text = "; ".join(missed_priorities[:2]) if missed_priorities else "No obvious priority gaps inferred."
            return (
                f"{summary.get('window_label', 'Requested window')} completed: {completed_text} | "
                f"Missed/at-risk priorities: {missed_text} | "
                f"Tracked minutes (full window): {round(float(summary.get('total_logged_minutes', 0) or 0))} | "
                f"Idle gaps: {summary.get('gap_count', 0)} (~{round(float(summary.get('idle_minutes', 0) or 0))}m)"
            )

        recent_entries = context.get("recent_time_entries", [])
        if not isinstance(recent_entries, list) or not recent_entries:
            return "No recent tracked sessions available."

        completed: List[str] = []
        total_minutes = 0.0
        combined_activity_text: List[str] = []

        for item in recent_entries[:4]:
            if not isinstance(item, dict):
                continue

            description = str(item.get("description") or "work session").strip()
            project = str(item.get("project_name") or "Unassigned").strip()
            duration = item.get("duration_minutes")

            duration_label = ""
            if duration is not None:
                try:
                    duration_float = float(duration)
                    total_minutes += max(0.0, duration_float)
                    duration_label = f" ({round(duration_float)}m)"
                except (TypeError, ValueError):
                    duration_label = ""

            completed.append(f"{project}: {description}{duration_label}")
            combined_activity_text.append(f"{project} {description}".lower())

        missed_priorities: List[str] = []
        joined_activity = " ".join(combined_activity_text)
        for priority in priorities[:4]:
            normalized_priority = priority.lower()
            if normalized_priority and normalized_priority not in joined_activity:
                missed_priorities.append(priority)

        completed_text = "; ".join(completed) if completed else "No concrete completions detected."
        missed_text = "; ".join(missed_priorities[:2]) if missed_priorities else "No obvious priority gaps inferred."

        return (
            f"Completed: {completed_text} | "
            f"Missed/at-risk priorities: {missed_text} | "
            f"Tracked minutes (partial sample): {round(total_minutes)}"
        )

    def _log_rag_trace(
        self,
        stage: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
    ) -> None:
        """Structured observability logs for context->prompt->response path."""
        context = context or {}
        context_counts = {
            "interactions": len(context.get("relevant_interactions", [])) if isinstance(context.get("relevant_interactions"), list) else 0,
            "preferences": len(context.get("user_preferences", [])) if isinstance(context.get("user_preferences"), list) else 0,
            "time_entries": len(context.get("recent_time_entries", [])) if isinstance(context.get("recent_time_entries"), list) else 0,
            "patterns": len(context.get("patterns_and_insights", [])) if isinstance(context.get("patterns_and_insights"), list) else 0,
        }

        logger.info(
            "[RAG_TRACE][productivity][%s] input=%s context_counts=%s summary=%s prompt=%s response=%s",
            stage,
            self._truncate_text(user_input, 140),
            context_counts,
            self._truncate_text(str(context.get("context_summary", "")), 180),
            self._truncate_text(prompt or "", 260),
            self._truncate_text(response or "", 260),
        )

    def _truncate_text(self, value: str, limit: int = 200) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[:limit - 3]}..."

    def _merge_with_routing_context(self, kb_context: Dict[str, Any], state_context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge orchestrator routing context with knowledge-base context."""
        merged_context = dict(kb_context or {})
        if not isinstance(state_context, dict):
            return merged_context

        profile_snapshot = state_context.get("profile_snapshot")
        if isinstance(profile_snapshot, dict) and profile_snapshot:
            merged_context["profile_snapshot"] = profile_snapshot

        knowledge_context_summary = state_context.get("knowledge_context_summary")
        if knowledge_context_summary:
            merged_context["knowledge_context_summary"] = knowledge_context_summary

        coach_profile = state_context.get("coach_profile")
        if isinstance(coach_profile, dict) and coach_profile:
            merged_context["coach_profile"] = coach_profile

        intent_blueprint = state_context.get("intent_blueprint")
        if isinstance(intent_blueprint, dict) and intent_blueprint:
            merged_context["intent_blueprint"] = intent_blueprint

        recent_time_entries = state_context.get("general_recent_time_entries")
        if isinstance(recent_time_entries, list) and recent_time_entries:
            existing = merged_context.get("recent_time_entries", [])
            if not isinstance(existing, list):
                existing = []
            merged_context["recent_time_entries"] = [*existing, *recent_time_entries][:5]

        return merged_context