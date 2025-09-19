"""
Journal Agent - Specialized agent for journaling, reflection, and personal growth tracking.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base import BaseAgent, AgentType, AgentCapability, AgentState
from ..prompts import get_agent_prompt
from ...llm.service import get_llm_service
from ...llm.base import CompletionRequest, ChatMessage
from ...services.knowledge_base import get_knowledge_base_service
from ...models.knowledge import KnowledgeEntryType
from ...services.interaction_recorder import get_interaction_recorder

logger = logging.getLogger(__name__)


class JournalAgent(BaseAgent):
    """Specialized agent for journaling, reflection, and personal growth tracking."""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="daily_reflection",
                description="Guide daily journaling and reflection practices",
                parameters={"mood_tracking": True, "gratitude_practice": True}
            ),
            AgentCapability(
                name="goal_tracking",
                description="Track personal growth goals and milestones",
                parameters={"progress_monitoring": True, "achievement_celebration": True}
            ),
            AgentCapability(
                name="habit_formation",
                description="Support habit building and behavior change",
                parameters={"habit_tracking": True, "streak_monitoring": True}
            ),
            AgentCapability(
                name="emotional_wellness",
                description="Support emotional processing and mental wellness",
                parameters={"mood_analysis": True, "stress_management": True}
            )
        ]
        
        super().__init__(
            agent_id=f"journal_{uuid.uuid4().hex[:8]}",
            agent_type=AgentType.JOURNAL,
            capabilities=capabilities,
            system_prompt=get_agent_prompt(AgentType.JOURNAL)
        )
        
        self.knowledge_base = get_knowledge_base_service()
    
    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute journaling-related requests with contextual knowledge."""
        try:
            user_input = state.get("user_input", "")
            logger.info(f"JournalAgent processing: {user_input}")
            
            # Get contextual knowledge from knowledge base
            contextual_knowledge = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="journal",
                max_results=10
            )
            
            # Determine journaling task type
            if any(keyword in user_input.lower() for keyword in ["journal", "reflection", "reflect", "mood"]):
                response = await self._handle_daily_journaling(user_input, contextual_knowledge)
            elif any(keyword in user_input.lower() for keyword in ["goal", "progress", "achievement", "milestone"]):
                response = await self._handle_goal_tracking(user_input, contextual_knowledge)
            elif any(keyword in user_input.lower() for keyword in ["habit", "streak", "routine", "consistency"]):
                response = await self._handle_habit_tracking(user_input, contextual_knowledge)
            elif any(keyword in user_input.lower() for keyword in ["feeling", "emotion", "stress", "wellness"]):
                response = await self._handle_emotional_wellness(user_input, contextual_knowledge)
            else:
                response = await self._handle_general_journaling(user_input, contextual_knowledge)
            
            # Intelligently record interaction if valuable
            recorder = get_interaction_recorder()
            await recorder.record_if_valuable(
                user_input=user_input,
                agent_response=response,
                agent_type="journal"
            )
            
            return {
                "response": response,
                "reasoning": {
                    "agent_type": "journal",
                    "context_used": len(contextual_knowledge.get("relevant_interactions", [])),
                    "specialized_handling": True
                }
            }
            
        except Exception as e:
            logger.error(f"JournalAgent execution failed: {e}")
            return {
                "response": "I'm having trouble with journaling assistance right now. Please try again later.",
                "reasoning": {"error": str(e), "agent_type": "journal"}
            }
    
    async def _handle_daily_journaling(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle daily journaling and reflection requests."""
        try:
            journal_context = self._build_journal_context(context, "reflection")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📝 I'd be happy to guide your journaling practice! What would you like to reflect on today?"
            
            prompt = f"""
            As a thoughtful journaling guide, help the user with their daily reflection and journaling practice.
            
            User Request: {user_input}
            Journal Context: {journal_context}
            
            Provide:
            1. Thoughtful reflection prompts
            2. Guided questions for deeper insight
            3. Mood and emotion processing support
            4. Gratitude practice suggestions
            5. Personal growth observations
            
            Use emojis and format nicely with gentle, encouraging guidance.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.4,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Daily journaling failed: {e}")
            return "📝 I'd be happy to guide your journaling practice! What would you like to reflect on today?"
    
    async def _handle_goal_tracking(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle goal tracking and milestone celebration."""
        try:
            goals_context = self._build_journal_context(context, "goals")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "🎯 I'd love to help you track your goals! What goals are you working on?"
            
            prompt = f"""
            As a personal growth coach, help the user with goal tracking and achievement recognition.
            
            User Request: {user_input}
            Goals Context: {goals_context}
            
            Provide:
            1. Goal progress assessment
            2. Milestone recognition and celebration
            3. Next steps and action items
            4. Motivation and encouragement
            5. Course correction suggestions if needed
            
            Use emojis and format nicely with motivational, supportive guidance.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Goal tracking failed: {e}")
            return "🎯 I'd love to help you track your goals! What goals are you working on?"
    
    async def _handle_habit_tracking(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle habit tracking and formation support."""
        try:
            habits_context = self._build_journal_context(context, "habits")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "🔄 I'd be happy to help with your habit tracking! What habits are you building?"
            
            prompt = f"""
            As a habit formation expert, help the user with habit tracking and consistency building.
            
            User Request: {user_input}
            Habits Context: {habits_context}
            
            Provide:
            1. Habit streak recognition and motivation
            2. Consistency strategies and tips
            3. Barrier identification and solutions
            4. Habit stacking suggestions
            5. Small wins celebration
            
            Use emojis and format nicely with encouraging, practical advice.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Habit tracking failed: {e}")
            return "🔄 I'd be happy to help with your habit tracking! What habits are you building?"
    
    async def _handle_emotional_wellness(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle emotional processing and wellness support."""
        try:
            wellness_context = self._build_journal_context(context, "wellness")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "💙 I'm here to support your emotional wellness! How are you feeling today?"
            
            prompt = f"""
            As a supportive wellness companion, help the user with emotional processing and mental wellness.
            
            User Request: {user_input}
            Wellness Context: {wellness_context}
            
            Provide:
            1. Emotional validation and support
            2. Gentle processing questions
            3. Stress management techniques
            4. Self-care suggestions
            5. Professional help recommendations if needed
            
            Use emojis and format nicely with compassionate, supportive guidance.
            Note: If serious mental health concerns are expressed, gently suggest professional support.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.4,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Emotional wellness failed: {e}")
            return "💙 I'm here to support your emotional wellness! How are you feeling today?"
    
    async def _handle_general_journaling(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle general journaling queries."""
        try:
            journal_context = self._build_journal_context(context, "general")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📖 I'm here to support your journaling journey! What would you like to explore?"
            
            prompt = f"""
            As a journaling companion, provide helpful support for the user's personal reflection needs.
            
            User Request: {user_input}
            Journal Context: {journal_context}
            
            Provide relevant journaling guidance, prompts, and support based on their question.
            Use emojis and format nicely with gentle, encouraging information.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.4,
                max_tokens=800
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"General journaling failed: {e}")
            return "📖 I'm here to support your journaling journey! What would you like to explore?"
    
    def _build_journal_context(self, context: Dict[str, Any], journal_type: str) -> str:
        """Build journaling context from available knowledge."""
        context_parts = []
        
        # Add agent preferences (journaling preferences from knowledge base)
        if "agent_preferences" in context and context["agent_preferences"]:
            prefs = context["agent_preferences"]
            if isinstance(prefs, dict):
                journal_prefs = {k: v for k, v in prefs.items() if any(term in k.lower() for term in ["journal", "mood", "goal", "habit", "reflection", "wellness"])}
                if journal_prefs:
                    context_parts.append(f"Journaling preferences: {journal_prefs}")
        
        # Add context summary
        if "context_summary" in context and context["context_summary"]:
            context_parts.append(f"Previous journaling context: {context['context_summary']}")
        
        return " | ".join(context_parts) if context_parts else f"No specific {journal_type} context available"