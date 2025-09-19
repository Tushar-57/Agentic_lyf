"""
Productivity Agent - Specialized agent for productivity, task management, and goal tracking.
"""

import logging
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
            logger.info(f"ProductivityAgent processing: {user_input}")
            
            # Get contextual knowledge from knowledge base
            contextual_knowledge = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="productivity",
                max_results=10
            )
            
            # Determine productivity task type
            if any(keyword in user_input.lower() for keyword in ["task", "todo", "organize", "project"]):
                response = await self._handle_task_management(user_input, contextual_knowledge)
            elif any(keyword in user_input.lower() for keyword in ["goal", "objective", "target", "achieve"]):
                response = await self._handle_goal_setting(user_input, contextual_knowledge)
            elif any(keyword in user_input.lower() for keyword in ["time", "schedule", "productivity", "focus"]):
                response = await self._handle_time_management(user_input, contextual_knowledge)
            else:
                response = await self._handle_general_productivity(user_input, contextual_knowledge)
            
            # Intelligently record interaction if valuable
            recorder = get_interaction_recorder()
            await recorder.record_if_valuable(
                user_input=user_input,
                agent_response=response,
                agent_type="productivity"
            )
            
            return {
                "response": response,
                "reasoning": {
                    "agent_type": "productivity",
                    "context_used": len(contextual_knowledge.get("relevant_interactions", [])),
                    "specialized_handling": True
                }
            }
            
        except Exception as e:
            logger.error(f"ProductivityAgent execution failed: {e}")
            return {
                "response": "I'm having trouble with productivity assistance right now. Please try again later.",
                "reasoning": {"error": str(e), "agent_type": "productivity"}
            }
    
    async def _handle_task_management(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle task management requests."""
        try:
            task_context = self._build_productivity_context(context, "tasks")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📋 I'd be happy to help you manage your tasks! What tasks would you like to organize?"
            
            prompt = f"""
            As a productivity coach, help the user with task management and organization.
            
            User Request: {user_input}
            Task Context: {task_context}
            
            Provide:
            1. Task organization strategies (using Eisenhower Matrix or similar)
            2. Priority recommendations based on their work style
            3. Deadline and milestone suggestions
            4. Task breakdown for complex projects
            5. Tracking and review methods
            
            Use emojis and format nicely with actionable task management advice.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Task management failed: {e}")
            return "📋 I'd be happy to help you manage your tasks! What specific tasks would you like to organize?"
    
    async def _handle_goal_setting(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle goal setting and tracking requests."""
        try:
            goal_context = self._build_productivity_context(context, "goals")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "🎯 I'd love to help you set and achieve your goals! What goals are you working on?"
            
            prompt = f"""
            As a goal-setting expert, help the user create and track meaningful goals.
            
            User Request: {user_input}
            Goal Context: {goal_context}
            
            Provide:
            1. SMART goal framework application
            2. Goal breakdown into actionable steps
            3. Progress tracking recommendations
            4. Motivation and accountability strategies
            5. Timeline and milestone suggestions
            
            Use emojis and format nicely with structured goal-setting guidance.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Goal setting failed: {e}")
            return "🎯 I'd love to help you set and achieve your goals! What specific goals would you like to work on?"
    
    async def _handle_time_management(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle time management and productivity optimization."""
        try:
            time_context = self._build_productivity_context(context, "time")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "⏰ I'd be happy to help optimize your time! What time management challenges are you facing?"
            
            prompt = f"""
            As a time management expert, help the user optimize their productivity and time usage.
            
            User Request: {user_input}
            Time Management Context: {time_context}
            
            Provide:
            1. Time blocking strategies for their schedule
            2. Productivity techniques (Pomodoro, deep work, etc.)
            3. Focus and concentration tips
            4. Energy management recommendations
            5. Work-life balance suggestions
            
            Use emojis and format nicely with practical time management advice.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Time management failed: {e}")
            return "⏰ I'd be happy to help optimize your time! What specific time management areas would you like to improve?"
    
    async def _handle_general_productivity(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle general productivity queries."""
        try:
            productivity_context = self._build_productivity_context(context, "general")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "🚀 I'm here to boost your productivity! What can I help you with?"
            
            prompt = f"""
            As a productivity expert, provide helpful advice for the user's productivity question.
            
            User Request: {user_input}
            Productivity Context: {productivity_context}
            
            Provide relevant productivity advice, tips, and recommendations based on their question.
            Use emojis and format nicely with clear, actionable information.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=800
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"General productivity failed: {e}")
            return "🚀 I'm here to boost your productivity! What specific area would you like help with?"
    
    def _build_productivity_context(self, context: Dict[str, Any], productivity_type: str) -> str:
        """Build productivity context from available knowledge."""
        context_parts = []
        
        # Add agent preferences (productivity preferences from knowledge base)
        if "agent_preferences" in context and context["agent_preferences"]:
            prefs = context["agent_preferences"]
            if isinstance(prefs, dict):
                productivity_prefs = {k: v for k, v in prefs.items() if any(term in k.lower() for term in ["work", "task", "goal", "time", "productivity", "schedule"])}
                if productivity_prefs:
                    context_parts.append(f"Productivity preferences: {productivity_prefs}")
        
        # Add context summary
        if "context_summary" in context and context["context_summary"]:
            context_parts.append(f"Previous productivity context: {context['context_summary']}")
        
        return " | ".join(context_parts) if context_parts else f"No specific {productivity_type} context available"