"""
Scheduling Agent - Specialized agent for calendar management, scheduling, and time optimization.
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


class SchedulingAgent(BaseAgent):
    """Specialized agent for calendar management, scheduling, and time optimization."""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="calendar_management",
                description="Manage appointments, meetings, and calendar events",
                parameters={"event_scheduling": True, "conflict_resolution": True}
            ),
            AgentCapability(
                name="time_optimization",
                description="Optimize daily schedules and time allocation",
                parameters={"schedule_analysis": True, "time_blocking": True}
            ),
            AgentCapability(
                name="appointment_booking",
                description="Book and manage appointments with proper time allocation",
                parameters={"availability_checking": True, "reminder_setting": True}
            ),
            AgentCapability(
                name="schedule_analytics",
                description="Analyze time usage and provide schedule optimization insights",
                parameters={"time_tracking": True, "efficiency_analysis": True}
            )
        ]
        
        super().__init__(
            agent_id=f"scheduling_{uuid.uuid4().hex[:8]}",
            agent_type=AgentType.SCHEDULING,
            capabilities=capabilities,
            system_prompt=get_agent_prompt(AgentType.SCHEDULING)
        )
        
        self.knowledge_base = get_knowledge_base_service()
    
    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute scheduling-related requests with contextual knowledge."""
        try:
            user_input = state.get("user_input", "")
            logger.info(f"SchedulingAgent processing: {user_input}")
            
            # Get contextual knowledge from knowledge base
            contextual_knowledge = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="scheduling",
                max_results=10
            )
            
            # Determine scheduling task type
            if any(keyword in user_input.lower() for keyword in ["schedule", "calendar", "appointment", "meeting"]):
                response = await self._handle_scheduling(user_input, contextual_knowledge)
            elif any(keyword in user_input.lower() for keyword in ["time block", "optimize", "time management"]):
                response = await self._handle_time_optimization(user_input, contextual_knowledge)
            elif any(keyword in user_input.lower() for keyword in ["book", "booking", "available", "availability"]):
                response = await self._handle_appointment_booking(user_input, contextual_knowledge)
            else:
                response = await self._handle_general_scheduling(user_input, contextual_knowledge)
            
            # Intelligently record interaction if valuable
            recorder = get_interaction_recorder()
            await recorder.record_if_valuable(
                user_input=user_input,
                agent_response=response,
                agent_type="scheduling"
            )
            
            return {
                "response": response,
                "reasoning": {
                    "agent_type": "scheduling",
                    "context_used": len(contextual_knowledge.get("relevant_interactions", [])),
                    "specialized_handling": True
                }
            }
            
        except Exception as e:
            logger.error(f"SchedulingAgent execution failed: {e}")
            return {
                "response": "I'm having trouble with scheduling right now. Please try again later.",
                "reasoning": {"error": str(e), "agent_type": "scheduling"}
            }
    
    async def _handle_scheduling(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle general scheduling requests."""
        try:
            schedule_context = self._build_schedule_context(context, "general")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📅 I'd be happy to help with your scheduling! What would you like to schedule?"
            
            prompt = f"""
            As a scheduling assistant, help the user with their calendar and scheduling needs.
            
            User Request: {user_input}
            Schedule Context: {schedule_context}
            
            Provide:
            1. Specific scheduling recommendations
            2. Time slot suggestions based on their preferences
            3. Calendar organization tips
            4. Conflict resolution if needed
            5. Follow-up reminders
            
            Use emojis and format nicely with clear time recommendations.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=800
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Scheduling failed: {e}")
            return "📅 I'd be happy to help you with scheduling! What specific appointment or event would you like to schedule?"
    
    async def _handle_time_optimization(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle time optimization and time blocking requests."""
        try:
            time_context = self._build_schedule_context(context, "optimization")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "⏰ I'd be happy to help optimize your time! What areas of your schedule would you like to improve?"
            
            prompt = f"""
            As a time management expert, help the user optimize their schedule and time usage.
            
            User Request: {user_input}
            Time Management Context: {time_context}
            
            Provide:
            1. Time blocking strategies
            2. Schedule optimization recommendations
            3. Productivity time slot identification
            4. Break and rest period suggestions
            5. Weekly schedule template
            
            Use emojis and format nicely with actionable time management advice.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Time optimization failed: {e}")
            return "⏰ I'd be happy to help you optimize your time! What specific areas of your schedule would you like to improve?"
    
    async def _handle_appointment_booking(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle appointment booking requests."""
        try:
            booking_context = self._build_schedule_context(context, "booking")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📞 I'd be happy to help you book appointments! What type of appointment do you need to schedule?"
            
            prompt = f"""
            As a scheduling assistant, help the user with appointment booking and management.
            
            User Request: {user_input}
            Booking Context: {booking_context}
            
            Provide:
            1. Appointment booking guidance
            2. Optimal time slot recommendations
            3. Preparation checklist for the appointment
            4. Reminder and follow-up suggestions
            5. Calendar integration tips
            
            Use emojis and format nicely with practical booking advice.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=800
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Appointment booking failed: {e}")
            return "📞 I'd be happy to help you book appointments! What type of appointment do you need to schedule?"
    
    async def _handle_general_scheduling(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle general scheduling queries."""
        try:
            schedule_context = self._build_schedule_context(context, "general")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📅 I'm here to help with your scheduling needs! What would you like assistance with?"
            
            prompt = f"""
            As a scheduling assistant, provide helpful advice for the user's scheduling question.
            
            User Request: {user_input}
            Schedule Context: {schedule_context}
            
            Provide relevant scheduling advice, tips, and recommendations based on their question.
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
            logger.error(f"General scheduling failed: {e}")
            return "📅 I'm here to help with your scheduling needs! What would you like assistance with?"
    
    def _build_schedule_context(self, context: Dict[str, Any], schedule_type: str) -> str:
        """Build scheduling context from available knowledge."""
        context_parts = []
        
        # Add agent preferences (scheduling preferences from knowledge base)
        if "agent_preferences" in context and context["agent_preferences"]:
            prefs = context["agent_preferences"]
            if isinstance(prefs, dict):
                schedule_prefs = {k: v for k, v in prefs.items() if any(term in k.lower() for term in ["schedule", "time", "work", "meeting", "appointment"])}
                if schedule_prefs:
                    context_parts.append(f"Scheduling preferences: {schedule_prefs}")
        
        # Add context summary
        if "context_summary" in context and context["context_summary"]:
            context_parts.append(f"Previous scheduling context: {context['context_summary']}")
        
        return " | ".join(context_parts) if context_parts else f"No specific {schedule_type} context available"