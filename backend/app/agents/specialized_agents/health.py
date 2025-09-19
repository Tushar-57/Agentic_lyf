"""
Health Agent - Specialized agent for health, wellness, and nutrition management.
"""

import logging
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

logger = logging.getLogger(__name__)


class HealthAgent(BaseAgent):
    """Specialized agent for health, wellness, and nutrition management."""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="preference_recording",
                description="Record and acknowledge user's health and dietary preferences",
                parameters={"preference_types": ["dietary", "exercise", "sleep", "health_goals"]}
            ),
            AgentCapability(
                name="meal_planning",
                description="Create personalized meal plans based on dietary preferences and goals",
                parameters={"dietary_restrictions": True, "nutrition_goals": True}
            ),
            AgentCapability(
                name="habit_tracking",
                description="Track and analyze health habits and routines",
                parameters={"habit_types": ["exercise", "sleep", "nutrition", "mood"]}
            ),
            AgentCapability(
                name="wellness_coaching",
                description="Provide personalized wellness advice and motivation",
                parameters={"coaching_style": "supportive", "goal_oriented": True}
            )
        ]
        
        super().__init__(
            agent_id="health_specialized",
            agent_type=AgentType.HEALTH,
            capabilities=capabilities,
            system_prompt=get_agent_prompt(AgentType.HEALTH)
        )
        self.knowledge_base = get_knowledge_base_service()

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute health-related requests with contextual awareness."""
        try:
            user_input = state.get("user_input", "")
            logger.info(f"HealthAgent executing request: {user_input}")
            
            # Get relevant context from knowledge base
            context = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="health",
                max_results=10
            )
            
            logger.info(f"Retrieved context with keys: {list(context.keys())}")
            logger.info(f"Context details: {context}")
            
            # Determine intent more intelligently
            intent = self._classify_user_intent(user_input)
            logger.info(f"Classified intent as: {intent}")
            
            # Route to appropriate handler based on intent
            if intent == "preference_sharing":
                response = await self._handle_preference_sharing(user_input, context)
            elif intent == "meal_planning":
                response = await self._handle_meal_planning(user_input, context)
            elif intent == "habit_tracking":
                response = await self._handle_habit_tracking(user_input, context)
            else:
                response = await self._handle_general_health_query(user_input, context)
            
            # Intelligently record interaction if valuable
            recorder = get_interaction_recorder()
            await recorder.record_if_valuable(
                user_input=user_input,
                agent_response=response,
                agent_type="health"
            )
            
            # Extract and store any new preferences
            await self.knowledge_base.extract_and_store_preferences(
                user_input=user_input,
                agent_type="health",
                agent_response=response
            )
            
            return {
                "response": response,
                "reasoning": {
                    "agent_type": "health",
                    "intent": intent,
                    "context_used": len(context.get("relevant_interactions", [])) + len(context.get("user_preferences", [])),
                    "specialized_handling": True
                }
            }
            
        except Exception as e:
            logger.error(f"Health agent execution failed: {e}")
            return {
                "response": "I apologize, but I encountered an issue while processing your health request. Please try again.",
                "reasoning": {"error": str(e), "agent_type": "health"}
            }

    def _classify_user_intent(self, user_input: str) -> str:
        """Classify user intent more intelligently to avoid over-triggering meal planning."""
        input_lower = user_input.lower()
        
        # Check for preference sharing patterns (I like, I prefer, I enjoy, etc.)
        preference_patterns = [
            r"i (like|love|enjoy|prefer|hate|dislike)",
            r"(my favorite|i usually|i normally|i typically)",
            r"i'm (vegetarian|vegan|allergic to)",
            r"(good|great|excellent|amazing).*source",
            r"i don't eat",
            r"i avoid"
        ]
        
        for pattern in preference_patterns:
            if re.search(pattern, input_lower):
                return "preference_sharing"
        
        # Check for explicit meal planning requests  
        explicit_meal_planning = [
            r"(plan|create|make|suggest|give me).*meal",
            r"(meal plan|weekly menu|daily menu)",
            r"what should i (eat|cook|prepare)",
            r"recipe for",
            r"help.*plan.*food",
            r"breakfast.*ideas",
            r"lunch.*suggestions",
            r"dinner.*recommendations"
        ]
        
        for pattern in explicit_meal_planning:
            if re.search(pattern, input_lower):
                return "meal_planning"
        
        # Check for habit tracking
        habit_tracking_patterns = [
            r"track.*habit",
            r"(log|record).*exercise",
            r"(monitor|track).*(sleep|workout|steps|water)",
            r"habit.*tracking",
            r"daily.*routine"
        ]
        
        for pattern in habit_tracking_patterns:
            if re.search(pattern, input_lower):
                return "habit_tracking"
        
        # Default to general health query for questions or concerns
        return "general_health"

    async def _handle_preference_sharing(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle when user is sharing preferences, not requesting meal plans."""
        try:
            # Build context about current preferences
            existing_prefs = self._get_existing_preferences(context)
            
            preference_prompt = f"""
            The user is sharing their dietary or health preferences with you. This is not a request for meal planning, 
            but rather them telling you about their likes, dislikes, or dietary habits.

            Existing user preferences: {existing_prefs}
            
            User's statement: {user_input}

            Respond by:
            1. Acknowledging their preference warmly
            2. Showing you understand and have noted it
            3. Briefly mentioning how this fits with healthy eating (if relevant)
            4. Asking if they'd like any specific help or information related to this preference
            
            Keep it conversational and supportive. Don't immediately jump into meal planning unless they specifically ask for it.
            """
            
            llm_service = await get_llm_service()
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=preference_prompt)],
                temperature=0.7,
                max_tokens=400
            )
            
            response = await llm_service.chat_completion(request)
            logger.info(f"Generated preference acknowledgment: {response.content[:200]}...")
            return response.content
            
        except Exception as e:
            logger.error(f"Preference sharing handling failed: {e}")
            return "Thank you for sharing that with me! I've noted your preference and it will help me provide better personalized advice in the future."

    async def _handle_meal_planning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle explicit meal planning requests with personalized context."""
        try:
            logger.info(f"Handling meal planning with context: {context}")
            
            # Build context-aware prompt
            context_info = self._build_meal_planning_context(context)
            logger.info(f"Built context info: {context_info}")
            
            meal_planning_prompt = f"""
            You are a health and nutrition expert helping with meal planning. The user has specifically requested meal planning assistance.
            
            Use the following context about the user:
            {context_info}

            User Request: {user_input}

            Based on the user's preferences and dietary requirements, provide a detailed and personalized response that includes:
            1. Specific meal suggestions that match their dietary preferences
            2. Consideration of their health goals and restrictions
            3. Practical preparation tips
            4. Nutritional benefits

            Make the response actionable and tailored to their specific needs.
            """
            
            llm_service = await get_llm_service()
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=meal_planning_prompt)],
                temperature=0.7,
                max_tokens=1000
            )
            
            response = await llm_service.chat_completion(request)
            logger.info(f"Generated meal planning response: {response.content[:200]}...")
            return response.content
            
        except Exception as e:
            logger.error(f"Meal planning failed: {e}")
            return "I'd be happy to help with meal planning! Could you tell me about your dietary preferences, any restrictions, and your health goals?"

    def _get_existing_preferences(self, context: Dict[str, Any]) -> str:
        """Extract existing preferences from context for reference."""
        prefs = []
        
        # Get agent preferences
        agent_prefs = context.get("agent_preferences", {})
        if agent_prefs.get("dietary_preferences"):
            prefs.append(f"Dietary: {', '.join(agent_prefs['dietary_preferences'])}")
        
        # Get user preferences
        user_prefs = context.get("user_preferences", [])
        for pref in user_prefs[:3]:  # Top 3
            prefs.append(f"• {pref['content']}")
        
        if not prefs:
            return "No existing preferences recorded"
        
        return "\n".join(prefs)

    def _build_meal_planning_context(self, context: Dict[str, Any]) -> str:
        """Build meal planning context from user's knowledge base."""
        context_parts = []
        
        # Add dietary preferences
        health_prefs = context.get("agent_preferences", {})
        if health_prefs:
            dietary_info = health_prefs.get("dietary_preferences", [])
            if dietary_info:
                context_parts.append(f"Dietary Preferences: {', '.join(dietary_info)}")
            
            health_goals = health_prefs.get("exercise_goals", "")
            if health_goals:
                context_parts.append(f"Health Goals: {health_goals}")

        # Add relevant user preferences
        user_prefs = context.get("user_preferences", [])
        for pref in user_prefs[:3]:  # Top 3 most relevant
            context_parts.append(f"User Preference: {pref['content']}")

        # Add recent relevant interactions
        interactions = context.get("relevant_interactions", [])
        if interactions:
            recent_interaction = interactions[0]
            context_parts.append(f"Recent Context: {recent_interaction['content'][:200]}...")

        if not context_parts:
            return "No specific dietary preferences or health information available. Please ask the user for their preferences."
        
        return "\n".join(context_parts)

    async def _handle_habit_tracking(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle habit tracking requests."""
        # Implementation for habit tracking
        return "I'll help you track your health habits. Based on your request, I can assist with monitoring exercise, sleep, nutrition, or mood patterns."

    async def _handle_general_health_query(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle general health queries with context."""
        try:
            context_summary = context.get("context_summary", "")
            
            health_prompt = f"""
            You are a knowledgeable health and wellness coach. Consider this context about the user:

            Context: {context_summary}

            User Query: {user_input}

            Provide helpful, personalized health advice that takes into account their background and previous interactions.
            Be supportive and practical in your suggestions.
            """
            
            llm_service = await get_llm_service()
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=health_prompt)],
                temperature=0.7,
                max_tokens=800
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"General health query failed: {e}")
            return "I'm here to help with your health and wellness goals. How can I assist you today?"