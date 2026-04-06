"""
Health Agent - Specialized agent for health, wellness, and nutrition management.
Enhanced with Deep Agent patterns and human-in-the-loop capabilities.
"""

from typing import Dict, Any, List
import logging
import re

from ..base import BaseAgent, AgentType, AgentCapability, AgentState
from ..prompts import get_agent_prompt
from ...llm.service import get_llm_service
from ...llm.base import CompletionRequest, ChatMessage
from ...services.knowledge_base import get_knowledge_base_service
from ...services.interaction_recorder import get_interaction_recorder

logger = logging.getLogger(__name__)


class HealthAgent(BaseAgent):
    """
    Specialized agent for health, wellness, and nutrition management.
    
    Enhanced with Deep Agent patterns including:
    - Strategic health planning with context-aware analysis
    - Human-in-the-loop approval for significant health changes
    - File-based persistence of health data and meal plans
    - TODO management for health goals and habits
    - Intelligent delegation to specialized health sub-functions
    
    Capabilities:
    - Personalized meal planning with dietary restrictions
    - Health habit tracking and analysis
    - Wellness coaching with motivational support
    - Health goal setting and progress monitoring
    - Integration with external health APIs (with approval)
    """
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="preference_recording",
                description="Record and acknowledge user's health and dietary preferences with optional human approval for significant changes",
                parameters={"preference_types": ["dietary", "exercise", "sleep", "health_goals"], "approval_required": True}
            ),
            AgentCapability(
                name="meal_planning",
                description="Create comprehensive personalized meal plans with nutritional analysis and shopping lists",
                parameters={"dietary_restrictions": True, "nutrition_goals": True, "file_output": True}
            ),
            AgentCapability(
                name="habit_tracking",
                description="Advanced habit tracking with progress analytics and personalized recommendations",
                parameters={"habit_types": ["exercise", "sleep", "nutrition", "mood"], "analytics": True, "goal_setting": True}
            ),
            AgentCapability(
                name="wellness_coaching",
                description="AI-powered wellness coaching with personalized motivation and goal achievement strategies",
                parameters={"coaching_style": "adaptive", "goal_oriented": True, "progress_tracking": True}
            ),
            AgentCapability(
                name="health_data_analysis",
                description="Analyze health trends and provide insights with optional external data integration",
                parameters={"trend_analysis": True, "external_apis": True, "approval_for_integrations": True}
            ),
            AgentCapability(
                name="emergency_health_guidance",
                description="Provide immediate health guidance with automatic escalation protocols",
                parameters={"emergency_detection": True, "escalation_protocols": True, "human_approval": "critical"}
            )
        ]
        
        super().__init__(
            agent_id="health_specialized",
            agent_type=AgentType.HEALTH,
            capabilities=capabilities
        )
        
        # Initialize knowledge base for health-specific data
        self.knowledge_base = get_knowledge_base_service()
        logger.info("HealthAgent initialized with enhanced capabilities")

    def get_enhanced_tools(self):
        """
        Get the enhanced tool list for the Health Agent including human-in-the-loop tools.
        
        Returns:
            List of tools available to this agent including:
            - Core health tools (meal planning, habit tracking, etc.)
            - Human approval tools for significant health changes
            - File management tools for health data persistence
            - TODO tools for health goal management
        """
        try:
            from ..human_loop_tools import create_human_loop_tools
            from ..file_tools import create_file_tools
            from ..todo_tools import create_todo_tools
            from ..think_tools import create_think_tools
            
            tools = []
            
            # Human-in-the-loop tools for critical health decisions
            tools.extend(create_human_loop_tools())
            
            # File management for health data persistence
            tools.extend(create_file_tools())
            
            # Management for health goals and habits
            tools.extend(create_todo_tools())
            
            # Strategic thinking tools for health planning
            tools.extend(create_think_tools())
            
            return tools
        except ImportError as e:
            logger.warning("Some enhanced tools not available: %s", e)
            return []

    def get_enhanced_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """
        Get an enhanced system prompt that incorporates Deep Agent patterns.
        
        Args:
            context: Current conversation and state context
            
        Returns:
            Enhanced system prompt with health-specific instructions and deep agent capabilities
        """
        base_prompt = get_agent_prompt(self.agent_type)
        
        context_summary = ""
        if context:
            recent_interactions = len(context.get("relevant_interactions", []))
            user_preferences = len(context.get("user_preferences", []))
            context_summary = f"""

## Current Context
- Recent health interactions: {recent_interactions}
- Known user preferences: {user_preferences}
- Session type: Health & Wellness Management"""

        enhanced_prompt = f"""{base_prompt}

## Deep Agent Enhancement - Health Specialist

You are an advanced health agent with sophisticated capabilities:

### Core Principles
1. **Strategic Health Planning**: Always consider long-term health implications and create comprehensive plans
2. **Human-in-the-Loop**: Request approval for significant health changes, dietary modifications, or exercise programs
3. **Context Preservation**: Store all health data, meal plans, and tracking information in files for continuity
4. **Goal-Oriented Thinking**: Break complex health goals into manageable TODOs with clear success metrics

### Enhanced Capabilities
- **Comprehensive Meal Planning**: Create detailed meal plans with nutritional analysis, shopping lists, and prep instructions
- **Advanced Habit Tracking**: Monitor progress across multiple health dimensions with trend analysis
- **Personalized Coaching**: Provide motivational support adapted to user's personality and goals
- **Emergency Response**: Recognize health emergencies and escalate appropriately
- **Data Integration**: Suggest and implement (with approval) external health app/device integrations

### Decision Framework
For ANY significant health recommendation:
1. **Assess Impact**: Evaluate potential health implications
2. **Consider Context**: Review user's medical history, preferences, and current goals
3. **Plan Strategically**: Create step-by-step implementation plan
4. **Seek Approval**: Use human approval tools for major changes
5. **Document Everything**: Store plans and progress in files
6. **Track Progress**: Create TODOs for monitoring and follow-up

### Critical Safety Protocols
- ALWAYS escalate potential health emergencies to human immediately
- NEVER provide specific medical diagnoses or treatment recommendations
- ALWAYS suggest consulting healthcare professionals for serious concerns
- Request approval before recommending significant dietary or exercise changes
- Use human guidance tools when facing ambiguous health situations

### Tool Usage Priorities
1. **Think Tools**: Use for strategic health planning and decision-making
2. **Human Loop Tools**: Critical for approval of significant health changes
3. **Health Tools**: Core functionality for meal planning, tracking, coaching
4. **File Tools**: Essential for storing health data, plans, and progress
5. **TODO Tools**: Vital for breaking down health goals into actionable steps{context_summary}

Remember: Your role is to be a trusted health partner that combines AI insights with human wisdom for optimal health outcomes.
"""
        
        return enhanced_prompt

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute health-related requests with contextual awareness."""
        try:
            user_input = state.get("user_input", "")
            state_context = state.get("context", {}) if isinstance(state, dict) else {}
            logger.info(f"HealthAgent executing request: {user_input}")
            
            # Get relevant context from knowledge base
            context = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="health",
                max_results=10
            )
            merged_context = self._merge_with_routing_context(context, state_context)
            
            logger.info(f"Retrieved context with keys: {list(merged_context.keys())}")
            logger.info(f"Context details: {merged_context}")
            
            # Determine intent more intelligently
            intent = self._classify_user_intent(user_input)
            logger.info(f"Classified intent as: {intent}")
            
            # Route to appropriate handler based on intent
            if intent == "preference_sharing":
                response = await self._handle_preference_sharing(user_input, merged_context)
            elif intent == "meal_planning":
                response = await self._handle_meal_planning(user_input, merged_context)
            elif intent == "habit_tracking":
                response = await self._handle_habit_tracking(user_input, merged_context)
            else:
                response = await self._handle_general_health_query(user_input, merged_context)
            
            # Create pending interaction for user approval (instead of auto-recording)
            recorder = get_interaction_recorder()
            if recorder:
                # Extract knowledge sources from context for transparency
                knowledge_sources = self._extract_knowledge_sources(merged_context)
                
                interaction_id = await recorder.create_pending_interaction(
                    user_input=user_input,
                    agent_response=response,
                    agent_type="health",
                    context=merged_context,
                    knowledge_sources=knowledge_sources
                )
                
                if interaction_id:
                    logger.info("Created pending interaction %s for user approval", interaction_id)
                    # Add approval request info to response
                    response += f"\n\n---\n**🔍 Review Needed**: This response contains health recommendations. Please review and approve if helpful for future personalization."
            
            # IMPORTANT: Do NOT auto-extract preferences without user approval
            # This prevents the knowledge base pollution issue you identified
            logger.info("Health response generated - waiting for user approval before storing preferences")
            
            # Extract knowledge sources for transparency
            knowledge_sources = self._extract_knowledge_sources(merged_context)
            
            return {
                "response": response,
                "reasoning": {
                    "agent_type": "health",
                    "intent": intent,
                    "context_used": len(merged_context.get("relevant_interactions", [])) + len(merged_context.get("user_preferences", [])),
                    "coach_style": (merged_context.get("coach_profile") or {}).get("style"),
                    "specialized_handling": True,
                    "knowledge_sources": knowledge_sources
                }
            }
            
        except Exception as e:
            logger.error(f"Health agent execution failed: {e}")
            return {
                "response": "I apologize, but I encountered an issue while processing your health request. Please try again.",
                "reasoning": {
                    "error": str(e), 
                    "agent_type": "health",
                    "knowledge_sources": []
                }
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
                max_tokens=520
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
            coach_profile = context.get("coach_profile", {}) if isinstance(context, dict) else {}
            coach_style = coach_profile.get("style", "Direct")
            coach_directive = coach_profile.get("directive", "Be clear and practical.")
            
            health_prompt = f"""
            You are a knowledgeable health and wellness coach. Consider this context about the user:

            Context: {context_summary}
            Coach style: {coach_style}
            Coach directive: {coach_directive}

            User Query: {user_input}

            Provide helpful, personalized health advice that takes into account their background and previous interactions.
            Be supportive and practical in your suggestions.
            """
            
            llm_service = await get_llm_service()
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=health_prompt)],
                temperature=0.45,
                max_tokens=420
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"General health query failed: {e}")
            return "I'm here to help with your health and wellness goals. How can I assist you today?"

    def _extract_knowledge_sources(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract knowledge base sources used for transparency."""
        sources = []
        
        if not context:
            return sources
        
        # Extract from relevant interactions
        for interaction in context.get('relevant_interactions', []):
            sources.append({
                "type": "Previous Interaction",
                "content": interaction.get('content', '')[:200] + "..." if len(interaction.get('content', '')) > 200 else interaction.get('content', ''),
                "similarity": round(interaction.get('similarity', 0), 2),
                "created_at": interaction.get('created_at', ''),
                "metadata": interaction.get('metadata', {})
            })
        
        # Extract from user preferences  
        for pref in context.get('user_preferences', []):
            sources.append({
                "type": "User Preference",
                "content": pref.get('content', '')[:200] + "..." if len(pref.get('content', '')) > 200 else pref.get('content', ''),
                "category": pref.get('category', ''),
                "similarity": round(pref.get('similarity', 0), 2),
                "metadata": pref.get('metadata', {})
            })
        
        # Extract from patterns and insights
        for pattern in context.get('patterns_and_insights', []):
            sources.append({
                "type": "Pattern/Insight", 
                "content": pattern.get('content', '')[:200] + "..." if len(pattern.get('content', '')) > 200 else pattern.get('content', ''),
                "similarity": round(pattern.get('similarity', 0), 2),
                "metadata": pattern.get('metadata', {})
            })
        
        # Add agent preferences if they exist
        agent_prefs = context.get('agent_preferences', {})
        if agent_prefs:
            sources.append({
                "type": "Agent Configuration",
                "content": f"Dietary preferences: {agent_prefs.get('dietary_preferences', [])}, Health metrics: {agent_prefs.get('health_metrics', [])}",
                "similarity": 1.0,
                "metadata": {"configured_preferences": agent_prefs}
            })
        
        return sources

    def _merge_with_routing_context(self, kb_context: Dict[str, Any], state_context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge orchestrator routing context with health KB context."""
        merged_context = dict(kb_context or {})
        if not isinstance(state_context, dict):
            return merged_context

        for key in ("profile_snapshot", "coach_profile", "intent_blueprint", "knowledge_context_summary"):
            value = state_context.get(key)
            if value:
                merged_context[key] = value

        return merged_context