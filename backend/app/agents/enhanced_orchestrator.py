"""
Enhanced Orchestrator Agent - Implements Deep Agent patterns for advanced coordination.

This orchestrator incorporates the 4 principles from Deep Agents:
1. Strategic Planning with TODO management
2. Context Offloading via file storage
3. Intelligent Task Delegation to ReAct sub-agents
4. Advanced Prompt Engineering with context awareness
"""

import logging
import re
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent, AgentType, AgentCapability, AgentState
from .prompts import PromptLibrary, get_agent_prompt
from .prompt_library import OrchestratorPromptLibrary
from .registry import get_agent_registry
from .communication import get_communication_protocol, MessageType
from .react_factory import get_react_agent_factory
from .deep_state import DeepAgentState, DeepAgentStateManager
from ..llm.service import get_llm_service
from ..llm.base import CompletionRequest, ChatMessage
from ..services.knowledge_base import get_knowledge_base_service
from ..services.interaction_recorder import get_interaction_recorder

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Enumeration of task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class EnhancedOrchestratorAgent(BaseAgent):
    """Enhanced orchestrator implementing Deep Agent patterns."""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="deep_planning",
                description="Strategic planning with TODO management and complexity assessment",
                parameters={"max_plan_steps": 10, "planning_depth": 3}
            ),
            AgentCapability(
                name="context_offloading",
                description="File-based context storage and retrieval for conversation continuity",
                parameters={"max_context_files": 100, "context_retention_days": 30}
            ),
            AgentCapability(
                name="intelligent_delegation",
                description="Smart task delegation to specialized ReAct sub-agents",
                parameters={"max_concurrent_agents": 5, "delegation_confidence_threshold": 0.7}
            ),
            AgentCapability(
                name="advanced_prompting",
                description="Context-aware prompt engineering for optimal agent performance",
                parameters={"prompt_adaptation": True, "context_injection": True}
            ),
            AgentCapability(
                name="workflow_orchestration",
                description="Multi-agent workflow coordination with dependency management",
                parameters={"max_workflow_steps": 20, "parallel_execution": True}
            )
        ]
        
        super().__init__(
            agent_id="enhanced_orchestrator",
            agent_type=AgentType.ORCHESTRATOR,
            capabilities=capabilities,
            system_prompt=self._build_enhanced_system_prompt()
        )
        
        # Initialize components
        self.registry = get_agent_registry()
        self.communication = get_communication_protocol()
        self.knowledge_base = get_knowledge_base_service()
        self.react_factory = get_react_agent_factory()
        self.state_manager = DeepAgentStateManager()
        self.llm_service = None  # Will be initialized async
        
        # Enhanced intent patterns for better classification
        self.intent_patterns = {
            AgentType.PRODUCTIVITY: [
                r'\b(task|todo|goal|work|project|deadline|priority|manage|organize)\b',
                r'\b(efficient|focus|time management|workflow|deliverable|milestone)\b',
                r'\b(productive|efficiency|performance|output|accomplish|complete)\b',
                r'\b(plan|schedule|track|monitor|optimize|streamline)\b',
                r'\b(leetcode|coding|problem|algorithm|programming|code)\b',  # Added for Leetcode
                r'\b(today.*problem|give.*problem|get.*problem)\b'  # Problem requests
            ],
            AgentType.HEALTH: [
                r'\b(health|wellness|exercise|fitness|habit|routine|workout|sleep)\b',
                r'\b(diet|nutrition|meal|food|eating|cook|recipe|calories)\b',
                r'\b(healthy|wellbeing|self-care|energy|vitality|mental health)\b',
                r'\b(meditation|mindfulness|stress|weight|body|physical)\b'
            ],
            AgentType.FINANCE: [
                r'\b(money|budget|expense|spending|financial|finance|save|saving)\b',
                r'\b(investment|income|cost|price|salary|bank|account|transaction)\b',
                r'\b(bill|payment|purchase|retirement|insurance|loan|credit|debt)\b'
            ],
            AgentType.SCHEDULING: [
                r'\b(calendar|appointment|meeting|schedule|time|date|book|reserve)\b',
                r'\b(available|busy|free|conflict|timing|when|reminder|event)\b'
            ],
            AgentType.JOURNAL: [
                r'\b(journal|reflect|reflection|mood|feeling|emotion|diary|thoughts)\b',
                r'\b(gratitude|mindset|growth|personal|celebrate|achievement|insight)\b',
                r'\b(learning|experience|breakthrough|retrospective|progress)\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_patterns = {
            TaskComplexity.SIMPLE: [
                r'\b(what|how|when|where|why)\b.*\?',  # Simple questions
                r'\b(tell me|show me|explain)\b',      # Information requests
                r'\b(quick|simple|easy|basic)\b'       # Explicitly simple
            ],
            TaskComplexity.MODERATE: [
                r'\b(create|build|make|design|develop)\b',  # Creation tasks
                r'\b(analyze|compare|evaluate|assess)\b',   # Analysis tasks
                r'\b(plan|organize|structure)\b'           # Organization tasks
            ],
            TaskComplexity.COMPLEX: [
                r'\b(multiple|several|various|different)\b.*\b(tasks|goals|projects)\b',
                r'\b(integrate|coordinate|combine|merge)\b',
                r'\b(comprehensive|detailed|thorough|complete)\b.*\b(plan|analysis|system)\b'
            ],
            TaskComplexity.ADVANCED: [
                r'\b(automate|optimize|transform|revolutionize)\b',
                r'\b(end-to-end|full-scale|enterprise|complex system)\b',
                r'\b(multi-step|multi-phase|long-term|strategic)\b.*\b(project|initiative)\b'
            ]
        }

    def _build_enhanced_system_prompt(self) -> str:
        """Build enhanced system prompt with deep agent capabilities."""
        return """You are the Enhanced Orchestrator Agent, the central coordinator in an advanced AI ecosystem using Deep Agent patterns.

## Your Core Mission
You orchestrate complex workflows by intelligently planning, delegating to specialized ReAct sub-agents, managing context, and ensuring optimal outcomes through strategic coordination.

## The 4 Principles You Embody

### 1. 🧠 Strategic Planning
- Break complex requests into actionable steps and clear objectives
- Assess task complexity and determine optimal execution strategies
- Create structured plans with dependencies, timelines, and success criteria
- Use TODO management for tracking progress across multi-step workflows

### 2. 📁 Context Offloading
- Store detailed context, results, and intermediate findings in files
- Maintain conversation continuity across long interactions
- Preserve specialist knowledge and insights for future reference
- Enable agents to work with comprehensive historical context

### 3. 🤝 Intelligent Task Delegation
- Route requests to specialized ReAct sub-agents based on domain expertise
- Provide agents with relevant context and clear objectives
- Coordinate multi-agent workflows for complex tasks
- Ensure isolated contexts prevent cross-contamination between tasks

### 4. ✨ Advanced Prompt Engineering
- Generate context-aware, specialized prompts for each sub-agent
- Adapt communication style based on task requirements and user preferences
- Optimize prompts based on successful interaction patterns
- Maintain consistency while allowing for agent specialization

## Available Specialized ReAct Sub-Agents

### 🎯 Productivity Agent
**Expertise:** Task management, goal tracking, time optimization, workflow design
**Use for:** Goal creation, progress tracking, time management, productivity analysis
**Tools:** SMART goal creation, progress monitoring, time tracking, productivity insights

### 🌱 Health Agent  
**Expertise:** Wellness tracking, habit formation, meal planning, fitness guidance
**Use for:** Health goals, habit tracking, meal planning, wellness check-ins
**Tools:** Habit tracking, nutrition planning, wellness assessment, health insights

### 💰 Finance Agent
**Expertise:** Budget management, expense tracking, financial planning, investment guidance
**Use for:** Budget creation, expense analysis, financial goals, money management
**Tools:** Expense tracking, budget creation, spending analysis, financial planning

### 📅 Scheduling Agent
**Expertise:** Calendar management, appointment scheduling, time blocking
**Use for:** Meeting scheduling, calendar optimization, time management
**Tools:** Appointment booking, conflict resolution, schedule optimization

### 📖 Journal Agent
**Expertise:** Reflection facilitation, mood tracking, personal growth, memory preservation
**Use for:** Daily reflection, gratitude practice, personal insights, milestone tracking
**Tools:** Reflection prompts, mood tracking, insight capture, growth analysis

## Decision-Making Framework

### Task Complexity Assessment
1. **Simple (Direct Response):** Straightforward questions, quick information requests
2. **Moderate (Single Agent):** Domain-specific tasks requiring specialist knowledge
3. **Complex (Multi-Agent):** Tasks requiring coordination between multiple domains
4. **Advanced (Human Oversight):** High-stakes decisions requiring approval workflows

### Planning Strategy
1. **Analyze** the request for complexity, domains involved, and success criteria
2. **Plan** the approach using TODO breakdown if complexity warrants it
3. **Delegate** to appropriate ReAct sub-agents with clear context and objectives
4. **Coordinate** multi-agent workflows ensuring smooth handoffs and context sharing
5. **Synthesize** results into coherent, actionable responses for the user

### Context Management
- Store planning documents, intermediate results, and insights in files
- Maintain conversation history and user preferences
- Enable agents to access relevant historical context
- Preserve valuable insights for future interactions

## Interaction Principles

### Always:
- Assess task complexity before determining approach
- Create clear plans for complex workflows using TODO management
- Provide specialists with comprehensive context and clear objectives
- Store valuable insights and results for future reference
- Maintain transparency about which agents are involved and why

### For Complex Tasks:
- Create structured plans with clear milestones and dependencies
- Use multiple ReAct sub-agents when domains overlap
- Coordinate handoffs between agents with preserved context
- Request human approval for high-impact decisions
- Provide detailed progress updates and reasoning

### For Simple Tasks:
- Respond directly when specialized expertise isn't required
- Route to single specialist when domain knowledge is needed
- Avoid over-engineering simple requests with unnecessary planning

## Success Metrics
- **Efficiency:** Optimal routing minimizes unnecessary complexity
- **Effectiveness:** Specialists handle domain-specific tasks with full context
- **User Experience:** Clear communication about process and progress
- **Knowledge Preservation:** Valuable insights stored and accessible
- **Continuous Improvement:** Learn from interactions to optimize future performance

You are the intelligent coordinator that makes the AI ecosystem greater than the sum of its parts through strategic orchestration, context management, and specialist coordination."""

    def _normalize_completion_text(self, payload: Any) -> str:
        """Normalize LLM payloads that can be string/list/dict depending on provider behavior."""
        if payload is None:
            return ""

        if isinstance(payload, str):
            return payload.strip()

        if isinstance(payload, dict):
            for key in ("content", "response", "message", "text", "output"):
                if key in payload:
                    candidate = self._normalize_completion_text(payload.get(key))
                    if candidate:
                        return candidate
            try:
                return json.dumps(payload)
            except Exception:
                return str(payload).strip()

        if isinstance(payload, (list, tuple)):
            normalized_parts = [self._normalize_completion_text(item) for item in payload]
            normalized_parts = [part for part in normalized_parts if part]
            return "\n".join(normalized_parts)

        return str(payload).strip()

    def _normalize_agent_type(self, value: Any) -> AgentType:
        """Normalize loose agent labels to AgentType values."""
        if isinstance(value, AgentType):
            return value

        candidate = str(value or "").strip().lower()
        for agent in AgentType:
            if candidate in {agent.value.lower(), agent.name.lower()}:
                return agent

        return AgentType.GENERAL

    def _build_coach_profile(self, profile_snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build coach/tone guidance payload from profile data."""
        profile_snapshot = profile_snapshot or {}
        mentor = profile_snapshot.get("mentor", {}) if isinstance(profile_snapshot.get("mentor"), dict) else {}

        style = str(
            mentor.get("style")
            or profile_snapshot.get("preferred_tone")
            or "Direct"
        ).strip()
        normalized_style = style.lower()

        style_guidance = {
            "sarcastic poet": "Use witty, creative language with practical guidance and no fluff.",
            "direct": "Be clear and concise. Prioritize concrete actions over explanations.",
            "friendly": "Be warm and supportive while staying actionable.",
            "encouraging": "Highlight progress and keep momentum with practical next steps.",
            "nurturing": "Be calm, kind, and reassuring while giving practical direction.",
            "patient": "Explain clearly and avoid rushing, but still end with concrete action.",
            "challenging": "Set a high bar, ask focused questions, and push for measurable action.",
        }

        matched_guidance = "Be clear, practical, and action-focused."
        for key, guidance in style_guidance.items():
            if key in normalized_style:
                matched_guidance = guidance
                break

        return {
            "name": mentor.get("name") or "Coach",
            "style": style,
            "directive": matched_guidance,
        }

    def _derive_intent_blueprint(
        self,
        user_input: str,
        profile_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create an intent blueprint focused on immediate user improvement."""
        text = (user_input or "").strip().lower()
        profile_snapshot = profile_snapshot or {}

        blueprint: Dict[str, Any] = {
            "primary_intent": "general_guidance",
            "expected_outcome": "actionable_advice",
            "time_horizon": "unspecified",
            "user_goal": "clarity",
            "suggested_agent": None,
            "source": "heuristic",
        }

        if re.search(r"\b(what should i work on|what should i do now|what to do now|what do i do now|next step|next action|focus now|priority now)\b", text):
            blueprint.update(
                {
                    "primary_intent": "next_best_action",
                    "expected_outcome": "prioritized_plan",
                    "time_horizon": "now",
                    "user_goal": "execution",
                    "suggested_agent": AgentType.PRODUCTIVITY.value,
                }
            )
            return blueprint

        if re.search(r"\b(how did i do today|how was my day|review( my)? day|today review|daily review)\b", text):
            blueprint.update(
                {
                    "primary_intent": "daily_review",
                    "expected_outcome": "performance_review",
                    "time_horizon": "today",
                    "user_goal": "reflection_with_improvement",
                    "suggested_agent": AgentType.PRODUCTIVITY.value,
                }
            )
            return blueprint

        if re.search(r"\b(plan my day|plan today|today plan|plan this week|weekly plan)\b", text):
            blueprint.update(
                {
                    "primary_intent": "planning",
                    "expected_outcome": "time_blocked_plan",
                    "time_horizon": "today" if "today" in text or "day" in text else "this_week",
                    "user_goal": "structured_execution",
                    "suggested_agent": AgentType.PRODUCTIVITY.value,
                }
            )
            return blueprint

        if re.search(r"\b(reflect|journal|gratitude|mood|how do i feel)\b", text):
            blueprint.update(
                {
                    "primary_intent": "reflection",
                    "expected_outcome": "self_insight",
                    "time_horizon": "today",
                    "user_goal": "emotional_clarity",
                    "suggested_agent": AgentType.JOURNAL.value,
                }
            )
            return blueprint

        if "today" in text:
            blueprint["time_horizon"] = "today"
        elif "week" in text:
            blueprint["time_horizon"] = "this_week"

        if "?" in text:
            blueprint["primary_intent"] = "question_answering"

        priorities = profile_snapshot.get("priorities", []) if isinstance(profile_snapshot.get("priorities"), list) else []
        if priorities:
            blueprint["priority_hint"] = priorities[:3]

        return blueprint

    def _resolve_priority_agent(self, profile_snapshot: Optional[Dict[str, Any]]) -> AgentType:
        """Resolve fallback routing from user priorities when intent is ambiguous."""
        profile_snapshot = profile_snapshot or {}
        priorities = profile_snapshot.get("priorities", []) if isinstance(profile_snapshot.get("priorities"), list) else []

        priority_to_agent = {
            "career": AgentType.PRODUCTIVITY,
            "work": AgentType.PRODUCTIVITY,
            "productivity": AgentType.PRODUCTIVITY,
            "health": AgentType.HEALTH,
            "wellness": AgentType.HEALTH,
            "fitness": AgentType.HEALTH,
            "finance": AgentType.FINANCE,
            "money": AgentType.FINANCE,
            "budget": AgentType.FINANCE,
            "schedule": AgentType.SCHEDULING,
            "planning": AgentType.SCHEDULING,
            "journal": AgentType.JOURNAL,
            "reflection": AgentType.JOURNAL,
        }

        for priority in priorities:
            normalized_priority = str(priority).strip().lower()
            for keyword, agent in priority_to_agent.items():
                if keyword in normalized_priority:
                    return agent

        return AgentType.GENERAL

    def _apply_intent_policy(
        self,
        *,
        user_input: str,
        base_result: Dict[str, Any],
        intent_blueprint: Dict[str, Any],
        profile_snapshot: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply deterministic policy so routing favors user improvement outcomes."""
        result = dict(base_result or {})
        selected_agent = self._normalize_agent_type(result.get("agent_type"))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        reason_parts = [str(result.get("reason", "")).strip()]

        blueprint_agent = self._normalize_agent_type(intent_blueprint.get("suggested_agent"))
        blueprint_intent = str(intent_blueprint.get("primary_intent", "")).strip().lower()

        if blueprint_agent != AgentType.GENERAL:
            if selected_agent == AgentType.GENERAL or confidence < 0.78:
                selected_agent = blueprint_agent
                confidence = max(confidence, 0.79)
                reason_parts.append(
                    f"Policy override: routed to {blueprint_agent.value} for {blueprint_intent or 'user-improvement'} intent."
                )

        if blueprint_intent == "daily_review":
            recent_entries = context.get("general_recent_time_entries", []) if isinstance(context, dict) else []
            if recent_entries:
                selected_agent = AgentType.PRODUCTIVITY
                confidence = max(confidence, 0.82)
                reason_parts.append("Detected review request with available history; routed to productivity for measurable feedback.")

        if selected_agent == AgentType.GENERAL and blueprint_intent in {"next_best_action", "planning"}:
            profile_agent = self._resolve_priority_agent(profile_snapshot)
            if profile_agent != AgentType.GENERAL:
                selected_agent = profile_agent
                confidence = max(confidence, 0.74)
                reason_parts.append(f"Used profile priorities to prefer {profile_agent.value} routing.")

        if selected_agent == AgentType.GENERAL and re.search(r"\b(work on|do now|next|priority|focus)\b", user_input.lower()):
            selected_agent = AgentType.PRODUCTIVITY
            confidence = max(confidence, 0.72)
            reason_parts.append("Action-oriented language detected; defaulted to productivity for concrete next steps.")

        result["agent_type"] = selected_agent
        result["confidence"] = min(confidence, 0.99)
        result["reason"] = " ".join(part for part in reason_parts if part).strip() or "Policy-guided routing"
        result["primary_intent"] = result.get("primary_intent") or intent_blueprint.get("primary_intent")
        result["expected_outcome"] = result.get("expected_outcome") or intent_blueprint.get("expected_outcome")
        result["time_horizon"] = result.get("time_horizon") or intent_blueprint.get("time_horizon")

        preferred_tone = (profile_snapshot or {}).get("preferred_tone")
        if preferred_tone and not result.get("coach_tone_hint"):
            result["coach_tone_hint"] = str(preferred_tone)

        result["method"] = f"{result.get('method', 'unknown')}_policy"
        return result

    def _compose_guided_user_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Compose a context-aware prompt for direct/simple responses."""
        profile_snapshot = context.get("profile_snapshot", {}) if isinstance(context.get("profile_snapshot"), dict) else {}
        coach_profile = context.get("coach_profile", {}) if isinstance(context.get("coach_profile"), dict) else {}
        intent_blueprint = context.get("intent_blueprint", {}) if isinstance(context.get("intent_blueprint"), dict) else {}
        knowledge_summary = str(context.get("knowledge_context_summary", "")).strip()

        priorities = profile_snapshot.get("priorities", []) if isinstance(profile_snapshot.get("priorities"), list) else []
        active_goals = profile_snapshot.get("active_goals", []) if isinstance(profile_snapshot.get("active_goals"), list) else []

        return f"""
User query:
{user_input}

Intent blueprint:
{json.dumps(intent_blueprint, indent=2)}

Profile snapshot:
{json.dumps({
    "role": profile_snapshot.get("role"),
    "priorities": priorities[:3],
    "active_goals": active_goals[:3],
}, indent=2)}

Coach profile:
{json.dumps(coach_profile, indent=2)}

Knowledge summary:
{knowledge_summary or 'No related summary available.'}

Response contract:
1. Align tone with coach style guidance.
2. Provide concrete, immediate value (not generic motivation).
3. Keep it concise and structured.
4. End with one practical next action.
""".strip()

    async def execute(self, state: Union[AgentState, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced execute method with deep agent patterns."""
        logger.info("=" * 80)
        logger.info("[ENHANCED ORCHESTRATOR] execute() method called!")
        logger.info("=" * 80)
        logger.debug("EnhancedOrchestrator.execute START")
        
        try:
            # Always refresh the service reference so provider switches are immediately visible.
            self.llm_service = await get_llm_service()
            
            # Normalize state input
            normalized_state = self._normalize_state(state)
            user_input = normalized_state["user_input"]
            context = normalized_state["context"]
            conversation_id = normalized_state.get("conversation_id")
            
            # Initialize or get deep state
            deep_state = self.state_manager.get_or_create_state(conversation_id or "default")
            
            # Store user message in deep state
            deep_state.add_message("user", user_input)

            # Pull user profile + recent knowledge context so routing reflects real user data.
            user_preferences_dict: Dict[str, Any] = {}
            profile_snapshot: Dict[str, Any] = {}
            coach_profile: Dict[str, Any] = {}
            try:
                user_preferences = await self.knowledge_base.get_user_preferences()
                if hasattr(user_preferences, "model_dump"):
                    user_preferences_dict = user_preferences.model_dump()
                elif isinstance(user_preferences, dict):
                    user_preferences_dict = user_preferences
                profile_snapshot = self._build_profile_snapshot(user_preferences_dict)
                coach_profile = self._build_coach_profile(profile_snapshot)
            except Exception as profile_error:
                logger.warning("Failed to load user preference snapshot: %s", profile_error)

            routing_context = dict(context or {})
            general_context: Dict[str, Any] = {}
            try:
                general_context = await self.knowledge_base.get_contextual_knowledge_for_agent(
                    user_input=user_input,
                    agent_type="general",
                    max_results=5,
                )
                if isinstance(general_context, dict):
                    routing_context["knowledge_context_summary"] = general_context.get("context_summary")
                    routing_context["general_recent_time_entries"] = general_context.get("recent_time_entries", [])[:3]
            except Exception as context_error:
                logger.warning("Failed to load general knowledge context: %s", context_error)

            if profile_snapshot:
                routing_context["profile_snapshot"] = profile_snapshot
            if coach_profile:
                routing_context["coach_profile"] = coach_profile

            intent_blueprint = self._derive_intent_blueprint(user_input, profile_snapshot)
            routing_context["intent_blueprint"] = intent_blueprint
            
            # Assess task complexity
            complexity = await self._assess_task_complexity(user_input)
            
            # Enhanced intent classification with complexity awareness
            intent_result = await self._enhanced_intent_classification(
                user_input,
                routing_context,
                complexity,
                profile_snapshot,
                intent_blueprint,
            )

            # Normalize the chosen target agent to avoid downstream type drift.
            target_agent = self._normalize_agent_type(intent_result.get("agent_type", AgentType.GENERAL))
            intent_result["agent_type"] = target_agent
            if coach_profile.get("style") and not intent_result.get("coach_tone_hint"):
                intent_result["coach_tone_hint"] = coach_profile.get("style")
            if intent_blueprint.get("primary_intent") and not intent_result.get("primary_intent"):
                intent_result["primary_intent"] = intent_blueprint.get("primary_intent")
            if intent_blueprint.get("expected_outcome") and not intent_result.get("expected_outcome"):
                intent_result["expected_outcome"] = intent_blueprint.get("expected_outcome")
            if intent_blueprint.get("time_horizon") and not intent_result.get("time_horizon"):
                intent_result["time_horizon"] = intent_blueprint.get("time_horizon")
            
            # Create strategic plan for complex tasks
            strategic_plan = None
            if complexity in [TaskComplexity.COMPLEX, TaskComplexity.ADVANCED]:
                strategic_plan = await self._create_strategic_plan(user_input, intent_result, complexity)
                if strategic_plan:
                    await self._store_plan_in_context(strategic_plan, deep_state)
            
            # Execute based on complexity and plan
            # Even for SIMPLE tasks, delegate to specialist if high-confidence intent classification
            confidence = intent_result.get("confidence", 0.0)
            
            logger.info(f"[DELEGATION DEBUG] complexity={complexity.value}, confidence={confidence}, target_agent={target_agent}")
            logger.info(f"[DELEGATION DEBUG] Condition check: SIMPLE={complexity == TaskComplexity.SIMPLE}, conf<0.8={confidence < 0.8}, is_GENERAL={target_agent == AgentType.GENERAL}")
            
            if complexity == TaskComplexity.SIMPLE and confidence < 0.8 and target_agent == AgentType.GENERAL:
                # Only handle directly if low confidence or explicitly GENERAL
                logger.info("[DELEGATION DEBUG] Taking _handle_simple_task path")
                response = await self._handle_simple_task(user_input, routing_context, deep_state)
            elif complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
                # Delegate to specialist for domain-specific tasks
                logger.info(f"[DELEGATION DEBUG] Taking _delegate_to_specialist path with agent={target_agent}")
                response = await self._delegate_to_specialist(
                    target_agent,
                    user_input,
                    routing_context,
                    deep_state
                )
            else:  # COMPLEX or ADVANCED
                logger.info("[DELEGATION DEBUG] Taking _orchestrate_complex_workflow path")
                response = await self._orchestrate_complex_workflow(
                    user_input,
                    strategic_plan,
                    intent_result,
                    deep_state
                )
            
            # Store response in deep state
            deep_state.add_message("assistant", response)
            
            # Update state manager
            self.state_manager.update_state(conversation_id or "default", deep_state)
            
            # Build reasoning for transparency (backward-compatible keys preserved for UI consumers).
            intent_payload = dict(intent_result or {})
            intent_agent = intent_payload.get("agent_type")
            resolved_agent_type = (
                intent_agent.value
                if isinstance(intent_agent, AgentType)
                else str(intent_agent or getattr(target_agent, "value", target_agent))
            )
            intent_payload["agent_type"] = resolved_agent_type

            execution_path = self._get_execution_path(complexity, strategic_plan)
            reasoning = {
                "complexity": complexity.value,
                "intent": intent_payload,
                "plan": strategic_plan,
                "execution_path": execution_path,
                "data_points_used": {
                    "role": profile_snapshot.get("role"),
                    "priorities": profile_snapshot.get("priorities", [])[:3],
                    "knowledge_context_summary": routing_context.get("knowledge_context_summary"),
                    "coach_style": coach_profile.get("style"),
                    "recent_time_entries_considered": len(routing_context.get("general_recent_time_entries", [])),
                },
                "coach_profile": coach_profile,
                "intent_blueprint": intent_blueprint,
                "finalAgent": resolved_agent_type,
                "classification": {
                    "agent_type": resolved_agent_type,
                    "confidence": float(intent_payload.get("confidence", 0.0) or 0.0),
                    "reason": str(intent_payload.get("reason", "")),
                },
                "steps": [
                    {
                        "agent": "orchestrator",
                        "action": f"Routed request to {resolved_agent_type} specialist",
                        "result": str(intent_payload.get("reason", ""))[:220],
                    },
                    *[
                        {
                            "agent": "orchestrator",
                            "action": str(path_step),
                        }
                        for path_step in execution_path
                    ],
                ],
            }
            
            return {
                "response": response,
                "reasoning": reasoning,
                "deep_state": deep_state.to_dict(),
                "context": routing_context,
            }
            
        except Exception as e:
            logger.error(f"Enhanced orchestrator execution failed: {e}")
            return {
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "reasoning": {"error": str(e)},
                "deep_state": {}
            }

    def _normalize_state(self, state: Union[AgentState, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize various state input formats."""
        if isinstance(state, str):
            return {
                "user_input": state,
                "context": {},
                "conversation_id": None,
                "agent": self.agent_id
            }
        elif isinstance(state, dict):
            return {
                "user_input": state.get("user_input", ""),
                "context": state.get("context", {}),
                "conversation_id": state.get("conversation_id"),
                "agent": state.get("agent", self.agent_id)
            }
        else:
            return {
                "user_input": str(state) if state else "",
                "context": {},
                "conversation_id": None,
                "agent": self.agent_id
            }

    async def _assess_task_complexity(self, user_input: str) -> TaskComplexity:
        """Assess task complexity using pattern matching and heuristics."""
        user_lower = user_input.lower()
        
        # Count complexity indicators
        complexity_scores = {complexity: 0 for complexity in TaskComplexity}
        
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, user_lower))
                complexity_scores[complexity] += matches
        
        # Additional heuristics
        word_count = len(user_input.split())
        if word_count > 50:
            complexity_scores[TaskComplexity.COMPLEX] += 2
        elif word_count > 20:
            complexity_scores[TaskComplexity.MODERATE] += 1
        
        # Check for multiple domains
        domain_count = sum(1 for patterns in self.intent_patterns.values() 
                          if any(re.search(pattern, user_lower) for pattern in patterns))
        if domain_count > 2:
            complexity_scores[TaskComplexity.COMPLEX] += 3
        elif domain_count > 1:
            complexity_scores[TaskComplexity.MODERATE] += 2
        
        # Find highest scoring complexity
        max_score = max(complexity_scores.values())
        if max_score == 0:
            return TaskComplexity.SIMPLE
        
        for complexity, score in complexity_scores.items():
            if score == max_score:
                return complexity
        
        return TaskComplexity.SIMPLE

    async def _enhanced_intent_classification(
        self, 
        user_input: str, 
        context: Dict[str, Any], 
        complexity: TaskComplexity,
        profile_snapshot: Optional[Dict[str, Any]] = None,
        intent_blueprint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enhanced intent classification with LLM support and complexity awareness."""
        try:
            intent_blueprint = intent_blueprint or {}

            # First, try pattern-based classification for speed
            pattern_result = self._pattern_based_classification(user_input, complexity)

            # For lower confidence or complex tasks, use LLM classification
            llm_result: Optional[Dict[str, Any]] = None
            if pattern_result["confidence"] <= 0.8:
                llm_result = await self._llm_based_classification(
                    user_input,
                    context,
                    complexity,
                    profile_snapshot,
                )

            # Combine results: prefer LLM when available and stronger confidence.
            base_result = pattern_result
            if llm_result and llm_result.get("confidence", 0.0) > pattern_result.get("confidence", 0.0):
                base_result = llm_result

            policy_result = self._apply_intent_policy(
                user_input=user_input,
                base_result=base_result,
                intent_blueprint=intent_blueprint,
                profile_snapshot=profile_snapshot,
                context=context,
            )

            normalized_agent = self._normalize_agent_type(policy_result.get("agent_type"))
            logger.info(
                "Intent routing decided: agent=%s confidence=%.2f method=%s primary_intent=%s",
                normalized_agent.value,
                float(policy_result.get("confidence", 0.0) or 0.0),
                policy_result.get("method"),
                policy_result.get("primary_intent"),
            )
            return policy_result
                
        except Exception as e:
            logger.error(f"Error in intent classification: {e}", exc_info=True)
            return {
                "agent_type": AgentType.GENERAL, 
                "confidence": 0.5, 
                "reason": f"Classification error: {str(e)}",
                "complexity_factor": complexity.value,
                "primary_intent": (intent_blueprint or {}).get("primary_intent", "general_guidance"),
                "expected_outcome": (intent_blueprint or {}).get("expected_outcome", "actionable_advice"),
                "time_horizon": (intent_blueprint or {}).get("time_horizon", "unspecified"),
            }
    
    def _pattern_based_classification(self, user_input: str, complexity: TaskComplexity) -> Dict[str, Any]:
        """Pattern-based intent classification using regex."""
        agent_scores = {}
        
        # Score each agent type
        for agent_type, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            matches = []
            
            for pattern in patterns:
                try:
                    regex = re.compile(pattern, flags=re.IGNORECASE)
                except re.error:
                    # Fallback to escaped pattern
                    regex = re.compile(re.escape(pattern), flags=re.IGNORECASE)
                
                found = regex.findall(user_input)
                if found:
                    score += len(found)
                    matched_patterns.append(pattern)
                    matches.extend(found)
            
            if score > 0:
                agent_scores[agent_type] = {
                    "score": score,
                    "patterns": matched_patterns,
                    "matches": matches,
                    "confidence": min(0.2 + (score * 0.25), 0.99)  # Normalize to 0.2-0.99
                }
        
        # No pattern matches
        if not agent_scores:
            return {
                "agent_type": AgentType.GENERAL,
                "confidence": 0.3,
                "reason": "No pattern matches found",
                "complexity_factor": complexity.value,
                "method": "pattern_based",
                "primary_intent": "general_guidance",
                "expected_outcome": "actionable_advice",
                "time_horizon": "unspecified",
            }
        
        # Find best agent
        best_agent = max(agent_scores.keys(), key=lambda k: agent_scores[k]["score"])
        result = agent_scores[best_agent]
        
        # Adjust confidence based on complexity
        complexity_multiplier = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.05,
            TaskComplexity.COMPLEX: 1.1,
            TaskComplexity.ADVANCED: 1.15
        }
        
        adjusted_confidence = min(result["confidence"] * complexity_multiplier[complexity], 0.99)
        
        return {
            "agent_type": best_agent,
            "confidence": adjusted_confidence,
            "reason": f"Pattern matches: {', '.join(map(str, result['matches'][:3]))}",
            "complexity_factor": complexity.value,
            "all_scores": {str(k.value): v["confidence"] for k, v in agent_scores.items()},
            "method": "pattern_based",
            "primary_intent": "execution_help" if best_agent == AgentType.PRODUCTIVITY else "general_guidance",
            "expected_outcome": "domain_specific_guidance",
            "time_horizon": "unspecified",
        }
    
    async def _llm_based_classification(
        self, 
        user_input: str, 
        context: Dict[str, Any],
        complexity: TaskComplexity,
        profile_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LLM-based intent classification for complex or ambiguous requests."""
        try:
            llm_service = await get_llm_service()
            
            agent_descriptions = {
                agent.value.upper(): self._get_agent_description(agent)
                for agent in AgentType
                if agent != AgentType.ORCHESTRATOR
            }

            classification_prompt = OrchestratorPromptLibrary.build_intent_classification_prompt(
                user_input=user_input,
                complexity=complexity.value,
                agent_descriptions=agent_descriptions,
                context=context,
                profile_snapshot=profile_snapshot,
            )

            # Make LLM request
            request = CompletionRequest(
                messages=[
                    ChatMessage(
                        role="system", 
                        content="You are an expert at classifying user intents for routing to specialized AI agents. You understand task complexity and can identify the most appropriate domain expert."
                    ),
                    ChatMessage(role="user", content=classification_prompt)
                ],
                max_tokens=200,
                temperature=0.0  # Deterministic for consistency
            )

            response = await llm_service.chat_completion(request)
            response_text = self._normalize_completion_text(getattr(response, "content", response))

            # Parse JSON response
            try:
                # Extract JSON if wrapped in markdown or other text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    parsed = json.loads(json_text)
                else:
                    parsed = json.loads(response_text)
                
                agent_label = parsed.get("agent_type", "GENERAL").upper()
                confidence = float(parsed.get("confidence", 0.5))
                reason = parsed.get("reason", response_text)
                primary_intent = str(parsed.get("primary_intent", "general_guidance"))
                expected_outcome = str(parsed.get("expected_outcome", "actionable_advice"))
                time_horizon = str(parsed.get("time_horizon", "unspecified"))
                coach_tone_hint = str(parsed.get("coach_tone_hint", "")).strip() or None
                
                # Map label to AgentType enum
                agent_type = self._normalize_agent_type(agent_label)
                
                return {
                    "agent_type": agent_type,
                    "confidence": confidence,
                    "reason": reason,
                    "complexity_factor": complexity.value,
                    "method": "llm_based",
                    "primary_intent": primary_intent,
                    "expected_outcome": expected_outcome,
                    "time_horizon": time_horizon,
                    "coach_tone_hint": coach_tone_hint,
                }
                
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.warning(f"Failed to parse LLM JSON response: {parse_error}. Response: {response_text}")
                
                # Fallback: scan for agent type keywords in response
                for agent in AgentType:
                    if agent.value.upper() in response_text.upper() or agent.name in response_text:
                        return {
                            "agent_type": agent,
                            "confidence": 0.65,
                            "reason": response_text[:100],
                            "complexity_factor": complexity.value,
                            "method": "llm_based_fallback",
                            "primary_intent": "general_guidance",
                            "expected_outcome": "actionable_advice",
                            "time_horizon": "unspecified",
                        }
                
                # Ultimate fallback
                return {
                    "agent_type": AgentType.GENERAL,
                    "confidence": 0.4,
                    "reason": f"Could not parse LLM response: {response_text[:100]}",
                    "complexity_factor": complexity.value,
                    "method": "llm_based_error",
                    "primary_intent": "general_guidance",
                    "expected_outcome": "actionable_advice",
                    "time_horizon": "unspecified",
                }
                
        except Exception as e:
            logger.error(f"Error in LLM-based classification: {e}", exc_info=True)
            return {
                "agent_type": AgentType.GENERAL,
                "confidence": 0.3,
                "reason": f"LLM classification error: {str(e)}",
                "complexity_factor": complexity.value,
                "method": "llm_error",
                "primary_intent": "general_guidance",
                "expected_outcome": "actionable_advice",
                "time_horizon": "unspecified",
            }
    
    def _get_agent_description(self, agent_type: AgentType) -> str:
        """Get description for each agent type."""
        descriptions = {
            AgentType.PRODUCTIVITY: "Task management, TODO lists, goals, Leetcode problems, coding practice, project tracking, time optimization",
            AgentType.HEALTH: "Wellness tracking, exercise routines, nutrition, meal planning, sleep, fitness goals, habit formation",
            AgentType.FINANCE: "Expense tracking, budgeting, financial planning, investment advice, money management",
            AgentType.SCHEDULING: "Calendar management, appointments, time scheduling, meeting coordination, reminders",
            AgentType.JOURNAL: "Daily reflections, mood tracking, gratitude journaling, personal growth, insights",
            AgentType.GENERAL: "General questions, casual conversation, requests that don't fit specialized domains"
        }
        return descriptions.get(agent_type, "General purpose agent")

    def _build_profile_snapshot(self, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Build a compact profile snapshot for routing and prompting."""
        general = user_preferences.get("general", {}) if isinstance(user_preferences, dict) else {}
        mentor = general.get("mentor", {}) if isinstance(general.get("mentor"), dict) else {}

        active_goals: List[str] = []
        for section in ("productivity", "health", "finance", "journal"):
            section_data = user_preferences.get(section, {}) if isinstance(user_preferences, dict) else {}
            goals = section_data.get("goals", []) if isinstance(section_data, dict) else []
            for goal in goals[:2]:
                if isinstance(goal, dict):
                    title = str(goal.get("title", "")).strip()
                    if title:
                        active_goals.append(title)

        priorities = general.get("priorities", []) if isinstance(general, dict) else []
        if not isinstance(priorities, list):
            priorities = []

        return {
            "role": general.get("role"),
            "priorities": priorities,
            "preferred_tone": mentor.get("style") or general.get("preferredTone"),
            "mentor": {
                "name": mentor.get("name"),
                "archetype": mentor.get("archetype"),
                "style": mentor.get("style"),
            },
            "active_goals": active_goals,
        }

    async def _create_strategic_plan(
        self, 
        user_input: str, 
        intent_result: Dict[str, Any], 
        complexity: TaskComplexity
    ) -> Optional[Dict[str, Any]]:
        """Create strategic plan for complex tasks."""
        try:
            # Use the think tool to generate plan
            from .think_tools import create_think_tools
            think_tools = create_think_tools()
            
            # Create planning prompt
            identified_domain = self._normalize_agent_type(intent_result.get("agent_type")).value
            planning_prompt = f"""Analyze this complex request and create a strategic plan:

**User Request:** {user_input}

**Identified Domain:** {identified_domain}
**Task Complexity:** {complexity.value}
**Classification Confidence:** {intent_result.get('confidence', 0):.2f}

Create a structured plan that:
1. Breaks down the request into logical steps
2. Identifies which specialized agents should handle each step
3. Determines the optimal sequence and dependencies
4. Specifies success criteria for each step
5. Identifies potential risks or challenges

Focus on practical, actionable steps that leverage our specialized ReAct agents effectively."""
            
            # Create mock deep state for tool execution
            mock_state = DeepAgentState(conversation_id="planning")
            
            # Execute thinking tool (simplified for now)
            plan = {
                "complexity": complexity.value,
                "primary_domain": identified_domain,
                "confidence": intent_result.get('confidence', 0),
                "steps": self._generate_plan_steps(user_input, intent_result, complexity),
                "agents_involved": self._identify_required_agents(user_input, intent_result),
                "success_criteria": self._define_success_criteria(user_input, complexity),
                "estimated_duration": self._estimate_duration(complexity),
                "risk_factors": self._identify_risks(complexity)
            }
            
            return plan
            
        except Exception as e:
            logger.warning(f"Failed to create strategic plan: {e}")
            return None

    def _generate_plan_steps(
        self, 
        user_input: str, 
        intent_result: Dict[str, Any], 
        complexity: TaskComplexity
    ) -> List[Dict[str, Any]]:
        """Generate logical plan steps based on request analysis."""
        steps = []
        
        # Basic step generation based on complexity
        if complexity == TaskComplexity.COMPLEX:
            primary_agent = self._normalize_agent_type(intent_result.get('agent_type', AgentType.GENERAL)).value
            steps = [
                {
                    "step": 1,
                    "action": "Analyze and understand the request",
                    "agent": "orchestrator",
                    "description": "Break down the user request into core components",
                    "estimated_time": 2
                },
                {
                    "step": 2,
                    "action": "Delegate to primary specialist",
                    "agent": primary_agent,
                    "description": f"Use {primary_agent} agent for domain expertise",
                    "estimated_time": 10
                },
                {
                    "step": 3,
                    "action": "Synthesize and validate results",
                    "agent": "orchestrator",
                    "description": "Combine specialist insights into comprehensive response",
                    "estimated_time": 3
                }
            ]
        elif complexity == TaskComplexity.ADVANCED:
            steps = [
                {
                    "step": 1,
                    "action": "Deep analysis and planning",
                    "agent": "orchestrator",
                    "description": "Comprehensive breakdown with dependency mapping",
                    "estimated_time": 5
                },
                {
                    "step": 2,
                    "action": "Multi-agent coordination",
                    "agent": "multiple",
                    "description": "Coordinate between multiple specialized agents",
                    "estimated_time": 20
                },
                {
                    "step": 3,
                    "action": "Integration and optimization",
                    "agent": "orchestrator",
                    "description": "Integrate results and optimize for user needs",
                    "estimated_time": 8
                },
                {
                    "step": 4,
                    "action": "Validation and delivery",
                    "agent": "orchestrator",
                    "description": "Final validation and formatted delivery",
                    "estimated_time": 5
                }
            ]
        
        return steps

    def _identify_required_agents(
        self, 
        user_input: str, 
        intent_result: Dict[str, Any]
    ) -> List[str]:
        """Identify which agents are required for the request."""
        required_agents = ["orchestrator"]
        
        # Add primary agent
        primary_agent = self._normalize_agent_type(intent_result.get('agent_type', AgentType.GENERAL)).value
        if primary_agent:
            required_agents.append(primary_agent)
        
        # Check for multi-domain requirements
        user_lower = user_input.lower()
        for agent_type, patterns in self.intent_patterns.items():
            if agent_type.value not in required_agents:
                for pattern in patterns:
                    if re.search(pattern, user_lower):
                        required_agents.append(agent_type.value)
                        break
        
        return required_agents

    def _define_success_criteria(
        self, 
        user_input: str, 
        complexity: TaskComplexity
    ) -> List[str]:
        """Define success criteria based on request and complexity."""
        criteria = [
            "User request is fully addressed",
            "Response is clear and actionable"
        ]
        
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.ADVANCED]:
            criteria.extend([
                "All plan steps are completed successfully",
                "Specialist insights are properly integrated",
                "Context is preserved for future reference"
            ])
        
        if complexity == TaskComplexity.ADVANCED:
            criteria.extend([
                "Multi-agent coordination is seamless",
                "Results exceed user expectations",
                "Workflow is optimized for efficiency"
            ])
        
        return criteria

    def _estimate_duration(self, complexity: TaskComplexity) -> int:
        """Estimate duration in minutes based on complexity."""
        duration_map = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 5,
            TaskComplexity.COMPLEX: 15,
            TaskComplexity.ADVANCED: 30
        }
        return duration_map.get(complexity, 5)

    def _identify_risks(self, complexity: TaskComplexity) -> List[str]:
        """Identify potential risks based on complexity."""
        base_risks = ["Misunderstanding user intent", "Technical execution issues"]
        
        if complexity == TaskComplexity.COMPLEX:
            base_risks.extend([
                "Agent coordination challenges",
                "Context loss between steps"
            ])
        elif complexity == TaskComplexity.ADVANCED:
            base_risks.extend([
                "Multi-agent synchronization issues",
                "Resource allocation conflicts",
                "Complexity overwhelm for user"
            ])
        
        return base_risks

    async def _store_plan_in_context(
        self, 
        plan: Dict[str, Any], 
        deep_state: DeepAgentState
    ) -> None:
        """Store plan in deep state for context preservation."""
        try:
            # Use file tools to store plan
            from .file_tools import create_file_tools
            file_tools = create_file_tools()
            
            plan_content = f"""# Strategic Plan - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Complexity:** {plan['complexity']}
- **Primary Domain:** {plan['primary_domain']}
- **Confidence:** {plan['confidence']:.2f}
- **Estimated Duration:** {plan['estimated_duration']} minutes

## Required Agents
{chr(10).join(f'- {agent}' for agent in plan['agents_involved'])}

## Execution Steps
{chr(10).join(f"{step['step']}. **{step['action']}** ({step['agent']}) - {step['description']} (~{step['estimated_time']}min)" for step in plan['steps'])}

## Success Criteria
{chr(10).join(f'- {criteria}' for criteria in plan['success_criteria'])}

## Risk Factors
{chr(10).join(f'- {risk}' for risk in plan['risk_factors'])}
"""
            
            # Store in deep state files
            plan_filename = f"strategic_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            deep_state.files[plan_filename] = plan_content
            
            logger.info(f"Strategic plan stored in {plan_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to store plan in context: {e}")

    async def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences from knowledge base."""
        try:
            kb_result = await self.knowledge_base.get_user_preferences(user_id)
            return kb_result if kb_result else {}
        except Exception as e:
            logger.warning(f"Failed to load user preferences: {e}")
            return {}

    async def _handle_simple_task(
        self, 
        user_input: str, 
        context: Dict[str, Any], 
        deep_state: DeepAgentState
    ) -> str:
        """Handle simple tasks directly without delegation."""
        try:
            # Use LLM service directly for simple responses
            guided_prompt = self._compose_guided_user_prompt(user_input, context or {})
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=guided_prompt)
            ]
            
            request = CompletionRequest(
                messages=messages,
                max_tokens=340,
                temperature=0.35
            )
            
            response = await self.llm_service.chat_completion(request)
            return self._normalize_completion_text(getattr(response, "content", response))
            
        except Exception as e:
            logger.error(f"Simple task handling failed: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

    async def _delegate_to_specialist(
        self, 
        agent_type: AgentType, 
        user_input: str, 
        context: Dict[str, Any], 
        deep_state: 'DeepAgentStateManager'  # This is actually a manager, not the state itself
    ) -> str:
        """Delegate to specialized agent."""
        try:
            normalized_agent_type = self._normalize_agent_type(agent_type)
            # Get the specialized agent from registry
            registry = get_agent_registry()
            agents = registry.get_agents_by_type(normalized_agent_type)
            
            if not agents:
                logger.warning(f"No agent found for type {normalized_agent_type}, falling back to simple handling")
                return await self._handle_simple_task(user_input, context, deep_state)
            
            # Use the first registered agent of this type
            specialist_agent = agents[0]
            logger.info(f"[DELEGATION DEBUG] Delegating to specialist: {specialist_agent.agent_id} ({normalized_agent_type.value})")
            
            # Prepare state for delegation
            delegation_state = AgentState(
                user_input=user_input,
                context=context,
                conversation_id=deep_state.state.get("conversation_id", "default"),
                agent=specialist_agent.agent_id,
                messages=[],
                next_agent=None,
                final_response=None
            )
            
            # Execute the specialist agent
            result = await specialist_agent.execute(delegation_state)
            logger.info(f"[DELEGATION DEBUG] Specialist returned result type: {type(result)}")
            
            # Extract response
            if isinstance(result, dict):
                response = result.get("final_response") or result.get("response") or str(result)
            elif isinstance(result, AgentState):
                response = result.final_response or "Task completed by specialist."
            else:
                response = str(result)
            
            logger.info(f"[DELEGATION DEBUG] Extracted response preview: {response[:200] if isinstance(response, str) else str(response)[:200]}")
            return response
                
        except Exception as e:
            logger.error(f"Specialist delegation failed: {e}")
            return f"I encountered an issue while delegating to the specialist: {str(e)}"

    async def _orchestrate_complex_workflow(
        self, 
        user_input: str, 
        plan: Dict[str, Any], 
        intent_result: Dict[str, Any], 
        deep_state: DeepAgentState
    ) -> str:
        """Orchestrate complex multi-agent workflow."""
        try:
            if not plan:
                # Fallback to single agent delegation
                fallback_agent = self._normalize_agent_type(intent_result.get('agent_type', AgentType.GENERAL))
                return await self._delegate_to_specialist(
                    fallback_agent,
                    user_input,
                    {},
                    deep_state
                )
            
            workflow_results = []
            
            # Execute each step in the plan
            for step in plan.get('steps', []):
                step_result = await self._execute_workflow_step(
                    step, 
                    user_input, 
                    workflow_results, 
                    deep_state
                )
                workflow_results.append(step_result)
            
            # Synthesize results
            return await self._synthesize_workflow_results(
                workflow_results, 
                plan, 
                user_input, 
                deep_state
            )
            
        except Exception as e:
            logger.error(f"Complex workflow orchestration failed: {e}")
            return f"I encountered an issue during workflow execution: {str(e)}"

    async def _execute_workflow_step(
        self, 
        step: Dict[str, Any], 
        user_input: str, 
        previous_results: List[Dict[str, Any]], 
        deep_state: DeepAgentState
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_start = datetime.now()
        
        try:
            agent = step.get('agent', 'orchestrator')
            action = step.get('action', 'Process step')
            description = step.get('description', '')
            
            if agent == 'orchestrator':
                result = await self._execute_orchestrator_step(step, user_input, previous_results)
            else:
                result = await self._execute_agent_step(step, user_input, previous_results, deep_state)
            
            return {
                "step": step.get('step', 0),
                "agent": agent,
                "action": action,
                "description": description,
                "result": result,
                "duration": (datetime.now() - step_start).total_seconds(),
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "step": step.get('step', 0),
                "agent": step.get('agent', 'unknown'),
                "action": step.get('action', 'Unknown action'),
                "description": step.get('description', ''),
                "result": f"Error: {str(e)}",
                "duration": (datetime.now() - step_start).total_seconds(),
                "status": "failed"
            }

    async def _execute_orchestrator_step(
        self, 
        step: Dict[str, Any], 
        user_input: str, 
        previous_results: List[Dict[str, Any]]
    ) -> str:
        """Execute orchestrator-specific step."""
        action = step.get('action', '').lower()
        
        if 'analyze' in action:
            return f"Analyzed request: '{user_input}' - Identified key components and requirements."
        elif 'synthesize' in action:
            return f"Synthesized {len(previous_results)} previous results into comprehensive response."
        elif 'validate' in action:
            return "Validated results for completeness and accuracy."
        else:
            return f"Completed orchestrator action: {step.get('action', 'Unknown')}"

    async def _execute_agent_step(
        self, 
        step: Dict[str, Any], 
        user_input: str, 
        previous_results: List[Dict[str, Any]], 
        deep_state: DeepAgentState
    ) -> str:
        """Execute step using specialized agent."""
        agent_name = step.get('agent', '').lower()
        action = step.get('action', '')
        description = step.get('description', '')
        
        # Map agent names to types
        agent_type_map = {
            'productivity': AgentType.PRODUCTIVITY,
            'health': AgentType.HEALTH,
            'finance': AgentType.FINANCE,
            'scheduling': AgentType.SCHEDULING,
            'journal': AgentType.JOURNAL
        }
        
        agent_type = agent_type_map.get(agent_name, AgentType.GENERAL)
        
        # Create focused prompt for this step
        step_prompt = f"""Focus specifically on this task: {description}

Original user request: {user_input}
Specific action needed: {action}

Please provide a focused response addressing this specific aspect of the user's request."""
        
        result = await self._delegate_to_specialist(agent_type, step_prompt, {}, deep_state)
        
        return result

    async def _synthesize_workflow_results(
        self, 
        results: List[Dict[str, Any]], 
        plan: Dict[str, Any], 
        user_input: str, 
        deep_state: DeepAgentState
    ) -> str:
        """Synthesize results from workflow execution."""
        try:
            synthesis_prompt = f"""Synthesize the following workflow results into a comprehensive response for the user.

Original Request: {user_input}

Plan Overview:
- Complexity: {plan.get('complexity', 'unknown')}
- Primary Domain: {plan.get('primary_domain', 'general')}
- Agents Involved: {', '.join(plan.get('agents_involved', []))}

Execution Results:
"""
            
            for result in results:
                synthesis_prompt += f"""
Step {result['step']}: {result['action']} ({result['agent']})
Result: {result['result']}
Status: {result['status']} ({result['duration']:.1f}s)
"""
            
            synthesis_prompt += """

Please create a comprehensive, well-structured response that:
1. Directly addresses the user's original request
2. Integrates insights from all workflow steps
3. Provides actionable recommendations
4. Maintains a helpful and professional tone
5. Highlights key achievements and next steps if applicable
"""
            
            # Use LLM for synthesis
            messages = [
                ChatMessage(role="system", content="You are an expert at synthesizing complex workflow results into clear, actionable responses."),
                ChatMessage(role="user", content=synthesis_prompt)
            ]
            
            request = CompletionRequest(
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response = await self.llm_service.chat_completion(request)
            return self._normalize_completion_text(getattr(response, "content", response))
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            # Fallback to simple concatenation
            results_text = "\n\n".join([
                f"**{result['action']}:** {result['result']}" 
                for result in results 
                if result['status'] == 'completed'
            ])
            return f"Here are the results from your request:\n\n{results_text}"

    def _get_execution_path(self, complexity: TaskComplexity, plan: Optional[Dict[str, Any]]) -> List[str]:
        """Get human-readable execution path steps."""
        if complexity == TaskComplexity.SIMPLE:
            return ["Direct response by orchestrator"]

        if complexity == TaskComplexity.MODERATE:
            return ["Single specialist agent delegation"]

        if plan:
            plan_steps = plan.get("steps", []) if isinstance(plan.get("steps"), list) else []
            formatted_steps: List[str] = []

            for index, step in enumerate(plan_steps, start=1):
                if not isinstance(step, dict):
                    continue

                action = str(step.get("action") or step.get("description") or "").strip()
                if not action:
                    continue

                agent = str(step.get("agent") or "orchestrator").strip() or "orchestrator"
                formatted_steps.append(f"Step {index}: {action} ({agent})")

            if formatted_steps:
                return formatted_steps

            agent_count = len(plan.get("agents_involved", []))
            step_count = len(plan_steps)
            return [f"Complex workflow: {step_count} steps across {agent_count} agents"]

        return ["Fallback to specialist delegation"]


# Factory function for getting enhanced orchestrator
_enhanced_orchestrator = None

def get_enhanced_orchestrator() -> EnhancedOrchestratorAgent:
    """Get the global enhanced orchestrator instance."""
    global _enhanced_orchestrator
    if _enhanced_orchestrator is None:
        _enhanced_orchestrator = EnhancedOrchestratorAgent()
    return _enhanced_orchestrator