"""
Enhanced Orchestrator Agent with Deep Agent Planning and ReAct Sub-Agent Integration
==================================================================================

Enhanced orchestrator that incorporates the 4 principles of Deep Agents:
1. Planning - Strategic task breakdown and planning capabilities
2. Context Offloading - File-based context storage and retrieval  
3. Task Delegation - Intelligent delegation to ReAct sub-agents
4. Prompt Engineering - Context-aware and specialized prompts

This orchestrator can handle complex workflows by planning, delegating to
specialized ReAct agents, and maintaining context across interactions.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId  
from typing_extensions import Annotated

# Internal imports
from .base import BaseAgent, AgentType, AgentCapability, AgentState
from .prompts import PromptLibrary, get_agent_prompt
from .registry import get_agent_registry
from .communication import get_communication_protocol, MessageType
from .react_factory import get_react_agent_factory
from .deep_state import DeepAgentState, DeepAgentStateManager, Todo, HumanApprovalRequest
from .file_tools import create_file_tools
from .todo_tools import create_todo_tools
from .think_tools import create_think_tools
from ..llm.service import get_llm_service
from ..llm.base import CompletionRequest, ChatMessage
from ..services.knowledge_base import get_knowledge_base_service
from ..services.interaction_recorder import get_interaction_recorder
from ..utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.AGENT)


class TaskComplexity(Enum):
    """Task complexity levels for planning decisions."""
    SIMPLE = "simple"           # Single step, direct response
    MODERATE = "moderate"       # Multiple steps, single agent
    COMPLEX = "complex"         # Multiple agents, planning required  
    ADVANCED = "advanced"       # Long-term planning, human oversight


class EnhancedOrchestratorAgent(BaseAgent):
    """Enhanced orchestrator with deep agent patterns and ReAct sub-agent integration."""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="strategic_planning",
                description="Break down complex tasks into actionable plans with delegation",
                parameters={"max_plan_depth": 5, "planning_confidence_threshold": 0.8}
            ),
            AgentCapability(
                name="context_offloading",
                description="Store and retrieve context using file-based storage",
                parameters={"max_context_size": 50000, "context_retention_days": 30}
            ),
            AgentCapability(
                name="react_delegation",
                description="Delegate tasks to specialized ReAct sub-agents with isolated contexts",
                parameters={"max_concurrent_delegations": 3, "delegation_timeout": 300}
            ),
            AgentCapability(
                name="prompt_engineering",
                description="Generate context-aware prompts for optimal agent performance", 
                parameters={"context_window": 8000, "prompt_optimization_level": "advanced"}
            ),
            AgentCapability(
                name="human_in_the_loop",
                description="Request human approval for critical decisions and actions",
                parameters={"approval_threshold": "high_impact", "timeout_seconds": 600}
            ),
            AgentCapability(
                name="intent_classification",
                description="Advanced intent classification with multi-agent delegation",
                parameters={"confidence_threshold": 0.7, "multi_agent_threshold": 0.9}
            )
        ]
        
        system_prompt = self._build_enhanced_system_prompt()
        
        super().__init__(
            agent_id="enhanced_orchestrator",
            agent_type=AgentType.ORCHESTRATOR,
            capabilities=capabilities,
            system_prompt=system_prompt
        )
        
        # Initialize services
        self.registry = get_agent_registry()
        self.communication = get_communication_protocol()
        self.knowledge_base = get_knowledge_base_service()
        self.react_factory = get_react_agent_factory()
        self.state_manager = DeepAgentStateManager()
        self.llm_service = None  # Will be initialized when needed
        
        # Enhanced intent classification patterns
        self.intent_patterns = {
            AgentType.PRODUCTIVITY: [
                r'\\b(task|todo|goal|work|project|deadline|priority|manage|organize)\\b',
                r'\\b(efficient|focus|time management|workflow|deliverable|milestone)\\b',
                r'\\b(meeting|agenda|assignment|schedule task|plan work|productivity)\\b',
                r'\\b(create goal|track progress|set deadline|prioritize|optimize)\\b'
            ],
            AgentType.HEALTH: [
                r'\\b(health|wellness|exercise|fitness|habit|routine|workout|diet)\\b',
                r'\\b(sleep|nutrition|meal|food|eating|recipe|calories|weight)\\b',
                r'\\b(meditation|mindfulness|stress|mental health|wellbeing)\\b',
                r'\\b(track habit|meal plan|workout plan|health goal|wellness check)\\b'
            ],
            AgentType.FINANCE: [
                r'\\b(money|budget|expense|spending|financial|finance|investment)\\b',
                r'\\b(save|saving|income|cost|price|salary|bank|account)\\b',
                r'\\b(transaction|bill|payment|purchase|retirement|loan|credit)\\b',
                r'\\b(track expense|create budget|financial plan|analyze spending)\\b'
            ],
            AgentType.SCHEDULING: [
                r'\\b(calendar|appointment|meeting|schedule|time|date|book)\\b',
                r'\\b(reserve|plan|arrange|organize|reschedule|available|busy)\\b',
                r'\\b(reminder|event|deadline|booking|slot|conflict|timing)\\b'
            ],
            AgentType.JOURNAL: [
                r'\\b(journal|reflect|reflection|mood|feeling|emotion|diary)\\b',
                r'\\b(thoughts|gratitude|mindset|growth|personal|celebrate)\\b',
                r'\\b(achievement|milestone|memory|insight|learning|experience)\\b',
                r'\\b(daily reflection|weekly reflection|progress reflection)\\b'
            ]
        }
        
        # Complexity indicators for planning decisions
        self.complexity_indicators = {
            TaskComplexity.SIMPLE: [
                r'\\b(what|who|when|where|how)\\s+is\\b',  # Simple questions
                r'\\b(tell me|show me|explain)\\b',         # Information requests
                r'\\b(quick|simple|brief)\\b'               # Quick tasks
            ],
            TaskComplexity.MODERATE: [
                r'\\b(create|make|build|generate|develop)\\b',     # Creation tasks
                r'\\b(analyze|review|check|evaluate)\\b',          # Analysis tasks
                r'\\b(plan|organize|structure)\\b'                 # Planning tasks
            ],
            TaskComplexity.COMPLEX: [
                r'\\b(multiple|several|various|different)\\b',     # Multi-part tasks
                r'\\b(coordinate|integrate|combine|merge)\\b',     # Integration tasks
                r'\\b(comprehensive|detailed|thorough)\\b'         # Comprehensive tasks
            ],
            TaskComplexity.ADVANCED: [
                r'\\b(long.?term|ongoing|continuous|strategic)\\b', # Long-term tasks
                r'\\b(enterprise|business|critical|important)\\b',  # High-stakes tasks
                r'\\b(approval|review|oversight|governance)\\b'     # Oversight tasks
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

    async def execute(self, state: Union[AgentState, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced execute method with deep agent patterns."""
        logger.debug("enhanced_execute_start", "Enhanced Orchestrator.execute START")
        
        try:
            # Normalize input state
            normalized_state = self._normalize_state(state)
            user_input = normalized_state.get("user_input", "")
            context = normalized_state.get("context", {})
            conversation_id = normalized_state.get("conversation_id")
            
            # Initialize deep agent state if needed
            if "deep_state" not in normalized_state:
                deep_state = await self.state_manager.get_or_create_state(conversation_id or "default")
                normalized_state["deep_state"] = deep_state
            else:
                deep_state = normalized_state["deep_state"]
            
            # Step 1: Assess task complexity
            complexity = await self._assess_task_complexity(user_input)
            
            # Step 2: Enhanced intent classification
            intent_result = await self._enhanced_intent_classification(user_input, context, complexity)
            
            # Step 3: Strategic planning (for complex tasks)
            plan = None
            if complexity in [TaskComplexity.COMPLEX, TaskComplexity.ADVANCED]:
                plan = await self._create_strategic_plan(user_input, intent_result, complexity)
                
                # Store plan in deep state
                if plan:
                    await self._store_plan_context(deep_state, plan)
            
            # Step 4: Execute based on complexity and plan
            if complexity == TaskComplexity.SIMPLE:
                response = await self._handle_simple_task(user_input, context)
                reasoning = {
                    "approach": "direct_response",
                    "complexity": complexity.value,
                    "agent": "orchestrator"
                }
            
            elif complexity == TaskComplexity.MODERATE:
                response = await self._delegate_to_specialist(
                    intent_result["agent_type"], user_input, context, deep_state
                )
                reasoning = {
                    "approach": "single_agent_delegation",
                    "complexity": complexity.value,
                    "agent": intent_result["agent_type"].value,
                    "confidence": intent_result["confidence"]
                }
            
            else:  # COMPLEX or ADVANCED
                response = await self._orchestrate_complex_workflow(
                    plan, user_input, context, deep_state, complexity
                )
                reasoning = {
                    "approach": "multi_agent_orchestration",
                    "complexity": complexity.value,
                    "plan_steps": len(plan.get("steps", [])) if plan else 0,
                    "agents_involved": plan.get("agents_involved", []) if plan else []
                }
            
            # Step 5: Store valuable insights and context
            await self._store_interaction_insights(
                deep_state, user_input, response, reasoning
            )
            
            # Step 6: Update state with results
            updated_state = {
                **normalized_state,
                "response": response,
                "reasoning": reasoning,
                "deep_state": deep_state,
                "last_complexity": complexity.value,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug("enhanced_execute_complete", "Enhanced Orchestrator.execute COMPLETE")
            return updated_state
            
        except Exception as e:
            logger.error("execution_error", "Error in enhanced orchestrator execution", error=e)
            return {
                **normalized_state,
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "reasoning": {
                    "approach": "error_handling",
                    "error": str(e)
                }
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
            return state
        else:
            return {
                "user_input": str(state) if state else "",
                "context": {},
                "conversation_id": None,
                "agent": self.agent_id
            }

    async def _assess_task_complexity(self, user_input: str) -> TaskComplexity:
        """Assess task complexity using pattern matching and heuristics."""
        user_input_lower = user_input.lower()
        
        # Check for complexity indicators
        complexity_scores = {complexity: 0 for complexity in TaskComplexity}
        
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    complexity_scores[complexity] += 1
        
        # Additional heuristics
        word_count = len(user_input.split())
        question_marks = user_input.count('?')
        and_or_indicators = len(re.findall(r'\\b(and|or|also|additionally|furthermore)\\b', user_input_lower))
        
        # Simple tasks: short, single questions
        if word_count <= 10 and question_marks == 1 and and_or_indicators == 0:
            complexity_scores[TaskComplexity.SIMPLE] += 2
        
        # Complex tasks: long requests with multiple parts
        if word_count > 50 or and_or_indicators >= 3:
            complexity_scores[TaskComplexity.COMPLEX] += 2
        
        # Advanced tasks: mentions of long-term, business, critical
        if any(word in user_input_lower for word in ['long-term', 'strategic', 'business', 'critical', 'enterprise']):
            complexity_scores[TaskComplexity.ADVANCED] += 3
        
        # Return highest scoring complexity
        max_complexity = max(complexity_scores.items(), key=lambda x: x[1])
        
        # Default to moderate if no clear indicators
        if max_complexity[1] == 0:
            return TaskComplexity.MODERATE
        
        return max_complexity[0]

    async def _enhanced_intent_classification(
        self, 
        user_input: str, 
        context: Dict[str, Any], 
        complexity: TaskComplexity
    ) -> Dict[str, Any]:
        """Enhanced intent classification with complexity awareness."""
        user_input_lower = user_input.lower()
        
        # Score each agent type
        agent_scores = {}
        for agent_type, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                matches = len(re.findall(pattern, user_input_lower))
                if matches > 0:
                    score += matches
                    matched_patterns.append(pattern)
            
            if score > 0:
                agent_scores[agent_type] = {
                    "score": score,
                    "patterns": matched_patterns
                }
        
        # Determine best match
        if not agent_scores:
            return {
                "agent_type": AgentType.GENERAL,
                "confidence": 0.1,
                "reason": "No specific domain patterns matched",
                "complexity_adjusted": False
            }
        
        best_match = max(agent_scores.items(), key=lambda x: x[1]["score"])
        agent_type = best_match[0]
        score_info = best_match[1]
        
        # Calculate confidence based on score and complexity
        max_possible_score = len(self.intent_patterns[agent_type])
        base_confidence = min(score_info["score"] / max_possible_score, 1.0)
        
        # Adjust confidence based on complexity
        if complexity == TaskComplexity.SIMPLE:
            confidence = base_confidence * 0.9  # Slightly lower for simple tasks
        elif complexity == TaskComplexity.MODERATE:
            confidence = base_confidence
        else:
            confidence = min(base_confidence * 1.1, 1.0)  # Higher for complex tasks
        
        return {
            "agent_type": agent_type,
            "confidence": confidence,
            "reason": f"Matched {score_info['score']} patterns for {agent_type.value}",
            "matched_patterns": score_info["patterns"],
            "complexity_adjusted": True,
            "all_scores": {str(k.value): v["score"] for k, v in agent_scores.items()}
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
            planning_prompt = f"""Analyze this complex request and create a strategic plan:

**User Request:** {user_input}

**Identified Domain:** {intent_result['agent_type'].value if intent_result.get('agent_type') else 'General'}
**Task Complexity:** {complexity.value}
**Classification Confidence:** {intent_result.get('confidence', 0):.2f}

Create a structured plan that:
1. Breaks down the request into logical steps
2. Identifies which specialized agents should handle each step
3. Determines the optimal sequence and dependencies
4. Specifies success criteria for each step
5. Identifies potential risks or challenges

Focus on practical, actionable steps that leverage our specialized ReAct agents effectively."""
            
            # For now, return a structured plan based on intent and complexity
            plan = {
                "request": user_input,
                "complexity": complexity.value,
                "primary_domain": intent_result.get('agent_type').value if intent_result.get('agent_type') else 'general',
                "confidence": intent_result.get('confidence', 0),
                "created_at": datetime.now().isoformat(),
                "steps": self._generate_plan_steps(user_input, intent_result, complexity),
                "agents_involved": self._identify_required_agents(user_input, intent_result),
                "success_criteria": self._define_success_criteria(user_input, complexity),
                "estimated_duration": self._estimate_duration(complexity),
                "risk_factors": self._identify_risks(complexity)
            }
            
            return plan
            
        except Exception as e:
            logger.error("plan_creation_error", "Error creating strategic plan", error=e)
            return None

    def _generate_plan_steps(
        self, 
        user_input: str, 
        intent_result: Dict[str, Any], 
        complexity: TaskComplexity
    ) -> List[Dict[str, Any]]:
        """Generate logical plan steps based on request analysis."""
        steps = []
        
        # Always start with analysis
        steps.append({
            "id": 1,
            "action": "analyze_request",
            "description": "Thoroughly analyze the user request and gather relevant context",
            "agent": "orchestrator",
            "estimated_time": 2,
            "dependencies": []
        })
        
        # Add domain-specific steps based on intent
        if intent_result.get('agent_type'):
            agent_type = intent_result['agent_type']
            
            if agent_type == AgentType.PRODUCTIVITY:
                steps.extend([
                    {
                        "id": 2,
                        "action": "assess_productivity_needs",
                        "description": "Analyze productivity requirements and current state",
                        "agent": "productivity",
                        "estimated_time": 5,
                        "dependencies": [1]
                    },
                    {
                        "id": 3,
                        "action": "create_productivity_plan",
                        "description": "Develop comprehensive productivity strategy",
                        "agent": "productivity", 
                        "estimated_time": 10,
                        "dependencies": [2]
                    }
                ])
            
            elif agent_type == AgentType.HEALTH:
                steps.extend([
                    {
                        "id": 2,
                        "action": "assess_health_goals",
                        "description": "Evaluate health and wellness objectives",
                        "agent": "health",
                        "estimated_time": 5,
                        "dependencies": [1]
                    },
                    {
                        "id": 3,
                        "action": "create_wellness_plan",
                        "description": "Develop personalized health and wellness strategy",
                        "agent": "health",
                        "estimated_time": 10,
                        "dependencies": [2]
                    }
                ])
            
            elif agent_type == AgentType.FINANCE:
                steps.extend([
                    {
                        "id": 2,
                        "action": "assess_financial_situation",
                        "description": "Analyze current financial state and goals",
                        "agent": "finance",
                        "estimated_time": 5,
                        "dependencies": [1]
                    },
                    {
                        "id": 3,
                        "action": "create_financial_plan",
                        "description": "Develop comprehensive financial strategy",
                        "agent": "finance",
                        "estimated_time": 10,
                        "dependencies": [2]
                    }
                ])
        
        # Add synthesis step for complex tasks
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.ADVANCED]:
            steps.append({
                "id": len(steps) + 1,
                "action": "synthesize_results",
                "description": "Combine insights from all agents into coherent recommendations",
                "agent": "orchestrator",
                "estimated_time": 5,
                "dependencies": list(range(2, len(steps) + 1))
            })
        
        return steps

    def _identify_required_agents(
        self, 
        user_input: str, 
        intent_result: Dict[str, Any]
    ) -> List[str]:
        """Identify which agents are required for the request."""
        agents = ["orchestrator"]  # Always include orchestrator
        
        # Add primary agent
        if intent_result.get('agent_type'):
            agents.append(intent_result['agent_type'].value)
        
        # Check for multi-domain requests
        user_input_lower = user_input.lower()
        
        # Check for cross-domain keywords
        if any(word in user_input_lower for word in ['budget', 'financial', 'money', 'cost']):
            if 'finance' not in agents:
                agents.append('finance')
        
        if any(word in user_input_lower for word in ['health', 'wellness', 'fitness', 'habit']):
            if 'health' not in agents:
                agents.append('health')
        
        if any(word in user_input_lower for word in ['task', 'goal', 'productivity', 'work']):
            if 'productivity' not in agents:
                agents.append('productivity')
        
        if any(word in user_input_lower for word in ['schedule', 'calendar', 'meeting', 'time']):
            if 'scheduling' not in agents:
                agents.append('scheduling')
        
        if any(word in user_input_lower for word in ['reflect', 'journal', 'mood', 'feeling']):
            if 'journal' not in agents:
                agents.append('journal')
        
        return agents

    def _define_success_criteria(self, user_input: str, complexity: TaskComplexity) -> List[str]:
        """Define success criteria based on request and complexity."""
        criteria = [
            "User request is fully understood and addressed",
            "Response is actionable and practical"
        ]
        
        if complexity == TaskComplexity.MODERATE:
            criteria.extend([
                "Domain expertise is properly applied",
                "Recommendations are personalized and relevant"
            ])
        
        elif complexity in [TaskComplexity.COMPLEX, TaskComplexity.ADVANCED]:
            criteria.extend([
                "All relevant domains are considered",
                "Plan is comprehensive and well-structured",
                "Integration between different aspects is seamless",
                "Long-term implications are considered"
            ])
        
        if complexity == TaskComplexity.ADVANCED:
            criteria.extend([
                "Strategic implications are addressed",
                "Stakeholder considerations are included",
                "Risk mitigation strategies are provided"
            ])
        
        return criteria

    def _estimate_duration(self, complexity: TaskComplexity) -> int:
        """Estimate duration in minutes based on complexity."""
        duration_map = {
            TaskComplexity.SIMPLE: 2,
            TaskComplexity.MODERATE: 5,
            TaskComplexity.COMPLEX: 15,
            TaskComplexity.ADVANCED: 30
        }
        return duration_map.get(complexity, 5)

    def _identify_risks(self, complexity: TaskComplexity) -> List[str]:
        """Identify potential risks based on complexity."""
        risks = []
        
        if complexity == TaskComplexity.MODERATE:
            risks = [
                "May require additional context for optimal results",
                "User preferences might need clarification"
            ]
        
        elif complexity == TaskComplexity.COMPLEX:
            risks = [
                "Multiple domains may have conflicting recommendations",
                "Integration complexity may require iteration",
                "May need user feedback between steps"
            ]
        
        elif complexity == TaskComplexity.ADVANCED:
            risks = [
                "Strategic decisions may require human oversight",
                "Long-term implications need careful consideration",
                "May require approval workflow for implementation",
                "Stakeholder alignment may be necessary"
            ]
        
        return risks

    async def _store_plan_context(self, deep_state: DeepAgentState, plan: Dict[str, Any]) -> None:
        """Store plan in deep state for context preservation."""
        try:
            plan_filename = f"strategic_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            plan_content = json.dumps(plan, indent=2)
            
            # Store in files
            if "files" not in deep_state:
                deep_state["files"] = {}
            deep_state["files"][plan_filename] = plan_content
            
            # Create todos from plan steps
            if "todos" not in deep_state:
                deep_state["todos"] = []
            
            for step in plan.get("steps", []):
                todo = Todo(
                    id=len(deep_state["todos"]) + 1,
                    title=step["action"],
                    description=step["description"],
                    status="not_started",
                    priority="medium",
                    assigned_agent=step.get("agent", "orchestrator"),
                    estimated_time=step.get("estimated_time", 5),
                    dependencies=step.get("dependencies", [])
                )
                deep_state["todos"].append(todo.model_dump())
            
            logger.info("plan_stored", f"Stored strategic plan with {len(plan.get('steps', []))} steps", {"step_count": len(plan.get('steps', []))})
            
        except Exception as e:
            logger.error("store_plan_error", "Error storing plan context", error=e)

    async def _handle_simple_task(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle simple tasks directly without delegation."""
        try:
            if not self.llm_service:
                self.llm_service = get_llm_service()
            
            # Create direct response using LLM
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_input)
            ]
            
            request = CompletionRequest(
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            response = await self.llm_service.complete(request)
            return response.content
            
        except Exception as e:
            logger.error("simple_task_error", "Error handling simple task", error=e)
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

    async def _delegate_to_specialist(
        self, 
        agent_type: AgentType, 
        user_input: str, 
        context: Dict[str, Any],
        deep_state: DeepAgentState
    ) -> str:
        """Delegate to specialized ReAct agent."""
        try:
            # Get the appropriate ReAct agent
            if agent_type == AgentType.PRODUCTIVITY:
                react_agent = self.react_factory.create_productivity_agent()
            elif agent_type == AgentType.HEALTH:
                react_agent = self.react_factory.create_health_agent()
            elif agent_type == AgentType.FINANCE:
                react_agent = self.react_factory.create_finance_agent()
            else:
                # Fallback to direct handling
                return await self._handle_simple_task(user_input, context)
            
            # Execute the ReAct agent with deep state
            agent_state = {
                "messages": [HumanMessage(content=user_input)],
                "files": deep_state.get("files", {}),
                "todos": deep_state.get("todos", []),
                "agent_context": deep_state.get("agent_context", {}),
                "metadata": deep_state.get("metadata", {})
            }
            
            result = await react_agent.agent.ainvoke(agent_state)
            
            # Extract response from result
            if isinstance(result, dict):
                if "messages" in result and result["messages"]:
                    last_message = result["messages"][-1]
                    if hasattr(last_message, 'content'):
                        response = last_message.content
                    else:
                        response = str(last_message)
                else:
                    response = str(result)
            else:
                response = str(result)
            
            # Update deep state with results
            if isinstance(result, dict):
                deep_state.update({
                    "files": result.get("files", deep_state.get("files", {})),
                    "todos": result.get("todos", deep_state.get("todos", [])),
                    "agent_context": result.get("agent_context", deep_state.get("agent_context", {}))
                })
            
            return response
            
        except Exception as e:
            logger.error("delegation_error", f"Error delegating to {agent_type.value} agent", {"agent_type": agent_type.value}, error=e)
            return f"I apologize, but I encountered an error while working with the {agent_type.value} specialist: {str(e)}"

    async def _orchestrate_complex_workflow(
        self, 
        plan: Dict[str, Any], 
        user_input: str, 
        context: Dict[str, Any],
        deep_state: DeepAgentState,
        complexity: TaskComplexity
    ) -> str:
        """Orchestrate complex multi-agent workflow."""
        try:
            if not plan:
                return await self._handle_simple_task(user_input, context)
            
            results = []
            executed_steps = []
            
            # Execute plan steps
            for step in plan.get("steps", []):
                step_id = step["id"]
                agent_name = step.get("agent", "orchestrator")
                action = step["action"]
                description = step["description"]
                dependencies = step.get("dependencies", [])
                
                # Check if dependencies are met
                if dependencies and not all(dep_id in [s["id"] for s in executed_steps] for dep_id in dependencies):
                    logger.warning("step_skipped", f"Skipping step {step_id}: dependencies not met", {"step_id": step_id})
                    continue
                
                logger.info("step_executing", f"Executing step {step_id}: {action}", {"step_id": step_id, "action": action, "agent": agent_name})
                
                try:
                    if agent_name == "orchestrator":
                        step_result = await self._execute_orchestrator_step(step, user_input, context)
                    else:
                        step_result = await self._execute_agent_step(step, user_input, deep_state)
                    
                    results.append({
                        "step": step_id,
                        "action": action,
                        "agent": agent_name,
                        "result": step_result,
                        "status": "completed"
                    })
                    
                    executed_steps.append(step)
                    
                except Exception as e:
                    logger.error("step_execution_error", f"Error executing step {step_id}", {"step_id": step_id}, error=e)
                    results.append({
                        "step": step_id,
                        "action": action,
                        "agent": agent_name,
                        "result": f"Error: {str(e)}",
                        "status": "failed"
                    })
            
            # Synthesize final response
            synthesis = await self._synthesize_workflow_results(plan, results, user_input)
            
            # Store workflow results in deep state
            workflow_filename = f"workflow_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            workflow_data = {
                "plan": plan,
                "executed_steps": executed_steps,
                "results": results,
                "synthesis": synthesis,
                "timestamp": datetime.now().isoformat()
            }
            
            if "files" not in deep_state:
                deep_state["files"] = {}
            deep_state["files"][workflow_filename] = json.dumps(workflow_data, indent=2)
            
            return synthesis
            
        except Exception as e:
            logger.error("workflow_error", "Error orchestrating complex workflow", error=e)
            return f"I apologize, but I encountered an error while coordinating the workflow: {str(e)}"

    async def _execute_orchestrator_step(
        self, 
        step: Dict[str, Any], 
        user_input: str, 
        context: Dict[str, Any]
    ) -> str:
        """Execute orchestrator-specific step."""
        action = step["action"]
        
        if action == "analyze_request":
            return f"Analyzed request: '{user_input}' - Identified as {step.get('description', 'complex task requiring coordination')}"
        
        elif action == "synthesize_results":
            return "Synthesizing results from all agents to provide comprehensive response"
        
        else:
            return f"Executed orchestrator action: {action}"

    async def _execute_agent_step(
        self, 
        step: Dict[str, Any], 
        user_input: str, 
        deep_state: DeepAgentState
    ) -> str:
        """Execute step using specialized agent."""
        agent_name = step.get("agent")
        action = step["action"]
        description = step["description"]
        
        # Map agent names to types
        agent_type_map = {
            "productivity": AgentType.PRODUCTIVITY,
            "health": AgentType.HEALTH,
            "finance": AgentType.FINANCE,
            "scheduling": AgentType.SCHEDULING,
            "journal": AgentType.JOURNAL
        }
        
        agent_type = agent_type_map.get(agent_name)
        if not agent_type:
            return f"Unknown agent type: {agent_name}"
        
        # Create context-specific prompt for this step
        step_prompt = f"""Focus specifically on this task: {description}

Original user request: {user_input}
Specific action needed: {action}

Please provide a focused response addressing this specific aspect of the user's request."""
        
        # Delegate to specialist
        result = await self._delegate_to_specialist(agent_type, step_prompt, {}, deep_state)
        
        return result

    async def _synthesize_workflow_results(
        self, 
        plan: Dict[str, Any], 
        results: List[Dict[str, Any]], 
        user_input: str
    ) -> str:
        """Synthesize results from workflow execution."""
        try:
            # Create synthesis prompt
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
Status: {result['status']}
Result: {result['result']}
"""
            
            synthesis_prompt += """

Please create a coherent, comprehensive response that:
1. Addresses the original user request completely
2. Integrates insights from all agents
3. Provides actionable recommendations
4. Maintains a helpful and clear tone
5. Acknowledges any limitations or next steps needed

Focus on value and practical guidance for the user."""
            
            # Use LLM to synthesize
            if not self.llm_service:
                self.llm_service = get_llm_service()
            
            messages = [
                ChatMessage(role="system", content="You are an expert at synthesizing complex information into clear, actionable guidance."),
                ChatMessage(role="user", content=synthesis_prompt)
            ]
            
            request = CompletionRequest(
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response = await self.llm_service.complete(request)
            return response.content
            
        except Exception as e:
            logger.error("synthesis_error", "Error synthesizing workflow results", error=e)
            
            # Fallback to simple concatenation
            successful_results = [r for r in results if r['status'] == 'completed']
            
            if successful_results:
                synthesis = f"Based on your request '{user_input}', here's what I've found:\\n\\n"
                for result in successful_results:
                    synthesis += f"**{result['action'].replace('_', ' ').title()}:** {result['result']}\\n\\n"
                return synthesis
            else:
                return "I apologize, but I wasn't able to complete the workflow successfully. Please try a simpler request or contact support."

    async def _store_interaction_insights(
        self, 
        deep_state: DeepAgentState, 
        user_input: str, 
        response: str, 
        reasoning: Dict[str, Any]
    ) -> None:
        """Store valuable insights from the interaction."""
        try:
            # Create insights entry
            insights = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "approach": reasoning.get("approach"),
                "complexity": reasoning.get("complexity"),
                "agents_involved": reasoning.get("agents_involved", []),
                "success_indicators": {
                    "response_length": len(response),
                    "approach_used": reasoning.get("approach"),
                    "error_occurred": "error" in reasoning
                }
            }
            
            # Store in insights file
            insights_filename = "interaction_insights.json"
            if "files" not in deep_state:
                deep_state["files"] = {}
            
            if insights_filename in deep_state["files"]:
                try:
                    existing_insights = json.loads(deep_state["files"][insights_filename])
                    if "interactions" not in existing_insights:
                        existing_insights["interactions"] = []
                except json.JSONDecodeError:
                    existing_insights = {"interactions": []}
            else:
                existing_insights = {"interactions": []}
            
            existing_insights["interactions"].append(insights)
            
            # Keep only last 50 interactions
            if len(existing_insights["interactions"]) > 50:
                existing_insights["interactions"] = existing_insights["interactions"][-50:]
            
            deep_state["files"][insights_filename] = json.dumps(existing_insights, indent=2)
            
        except Exception as e:
            logger.error("insights_storage_error", "Error storing interaction insights", error=e)


# Create global instance for use in workflows
_enhanced_orchestrator = None

def get_enhanced_orchestrator() -> EnhancedOrchestratorAgent:
    """Get global enhanced orchestrator instance."""
    global _enhanced_orchestrator
    if _enhanced_orchestrator is None:
        _enhanced_orchestrator = EnhancedOrchestratorAgent()
    return _enhanced_orchestrator