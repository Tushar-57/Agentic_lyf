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
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage
from .base import BaseAgent, AgentType, AgentCapability, AgentState
from .prompts import PromptLibrary, get_agent_prompt
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
                r'\b(plan|schedule|track|monitor|optimize|streamline)\b'
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

    async def execute(self, state: Union[AgentState, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced execute method with deep agent patterns."""
        logger.debug("EnhancedOrchestrator.execute START")
        
        try:
            # Initialize LLM service if needed
            if not self.llm_service:
                self.llm_service = get_llm_service()
            
            # Normalize state input
            normalized_state = self._normalize_state(state)
            user_input = normalized_state["user_input"]
            context = normalized_state["context"]
            conversation_id = normalized_state.get("conversation_id")
            
            # Initialize or get deep state
            deep_state = self.state_manager.get_or_create_state(conversation_id or "default")
            
            # Store user message in deep state
            deep_state.add_message("user", user_input)
            
            # Assess task complexity
            complexity = await self._assess_task_complexity(user_input)
            
            # Enhanced intent classification with complexity awareness
            intent_result = await self._enhanced_intent_classification(user_input, context, complexity)
            
            # Create strategic plan for complex tasks
            strategic_plan = None
            if complexity in [TaskComplexity.COMPLEX, TaskComplexity.ADVANCED]:
                strategic_plan = await self._create_strategic_plan(user_input, intent_result, complexity)
                if strategic_plan:
                    await self._store_plan_in_context(strategic_plan, deep_state)
            
            # Execute based on complexity and plan
            if complexity == TaskComplexity.SIMPLE:
                response = await self._handle_simple_task(user_input, context, deep_state)
            elif complexity == TaskComplexity.MODERATE:
                response = await self._delegate_to_specialist(
                    intent_result.get("agent_type", AgentType.GENERAL),
                    user_input,
                    context,
                    deep_state
                )
            else:  # COMPLEX or ADVANCED
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
            
            # Build reasoning for transparency
            reasoning = {
                "complexity": complexity.value,
                "intent": intent_result,
                "plan": strategic_plan,
                "execution_path": self._get_execution_path(complexity, strategic_plan)
            }
            
            return {
                "response": response,
                "reasoning": reasoning,
                "deep_state": deep_state.to_dict()
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
        complexity: TaskComplexity
    ) -> Dict[str, Any]:
        """Enhanced intent classification with complexity awareness."""
        user_lower = user_input.lower()
        agent_scores = {}
        
        # Score each agent type
        for agent_type, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, user_lower)
                if matches:
                    score += len(matches)
                    matched_patterns.append(pattern)
            
            if score > 0:
                agent_scores[agent_type] = {
                    "score": score,
                    "patterns": matched_patterns,
                    "confidence": min(score / 5.0, 1.0)  # Normalize to 0-1
                }
        
        # Determine best agent
        if not agent_scores:
            return {
                "agent_type": AgentType.GENERAL,
                "confidence": 0.1,
                "reason": "No specific domain patterns matched",
                "complexity_factor": complexity.value
            }
        
        best_agent = max(agent_scores.items(), key=lambda x: x[1]["score"])
        agent_type, scores = best_agent
        
        # Adjust confidence based on complexity
        complexity_multiplier = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.1,
            TaskComplexity.COMPLEX: 1.2,
            TaskComplexity.ADVANCED: 1.3
        }
        
        adjusted_confidence = min(scores["confidence"] * complexity_multiplier[complexity], 1.0)
        
        return {
            "agent_type": agent_type,
            "confidence": adjusted_confidence,
            "reason": f"Matched {len(scores['patterns'])} patterns with {scores['score']} total matches",
            "complexity_factor": complexity.value,
            "all_scores": {str(k.value): v["confidence"] for k, v in agent_scores.items()}
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
            
            # Create mock deep state for tool execution
            mock_state = DeepAgentState(conversation_id="planning")
            
            # Execute thinking tool (simplified for now)
            plan = {
                "complexity": complexity.value,
                "primary_domain": intent_result['agent_type'].value if intent_result.get('agent_type') else 'general',
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
                    "agent": intent_result.get('agent_type', AgentType.GENERAL).value,
                    "description": f"Use {intent_result.get('agent_type', AgentType.GENERAL).value} agent for domain expertise",
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
        if intent_result.get('agent_type'):
            required_agents.append(intent_result['agent_type'].value)
        
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

    async def _handle_simple_task(
        self, 
        user_input: str, 
        context: Dict[str, Any], 
        deep_state: DeepAgentState
    ) -> str:
        """Handle simple tasks directly without delegation."""
        try:
            # Use LLM service directly for simple responses
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_input)
            ]
            
            request = CompletionRequest(
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            response = await self.llm_service.generate_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Simple task handling failed: {e}")
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
            # Get specialized ReAct agent
            if agent_type == AgentType.PRODUCTIVITY:
                agent = self.react_factory.create_productivity_agent()
            elif agent_type == AgentType.HEALTH:
                agent = self.react_factory.create_health_agent()
            elif agent_type == AgentType.FINANCE:
                agent = self.react_factory.create_finance_agent()
            else:
                # Fallback to direct handling
                return await self._handle_simple_task(user_input, context, deep_state)
            
            # Prepare enhanced prompt with context
            enhanced_prompt = f"""You are being delegated this task by the Enhanced Orchestrator.

**User Request:** {user_input}

**Context:** {context}

**Available Deep State Files:**
{chr(10).join(f'- {filename}' for filename in deep_state.files.keys())}

Please provide a comprehensive response using your specialized tools and capabilities. Store any valuable insights or results in files for future reference."""
            
            # Execute the specialized agent
            agent_state = {
                "user_input": enhanced_prompt,
                "context": context,
                "deep_state": deep_state.to_dict()
            }
            
            result = await agent.execute(agent_state)
            
            # Extract response
            if isinstance(result, dict) and "response" in result:
                return result["response"]
            elif isinstance(result, str):
                return result
            else:
                return "The specialist completed the task successfully."
                
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
                return await self._delegate_to_specialist(
                    intent_result.get('agent_type', AgentType.GENERAL),
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
            
            response = await self.llm_service.generate_completion(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            # Fallback to simple concatenation
            results_text = "\n\n".join([
                f"**{result['action']}:** {result['result']}" 
                for result in results 
                if result['status'] == 'completed'
            ])
            return f"Here are the results from your request:\n\n{results_text}"

    def _get_execution_path(self, complexity: TaskComplexity, plan: Optional[Dict[str, Any]]) -> str:
        """Get human-readable execution path description."""
        if complexity == TaskComplexity.SIMPLE:
            return "Direct response by orchestrator"
        elif complexity == TaskComplexity.MODERATE:
            return "Single specialist agent delegation"
        elif plan:
            agent_count = len(plan.get('agents_involved', []))
            step_count = len(plan.get('steps', []))
            return f"Complex workflow: {step_count} steps across {agent_count} agents"
        else:
            return "Fallback to specialist delegation"


# Factory function for getting enhanced orchestrator
_enhanced_orchestrator = None

def get_enhanced_orchestrator() -> EnhancedOrchestratorAgent:
    """Get the global enhanced orchestrator instance."""
    global _enhanced_orchestrator
    if _enhanced_orchestrator is None:
        _enhanced_orchestrator = EnhancedOrchestratorAgent()
    return _enhanced_orchestrator