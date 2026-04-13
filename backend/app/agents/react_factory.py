"""
ReAct Agent Factory for Deep Agents
===================================

Factory for creating ReAct-powered sub-agents with isolated contexts,
specialized tool sets, and deep agent capabilities.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool

from .deep_state import DeepAgentState, DeepAgentStateManager
from .file_tools import create_file_tools
from .todo_tools import create_todo_tools  
from .think_tools import create_think_tools
from .base import AgentType
from ..llm.service import get_llm_service
from .specialized_productivity_tools import create_productivity_tools
from .specialized_finance_tools import create_finance_tools
from .specialized_health_tools import create_health_tools

logger = logging.getLogger(__name__)


class ReActSubAgent:
    """
    A ReAct-powered sub-agent with isolated context and specialized capabilities.
    
    Each sub-agent operates with:
    - Isolated state and context
    - Specialized tool set for their domain
    - File-based context storage
    - TODO management capabilities
    - Strategic thinking tools
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        name: str,
        description: str,
        system_prompt: str,
        specialized_tools: List[BaseTool],
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 10
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.specialized_tools = specialized_tools
        self.max_iterations = max_iterations
        
        # Initialize LLM
        self.model = init_chat_model(model=model_name, temperature=0.1)
        
        # Create comprehensive tool set
        self.tools = self._create_tool_set()
        
        # Create the ReAct agent  
        # Note: create_react_agent uses its own internal state schema
        self.react_agent = create_react_agent(
            self.model,
            self.tools,
            prompt=self._build_full_prompt()
        )
        
        logger.info("react_agent_created", f"Created ReAct sub-agent: {self.name} ({self.agent_id})", {"agent_name": self.name, "agent_id": self.agent_id})
    
    def _create_tool_set(self) -> List[BaseTool]:
        """Create the complete tool set for this agent."""
        tools = []
        
        # Add core deep agent tools
        tools.extend(create_file_tools())
        tools.extend(create_todo_tools())
        tools.extend(create_think_tools())
        
        # Add specialized tools for this agent
        tools.extend(self.specialized_tools)
        
        return tools
    
    def _build_full_prompt(self) -> str:
        """Build the complete system prompt with instructions."""
        base_instructions = f"""You are {self.name}, a specialized ReAct agent in an AI ecosystem.

{self.system_prompt}

## Your Capabilities

You have access to powerful tools for:

### 📁 File Management
- **write_file**: Store detailed results, analysis, or context
- **read_file**: Retrieve previously stored information
- **list_files**: See what information is available
- **search_files**: Find specific content across files

### 📋 TODO Management  
- **create_todo**: Break down tasks and plan work
- **list_todos**: Review current tasks and priorities
- **update_todo_status**: Track progress on tasks
- **breakdown_task**: Decompose complex work into steps

### 🤔 Strategic Thinking
- **think_tool**: Reflect on progress and plan next steps
- **analyze_progress**: Review overall status and priorities
- **plan_next_steps**: Create structured plans for moving forward
- **decision_matrix**: Systematically evaluate options

### 🔧 Specialized Tools
You also have domain-specific tools for your area of expertise.

## Operating Principles

1. **Context Offloading**: Store detailed information in files, return summaries
2. **Strategic Planning**: Use TODO tools to break down complex requests
3. **Regular Reflection**: Use think_tool to analyze progress and plan next steps
4. **Isolated Context**: Your work is separate from other agents - use files to share

## Workflow Pattern

For complex requests:
1. **Plan**: Create TODOs or use plan_next_steps
2. **Execute**: Use specialized tools to gather information/perform actions
3. **Store**: Save detailed results to files
4. **Reflect**: Use think_tool to analyze findings
5. **Summarize**: Return concise summary with file references

## Communication Style

- Be helpful and professional
- Provide clear, actionable responses
- Reference stored files when relevant
- Explain your reasoning and next steps
- Ask for clarification when needed

Your goal is to provide excellent specialized assistance while maintaining clean context and systematic progress tracking."""

        return base_instructions
    
    async def execute(self, request: str, state: Optional[DeepAgentState] = None) -> Dict[str, Any]:
        """Execute a request using the ReAct agent."""
        # Create state if not provided
        if state is None:
            state_manager = DeepAgentStateManager()
            state = state_manager.get_state()
        
        # Set current agent in state
        state["current_agent"] = self.agent_id
        
        # Ensure agent context exists
        if self.agent_id not in state.get("agent_contexts", {}):
            state["agent_contexts"][self.agent_id] = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "status": "executing",
                "current_task": request[:100],
                "tools_used": [],
                "files_created": [],
                "todos_assigned": [],
                "execution_history": [],
                "error_count": 0,
                "last_activity": datetime.now().isoformat()
            }
        
        try:
            # Execute the ReAct agent
            result = await self.react_agent.ainvoke({
                **state,
                "messages": [HumanMessage(content=request)]
            })
            
            # Update agent status
            state["agent_contexts"][self.agent_id]["status"] = "completed"
            state["agent_contexts"][self.agent_id]["last_activity"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "result": result,
                "state": result  # The updated state from the agent
            }
            
        except Exception as e:
            logger.error("react_agent_execution_failed", f"Error executing ReAct agent {self.agent_id}: {str(e)}", {"agent_id": self.agent_id, "error": str(e)})
            
            # Update error count
            state["agent_contexts"][self.agent_id]["error_count"] += 1
            state["agent_contexts"][self.agent_id]["status"] = "error"
            
            return {
                "success": False,
                "error": str(e),
                "state": state
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "description": self.description,
            "tool_count": len(self.tools),
            "specialized_tools": [tool.name for tool in self.specialized_tools]
        }


class ReActAgentFactory:
    """Factory for creating ReAct sub-agents with standardized configurations."""
    
    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model
        self.agents: Dict[str, ReActSubAgent] = {}
        
    def create_agent(
        self,
        agent_id: str,
        agent_type: AgentType,
        name: str,
        description: str,
        system_prompt: str,
        specialized_tools: List[BaseTool],
        model_name: Optional[str] = None,
        max_iterations: int = 10
    ) -> ReActSubAgent:
        """Create a new ReAct sub-agent."""
        agent = ReActSubAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            name=name,
            description=description,
            system_prompt=system_prompt,
            specialized_tools=specialized_tools,
            model_name=model_name or self.default_model,
            max_iterations=max_iterations
        )
        
        self.agents[agent_id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[ReActSubAgent]:
        """Get an existing agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all created agents."""
        return [agent.get_info() for agent in self.agents.values()]
    
    def create_health_agent(self) -> ReActSubAgent:
        """Create a specialized Health ReAct agent."""
        
        system_prompt = """You are the Health Agent, a specialized ReAct agent focused on health, wellness, and nutrition management.

Your expertise includes:
- Habit formation and tracking
- Meal planning and nutrition guidance  
- Wellness coaching and motivation
- Exercise and fitness planning
- Sleep optimization
- Mental health and stress management

You help users build sustainable healthy lifestyles through:
- Personalized recommendations based on their preferences and goals
- Evidence-based health advice
- Supportive and non-judgmental guidance
- Practical, actionable strategies
- Progress tracking and motivation

Always prioritize user safety and well-being. For serious health concerns, recommend consulting healthcare professionals."""

        specialized_tools = create_health_tools()
        
        return self.create_agent(
            agent_id="health_react_agent",
            agent_type=AgentType.HEALTH,
            name="Health ReAct Agent",
            description="Specialized agent for health, wellness, and nutrition management",
            system_prompt=system_prompt,
            specialized_tools=specialized_tools
        )
    
    def create_finance_agent(self) -> ReActSubAgent:
        """Create a specialized Finance ReAct agent."""
        
        system_prompt = """You are the Finance Agent, a specialized ReAct agent focused on personal finance management and financial planning.

Your expertise includes:
- Expense tracking and budget management
- Financial goal setting and planning
- Investment guidance and portfolio management
- Debt management and optimization
- Savings strategies and planning
- Financial risk assessment

You help users achieve financial wellness through:
- Practical budgeting and spending analysis
- Goal-oriented financial planning
- Investment education and guidance
- Debt reduction strategies
- Emergency fund planning
- Long-term wealth building

Always provide responsible financial advice and encourage users to consult financial professionals for complex situations."""

        specialized_tools = create_finance_tools()
        
        return self.create_agent(
            agent_id="finance_react_agent", 
            agent_type=AgentType.FINANCE,
            name="Finance ReAct Agent",
            description="Specialized agent for personal finance management and planning",
            system_prompt=system_prompt,
            specialized_tools=specialized_tools
        )
    
    def create_productivity_agent(self) -> ReActSubAgent:
        """Create a specialized Productivity ReAct agent."""
        
        system_prompt = """You are the Productivity Agent, a specialized ReAct agent focused on task management, goal achievement, and productivity optimization.

Your expertise includes:
- Task organization and prioritization
- Goal setting and tracking
- Time management and scheduling
- Workflow optimization
- Focus and concentration strategies
- Productivity analytics and insights

You help users achieve their goals through:
- Effective task and project management
- SMART goal setting and breakdown
- Time blocking and scheduling strategies
- Productivity tool recommendations
- Habit formation for productive behaviors
- Performance analysis and optimization

Focus on practical, sustainable approaches that fit the user's lifestyle and work style."""

        specialized_tools = create_productivity_tools()
        
        return self.create_agent(
            agent_id="productivity_react_agent",
            agent_type=AgentType.PRODUCTIVITY, 
            name="Productivity ReAct Agent",
            description="Specialized agent for task management and productivity optimization",
            system_prompt=system_prompt,
            specialized_tools=specialized_tools
        )


# Global factory instance
_react_agent_factory = None

def get_react_agent_factory() -> ReActAgentFactory:
    """Get the global ReAct agent factory instance."""
    global _react_agent_factory
    if _react_agent_factory is None:
        _react_agent_factory = ReActAgentFactory()
    return _react_agent_factory