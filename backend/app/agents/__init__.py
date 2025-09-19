"""
Agent Module - AI Agent Implementations and Management

This module contains:
- BaseAgent: Abstract base class for all agents
- AgentType: Enumeration of available agent types
- Specialized agents: Domain-specific agent implementations
- Agent factory: Creation and management of agents
- Agent registry: Registration and discovery of agents
- Communication protocols: Inter-agent communication
"""

from .base import BaseAgent, AgentType, AgentCapability, AgentStatus, AgentMessage, AgentState
from .registry import get_agent_registry, reset_agent_registry
from .factory import AgentFactory
from .orchestrator import OrchestratorAgent, create_orchestrator_agent
from .specialized_agents import HealthAgent, ProductivityAgent, FinanceAgent, SchedulingAgent, JournalAgent
from .prompts import get_agent_prompt, PromptLibrary
from .communication import get_communication_protocol, start_communication_protocol, MessageType

__all__ = [
    # Core classes
    "BaseAgent",
    "AgentType", 
    "AgentCapability",
    "AgentStatus",
    "AgentMessage",
    "AgentState",
    
    # Registry and factory
    "get_agent_registry",
    "reset_agent_registry",
    "AgentFactory",
    
    # Orchestrator
    "OrchestratorAgent",
    "create_orchestrator_agent",
    
    # Specialized agents
    "HealthAgent",
    "ProductivityAgent", 
    "FinanceAgent",
    "SchedulingAgent",
    "JournalAgent",
    
    # Prompts and communication
    "get_agent_prompt",
    "PromptLibrary",
    "get_communication_protocol",
    "start_communication_protocol",
    "MessageType",
]