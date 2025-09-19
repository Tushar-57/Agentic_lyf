"""
AI Agent Ecosystem - Core Application Package

This package contains the core components of the AI agent ecosystem including:
- Agents: Specialized AI agents with domain expertise
- API: REST API endpoints and handlers  
- LLM: Language model providers and services
- Models: Data models and schemas
- Services: Business logic and data management
- Utils: Utility functions and helpers
"""

# Version information
__version__ = "1.0.0"
__author__ = "AI Agent Ecosystem Team"

# Core imports for easy access (avoiding circular imports at package level)
# Users should import these directly from their modules:
# from app.agents import BaseAgent, AgentType
# from app.llm import get_llm_service  
# from app.services import get_knowledge_base_service

__all__ = [
    # Core package information
    "__version__",
    "__author__",
]