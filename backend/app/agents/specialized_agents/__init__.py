"""
Specialized Agents Module

This module contains domain-specific agent implementations:
- HealthAgent: Health, wellness, and nutrition management
- ProductivityAgent: Task management and productivity optimization
- FinanceAgent: Financial planning and expense tracking
- SchedulingAgent: Calendar and appointment management
- JournalAgent: Personal journaling and reflection
"""

from .health import HealthAgent
from .productivity import ProductivityAgent
from .finance import FinanceAgent
from .scheduling import SchedulingAgent
from .journal import JournalAgent

__all__ = [
    'HealthAgent',
    'ProductivityAgent', 
    'FinanceAgent',
    'SchedulingAgent',
    'JournalAgent'
]