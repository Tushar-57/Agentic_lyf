"""
API Module - REST API Endpoints and Handlers

This module provides REST API endpoints for:
- Knowledge base operations (CRUD, search, etc.)
- Agent management and interaction
- System health and status checks
"""

from .knowledge import router as knowledge_router

__all__ = [
    "knowledge_router",
]