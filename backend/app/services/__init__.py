"""
Services Module - Business Logic and Data Management

This module contains services for:
- Knowledge base management and operations
- Vector storage and similarity search
- Data persistence and retrieval
"""

from .knowledge_base import get_knowledge_base_service, KnowledgeBaseService
from .interaction_recorder import get_interaction_recorder, InteractionRecorder
from .vector_store import VectorStore

__all__ = [
    "get_knowledge_base_service",
    "KnowledgeBaseService",
    "VectorStore",
]