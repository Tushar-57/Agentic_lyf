"""
LLM Module - Language Model Providers and Services

This module contains:
- Base classes and interfaces for LLM providers
- OpenAI provider implementation
- Ollama provider implementation
- LLM service management and configuration
- Factory for creating LLM providers
"""

from .base import BaseLLMProvider, CompletionRequest, CompletionResponse, ChatMessage
from .config import LLMConfig
from .factory import LLMProviderFactory
from .service import get_llm_service, LLMService
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "CompletionRequest", 
    "CompletionResponse",
    "ChatMessage",
    
    # Configuration
    "LLMConfig",
    
    # Factory and service
    "LLMProviderFactory",
    "get_llm_service",
    "LLMService",
    
    # Providers
    "OpenAIProvider",
    "OllamaProvider",
]