"""
LLM service that provides a high-level interface to the LLM providers.
"""

import os
from typing import AsyncGenerator, Dict, Any, List, Optional

from .base import (
    CompletionRequest, 
    CompletionResponse, 
    EmbeddingRequest, 
    EmbeddingResponse,
    HealthCheckResult,
    LLMProviderType
)
from .config import LLMConfig
from .factory import LLMProviderFactory
from ..utils.logging import get_llm_category_logger

logger = get_llm_category_logger(__name__)


class LLMService:
    """High-level service for LLM operations with automatic provider management."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            # Load configuration from environment variables
            config = LLMConfig.from_env(dict(os.environ))
        
        self.config = config
        self.factory = LLMProviderFactory(config)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the LLM service."""
        try:
            await self.factory.initialize()
            self._initialized = True
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    async def chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a chat completion using the active provider."""
        if not self._initialized:
            raise Exception("LLM service not initialized")
        
        try:
            provider = await self.factory.get_provider()
            return await provider.chat_completion(request)
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def chat_completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming chat completion using the active provider."""
        if not self._initialized:
            raise Exception("LLM service not initialized")
        
        try:
            provider = await self.factory.get_provider()
            async for chunk in provider.chat_completion_stream(request):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            raise
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using the active provider."""
        if not self._initialized:
            raise Exception("LLM service not initialized")
        
        try:
            provider = await self.factory.get_provider()
            return await provider.generate_embedding(request)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def health_check(self) -> Dict[LLMProviderType, HealthCheckResult]:
        """Get health status of all providers."""
        if not self._initialized:
            raise Exception("LLM service not initialized")
        
        return self.factory.get_health_status()
    
    async def switch_provider(self, provider_type: LLMProviderType, skip_health_check: bool = False) -> bool:
        """Switch to a specific provider."""
        if not self._initialized:
            raise Exception("LLM service not initialized")
        
        return await self.factory.switch_provider(provider_type, skip_health_check)
    
    def get_current_provider(self) -> Optional[LLMProviderType]:
        """Get the current active provider type."""
        if not self._initialized:
            return None
        
        return self.factory.get_current_provider_type()
    
    def get_available_models(self, provider_type: Optional[LLMProviderType] = None) -> List[str]:
        """Get available models for a provider."""
        if not self._initialized:
            return []
        
        try:
            if provider_type is None:
                provider_type = self.factory.get_current_provider_type()
            
            if provider_type in self.factory._providers:
                return self.factory._providers[provider_type].get_available_models()
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    async def update_config(self, new_config: LLMConfig) -> bool:
        """Update the service configuration and reinitialize if needed."""
        try:
            self.config = new_config
            self.factory = LLMProviderFactory(new_config)
            await self.factory.initialize()
            logger.info("LLM service configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update LLM service configuration: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the LLM service."""
        if self._initialized:
            await self.factory.shutdown()
            self._initialized = False
            logger.info("LLM service shutdown complete")


# Global service instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    
    if _llm_service is None:
        # Create service but don't initialize if no valid providers are configured
        config = LLMConfig.from_env(dict(os.environ))
        
        # Check if any provider has valid configuration
        has_valid_provider = (
            config.validate_provider_config(config.provider) or
            (config.fallback_enabled and config.fallback_provider and 
             config.validate_provider_config(config.fallback_provider))
        )
        
        if not has_valid_provider:
            logger.warning("No valid LLM provider configurations found - LLM service will be created but not initialized")
            _llm_service = LLMService(config)
            # Don't initialize - service will be available but not functional until provider is configured
        else:
            _llm_service = LLMService(config)
            await _llm_service.initialize()
    
    return _llm_service


async def shutdown_llm_service() -> None:
    """Shutdown the global LLM service instance."""
    global _llm_service
    
    if _llm_service is not None:
        await _llm_service.shutdown()
        _llm_service = None


async def reset_llm_service() -> LLMService:
    """Reset and reinitialize the global LLM service instance."""
    global _llm_service
    
    # Shutdown existing service
    if _llm_service is not None:
        await _llm_service.shutdown()
    
    # Create new service with fresh configuration
    _llm_service = LLMService()
    await _llm_service.initialize()
    
    return _llm_service