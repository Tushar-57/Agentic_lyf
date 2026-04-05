"""
LLM service that provides a high-level interface to the LLM providers.
"""

import os
from typing import AsyncGenerator, Dict, List, Optional

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


def _normalize_provider_exception(error: Exception) -> Exception:
    """Convert provider failures into a stable error shape for API handlers."""
    error_text = str(error).lower()

    if "llm_provider_unavailable" in error_text or "no healthy providers available" in error_text:
        return RuntimeError("LLM_PROVIDER_UNAVAILABLE: No healthy providers available")

    if "llm service not initialized" in error_text:
        return RuntimeError("LLM_PROVIDER_UNAVAILABLE: LLM service not initialized")

    return error


def _is_provider_unavailable_error(error: Exception) -> bool:
    """Check whether an exception represents provider unavailability."""
    error_text = str(error).lower()
    return (
        "llm_provider_unavailable" in error_text
        or "no healthy providers available" in error_text
        or "llm service not initialized" in error_text
    )


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

    def has_valid_provider_config(self, config: Optional[LLMConfig] = None) -> bool:
        """Return True when at least one configured provider can be initialized."""
        effective_config = config or self.config
        return (
            effective_config.validate_provider_config(effective_config.provider)
            or (
                effective_config.fallback_enabled
                and effective_config.fallback_provider
                and effective_config.validate_provider_config(effective_config.fallback_provider)
            )
        )

    async def _reload_from_environment(self) -> bool:
        """Reload LLM config from env/storage and reinitialize provider factory."""
        refreshed_config = LLMConfig.from_env(dict(os.environ))
        self.config = refreshed_config
        self.factory = LLMProviderFactory(refreshed_config)
        self._initialized = False

        if not self.has_valid_provider_config(refreshed_config):
            logger.warning("No valid provider config available after reload")
            return False

        try:
            await self.factory.initialize()
            self._initialized = True
            logger.info("LLM service reloaded from latest configuration")
            return True
        except Exception as reload_error:
            logger.error(f"Failed to reload LLM service from config: {reload_error}")
            return False

    async def _resolve_provider(self):
        """Resolve a healthy provider with a one-time automatic recovery attempt."""
        if not self._initialized:
            ready = await self._reload_from_environment()
            if not ready:
                raise RuntimeError("LLM_PROVIDER_UNAVAILABLE: No healthy providers available")

        for attempt in range(2):
            try:
                return await self.factory.get_provider()
            except Exception as provider_error:
                normalized_error = _normalize_provider_exception(provider_error)

                if attempt == 0 and _is_provider_unavailable_error(normalized_error):
                    recovered = await self._reload_from_environment()
                    if recovered:
                        continue

                raise normalized_error from provider_error

        raise RuntimeError("LLM_PROVIDER_UNAVAILABLE: No healthy providers available")
    
    async def chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a chat completion using the active provider."""
        try:
            provider = await self._resolve_provider()
            return await provider.chat_completion(request)
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            normalized = _normalize_provider_exception(e)
            raise normalized from e
    
    async def chat_completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming chat completion using the active provider."""
        try:
            provider = await self._resolve_provider()
            async for chunk in provider.chat_completion_stream(request):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            normalized = _normalize_provider_exception(e)
            raise normalized from e
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using the active provider."""
        try:
            provider = await self._resolve_provider()
            return await provider.generate_embedding(request)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            normalized = _normalize_provider_exception(e)
            raise normalized from e
    
    async def health_check(self) -> Dict[LLMProviderType, HealthCheckResult]:
        """Get health status of all providers."""
        if not self._initialized:
            ready = await self._reload_from_environment()
            if not ready:
                raise RuntimeError("LLM service not initialized")
        
        return self.factory.get_health_status()
    
    async def switch_provider(self, provider_type: LLMProviderType, skip_health_check: bool = False) -> bool:
        """Switch to a specific provider."""
        if not self._initialized:
            ready = await self._reload_from_environment()
            if not ready:
                raise RuntimeError("LLM service not initialized")
        
        return await self.factory.switch_provider(provider_type, skip_health_check)

    def is_initialized(self) -> bool:
        """Return whether service providers are initialized and ready."""
        return self._initialized

    def set_uninitialized(self, new_config: LLMConfig) -> None:
        """Reset service state to an uninitialized factory for new config."""
        self.config = new_config
        self.factory = LLMProviderFactory(new_config)
        self._initialized = False
    
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
            self._initialized = False

            if not self.has_valid_provider_config(new_config):
                logger.warning("Updated LLM config has no valid providers; service is uninitialized")
                return False

            await self.factory.initialize()
            self._initialized = True
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

    latest_config = LLMConfig.from_env(dict(os.environ))

    if _llm_service is None:
        _llm_service = LLMService(latest_config)
        if _llm_service.has_valid_provider_config(latest_config):
            await _llm_service.initialize()
        else:
            logger.warning(
                "No valid LLM provider configurations found - LLM service will be created but not initialized"
            )
        return _llm_service

    if _llm_service.config.model_dump() != latest_config.model_dump():
        logger.info("Detected LLM configuration change; refreshing service configuration")
        updated = await _llm_service.update_config(latest_config)
        if not updated:
            _llm_service.set_uninitialized(latest_config)
    elif not _llm_service.is_initialized() and _llm_service.has_valid_provider_config(latest_config):
        await _llm_service.update_config(latest_config)
    
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