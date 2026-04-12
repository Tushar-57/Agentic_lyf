"""
LLM service that provides a high-level interface to the LLM providers.
"""

import os
import json
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
from ..utils.logging import get_conversation_category_logger, get_llm_category_logger

logger = get_llm_category_logger(__name__)
conversation_logger = get_conversation_category_logger("app.conversation.llm")


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, int(raw_value))
    except (TypeError, ValueError):
        return default


AI_CONVERSATION_LOG_ENABLED = _parse_bool_env("AI_CONVERSATION_LOG_ENABLED", default=True)
AI_CONVERSATION_LOG_FULL_TEXT = _parse_bool_env("AI_CONVERSATION_LOG_FULL_TEXT", default=True)
AI_CONVERSATION_LOG_MAX_CHARS = _parse_positive_int_env(
    "AI_CONVERSATION_LOG_MAX_CHARS",
    default=64000,
    minimum=500,
)


def _normalize_log_payload(payload) -> str:
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)


def _prepare_conversation_text_for_log(payload) -> str:
    text = _normalize_log_payload(payload)
    if AI_CONVERSATION_LOG_FULL_TEXT:
        return text

    if len(text) <= AI_CONVERSATION_LOG_MAX_CHARS:
        return text

    return f"{text[:AI_CONVERSATION_LOG_MAX_CHARS - 3]}..."


def _serialize_request_messages(request: CompletionRequest):
    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]


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
            logger.error("init_failed", "Failed to initialize LLM service", error=e)
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
            logger.error("reload_failed", "Failed to reload LLM service from config", error=reload_error)
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
            provider_name = str(provider.provider_type.value)

            if AI_CONVERSATION_LOG_ENABLED:
                conversation_logger.info(
                    "CHAT_COMPLETION_REQUEST provider=%s messages=%s",
                    provider_name,
                    _prepare_conversation_text_for_log(_serialize_request_messages(request)),
                )

            response = await provider.chat_completion(request)

            if AI_CONVERSATION_LOG_ENABLED:
                conversation_logger.info(
                    "CHAT_COMPLETION_RESPONSE provider=%s model=%s usage=%s content=%s",
                    provider_name,
                    response.model or "unknown",
                    _prepare_conversation_text_for_log(response.usage or {}),
                    _prepare_conversation_text_for_log(response.content),
                )

            return response
        except Exception as e:
            logger.error("completion_failed", "Chat completion failed", error=e)
            normalized = _normalize_provider_exception(e)
            raise normalized from e
    
    async def chat_completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming chat completion using the active provider."""
        try:
            provider = await self._resolve_provider()
            provider_name = str(provider.provider_type.value)

            if AI_CONVERSATION_LOG_ENABLED:
                conversation_logger.info(
                    "CHAT_COMPLETION_STREAM_REQUEST provider=%s messages=%s",
                    provider_name,
                    _prepare_conversation_text_for_log(_serialize_request_messages(request)),
                )

            streamed_parts: List[str] = []
            async for chunk in provider.chat_completion_stream(request):
                streamed_parts.append(chunk)
                yield chunk

            if AI_CONVERSATION_LOG_ENABLED:
                conversation_logger.info(
                    "CHAT_COMPLETION_STREAM_RESPONSE provider=%s content=%s",
                    provider_name,
                    _prepare_conversation_text_for_log("".join(streamed_parts)),
                )
        except Exception as e:
            logger.error("streaming_failed", "Streaming chat completion failed", error=e)
            normalized = _normalize_provider_exception(e)
            raise normalized from e
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using the active provider."""
        try:
            provider = await self._resolve_provider()
            return await provider.generate_embedding(request)
        except Exception as e:
            logger.error("embedding_failed", "Embedding generation failed", error=e)
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
            logger.error("models_failed", "Failed to get available models", error=e)
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
            logger.error("update_config_failed", "Failed to update LLM service configuration", error=e)
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