"""
LLM provider configuration management.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from app.llm.base import LLMProviderType


def _parse_provider(value: Optional[str]) -> Optional[LLMProviderType]:
    """Safely parse a provider value from env or stored config."""
    if not value:
        return None

    normalized = str(value).strip().lower()
    if not normalized:
        return None

    if normalized in {LLMProviderType.OPENAI.value, "openai"}:
        return LLMProviderType.OPENAI

    if normalized in {LLMProviderType.OLLAMA.value, "ollama"}:
        return LLMProviderType.OLLAMA

    return None


def _has_openai_config(api_key: Optional[str], model: Optional[str]) -> bool:
    return bool((api_key or "").strip() and (model or "").strip())


def _has_ollama_config(endpoint: Optional[str], model: Optional[str]) -> bool:
    return bool((endpoint or "").strip() and (model or "").strip())


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    # Provider selection
    provider: LLMProviderType = LLMProviderType.OLLAMA  # Default to Ollama
    fallback_enabled: bool = True
    fallback_provider: Optional[LLMProviderType] = LLMProviderType.OPENAI  # OpenAI as fallback
    
    # OpenAI configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_base_url: Optional[str] = None
    
    # Ollama configuration
    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    
    # Common parameters
    max_tokens: int = 4000
    temperature: float = 0.7
    
    # Health check settings
    health_check_timeout: float = 30.0
    health_check_interval: float = 300.0  # 5 minutes
    
    class Config:
        use_enum_values = True
    
    def get_provider_config(self, provider_type: LLMProviderType) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        if provider_type == LLMProviderType.OPENAI:
            return {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "base_url": self.openai_base_url,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        elif provider_type == LLMProviderType.OLLAMA:
            return {
                "endpoint": self.ollama_endpoint,
                "model": self.ollama_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    def validate_provider_config(self, provider_type: LLMProviderType) -> bool:
        """Validate configuration for a specific provider."""
        if provider_type == LLMProviderType.OPENAI:
            return bool(self.openai_api_key and self.openai_model)
        elif provider_type == LLMProviderType.OLLAMA:
            return bool(self.ollama_endpoint and self.ollama_model)
        else:
            return False
    
    @classmethod
    def from_env(cls, env_vars: Dict[str, str]) -> "LLMConfig":
        """Create configuration from environment variables."""
        # Import here to avoid circular imports
        from app.services.config_storage import get_config_storage
        
        config_storage = get_config_storage()
        openai_config = config_storage.get_openai_config() or {}
        ollama_config = config_storage.get_ollama_config() or {}

        provider_preference = config_storage.get_provider_preference() or ""
        requested_provider = _parse_provider(env_vars.get("LLM_PROVIDER"))
        preferred_provider = _parse_provider(provider_preference)

        openai_api_key = (env_vars.get("OPENAI_API_KEY") or openai_config.get("api_key") or "").strip() or None
        openai_model = (env_vars.get("OPENAI_MODEL") or openai_config.get("model") or "gpt-3.5-turbo").strip()
        openai_base_url = (env_vars.get("OPENAI_BASE_URL") or openai_config.get("base_url") or None)

        ollama_endpoint = (
            env_vars.get("OLLAMA_ENDPOINT")
            or env_vars.get("OLLAMA_BASE_URL")
            or ollama_config.get("endpoint")
            or ollama_config.get("base_url")
            or "http://localhost:11434"
        )
        ollama_endpoint = str(ollama_endpoint).strip()
        ollama_model = (
            env_vars.get("OLLAMA_MODEL")
            or ollama_config.get("model")
            or "llama3.2:3b"
        )
        ollama_model = str(ollama_model).strip()

        # Provider resolution precedence:
        # 1) explicit env override (LLM_PROVIDER)
        # 2) stored provider preference if that provider is actually configured
        # 3) auto-select OpenAI when OPENAI_API_KEY is present (hosted default)
        # 4) fallback to Ollama
        if requested_provider:
            configured_provider = requested_provider
        elif preferred_provider == LLMProviderType.OPENAI and _has_openai_config(openai_api_key, openai_model):
            configured_provider = LLMProviderType.OPENAI
        elif preferred_provider == LLMProviderType.OLLAMA and _has_ollama_config(ollama_endpoint, ollama_model):
            configured_provider = LLMProviderType.OLLAMA
        elif _has_openai_config(openai_api_key, openai_model):
            configured_provider = LLMProviderType.OPENAI
        else:
            configured_provider = LLMProviderType.OLLAMA

        configured_fallback = _parse_provider(env_vars.get("LLM_FALLBACK_PROVIDER"))
        if configured_fallback and configured_fallback != configured_provider:
            fallback_provider = configured_fallback
        else:
            fallback_provider = (
                LLMProviderType.OPENAI
                if configured_provider == LLMProviderType.OLLAMA
                else LLMProviderType.OLLAMA
            )

        return cls(
            provider=configured_provider,
            fallback_enabled=env_vars.get("LLM_FALLBACK_ENABLED", "true").lower() == "true",
            fallback_provider=fallback_provider,
            
            # Allow deployment env vars to override persisted config.
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            
            ollama_endpoint=ollama_endpoint,
            ollama_model=ollama_model,
            
            max_tokens=int(env_vars.get("LLM_MAX_TOKENS", "4000")),
            temperature=float(env_vars.get("LLM_TEMPERATURE", "0.7")),
            
            health_check_timeout=float(env_vars.get("LLM_HEALTH_CHECK_TIMEOUT", "30.0")),
            health_check_interval=float(env_vars.get("LLM_HEALTH_CHECK_INTERVAL", "300.0"))
        )