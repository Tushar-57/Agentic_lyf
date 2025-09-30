"""
Configuration storage service for persisting user settings and API keys.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigStorage:
    """Simple file-based configuration storage."""
    
    def __init__(self, config_dir: str = "data/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "user_config.json"
        self._config: Dict[str, Any] = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2)
        except (OSError, ValueError) as e:
            print(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
        self._save_config()
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return self.get('openai', {})
    
    def set_openai_config(self, config: Dict[str, Any]) -> None:
        """Set OpenAI configuration."""
        self.set('openai', config)
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration."""
        return self.get('ollama', {'endpoint': 'http://localhost:11434'})
    
    def set_ollama_config(self, config: Dict[str, Any]) -> None:
        """Set Ollama configuration."""
        self.set('ollama', config)
    
    def get_provider_preference(self) -> str:
        """Get preferred LLM provider."""
        return self.get('provider_preference', 'ollama')
    
    def set_provider_preference(self, provider: str) -> None:
        """Set preferred LLM provider."""
        self.set('provider_preference', provider)


# Global instance storage
class _ConfigStorageHolder:
    instance: Optional[ConfigStorage] = None

_holder = _ConfigStorageHolder()


def get_config_storage() -> ConfigStorage:
    """Get global configuration storage instance."""
    if _holder.instance is None:
        _holder.instance = ConfigStorage()
    return _holder.instance