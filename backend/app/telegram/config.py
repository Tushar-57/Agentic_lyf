"""Configuration for Telegram bot."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class TelegramConfig:
    """Configuration for Telegram bot."""
    
    # Telegram Bot Token (get from @BotFather)
    BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Bot settings
    BOT_USERNAME: str = os.getenv("TELEGRAM_BOT_USERNAME", "AgenticLyfBot")
    
    # Enable/disable bot
    TELEGRAM_ENABLED: bool = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
    
    # Webhook settings (optional - for production)
    USE_WEBHOOK: bool = os.getenv("TELEGRAM_USE_WEBHOOK", "false").lower() == "true"
    WEBHOOK_URL: Optional[str] = os.getenv("TELEGRAM_WEBHOOK_URL")
    WEBHOOK_PORT: int = int(os.getenv("TELEGRAM_WEBHOOK_PORT", "8443"))
    
    # Rate limiting
    MAX_MESSAGES_PER_MINUTE: int = int(os.getenv("TELEGRAM_MAX_MESSAGES_PER_MINUTE", "20"))
    
    # Message settings
    MAX_MESSAGE_LENGTH: int = 4096  # Telegram's limit
    TYPING_DELAY: float = 0.5  # Delay before showing typing indicator
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration."""
        if not cls.TELEGRAM_ENABLED:
            return False
        
        if not cls.BOT_TOKEN:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is required. "
                "Get it from @BotFather on Telegram and add to .env file"
            )
        
        return True
    
    @classmethod
    def get_bot_token(cls) -> str:
        """Get bot token, raising error if not configured."""
        if not cls.BOT_TOKEN:
            raise ValueError(
                "Telegram bot token not configured. "
                "Set TELEGRAM_BOT_TOKEN in your .env file"
            )
        return cls.BOT_TOKEN
