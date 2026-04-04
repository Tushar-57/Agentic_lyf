"""Message and command handlers for Telegram bot."""

from typing import Dict, Any


class MessageHandler:
    """Handler for processing user messages."""
    
    @staticmethod
    async def process_message(message: str, user_context: Dict[str, Any]) -> str:
        """
        Process incoming user message.
        
        Args:
            message: User message text
            user_context: User context and preferences
        
        Returns:
            Processed message ready for agent
        """
        # Basic message preprocessing
        message = message.strip()
        
        # Could add more preprocessing here:
        # - Spell check
        # - Intent pre-classification
        # - Language detection
        
        return message


class CommandHandler:
    """Handler for bot commands."""
    
    COMMANDS = {
        "start": "Initialize the bot and show welcome message",
        "help": "Display help information",
        "profile": "View your profile settings",
        "settings": "Configure bot preferences",
        "status": "Check bot status",
        "reset": "Reset conversation context"
    }
    
    @classmethod
    def get_command_list(cls) -> Dict[str, str]:
        """Get list of available commands."""
        return cls.COMMANDS
    
    @classmethod
    def is_valid_command(cls, command: str) -> bool:
        """Check if command is valid."""
        return command.lstrip('/') in cls.COMMANDS
