#!/usr/bin/env python3
"""Launcher script for Telegram bot."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.telegram.bot import TelegramBot
from app.telegram.config import TelegramConfig
from app.agents.factory import AgentFactory
from app.utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.NOTIFICATION)


async def main():
    """Main entry point for Telegram bot."""
    
    logger.info("telegram_bot_start", "Starting Agentic Lyf Telegram Bot")
    
    # Validate configuration
    try:
        if not TelegramConfig.validate():
            logger.error("telegram_disabled", "Telegram bot is disabled (TELEGRAM_ENABLED=false)")
            logger.info("telegram_enable_help", "To enable: Get a bot token from @BotFather, add TELEGRAM_BOT_TOKEN to .env, set TELEGRAM_ENABLED=true")
            return
    except ValueError as e:
        logger.error("config_error", "Configuration error", error=e)
        return
    
    # Initialize agent factory
    logger.info("init_agent_factory", "Initializing agent factory")
    agent_factory = AgentFactory()
    await agent_factory.initialize_agent_ecosystem()
    logger.info("agents_initialized", f"Initialized {len(agent_factory.registry.get_all_agents())} agents", {"agent_count": len(agent_factory.registry.get_all_agents())})
    
    # Create and start bot
    token = TelegramConfig.get_bot_token()
    bot = TelegramBot(token=token, agent_factory=agent_factory)
    
    try:
        logger.info("telegram_bot_running", "Starting Telegram bot")
        await bot.run()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt", "Received keyboard interrupt")
    except Exception as e:
        logger.error("bot_error", "Bot error", error=e)
    finally:
        logger.info("shutting_down", "Shutting down")
        await bot.stop()
        logger.info("bot_stopped", "Bot stopped successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
