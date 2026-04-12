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
    
    logger.info("=" * 60)
    logger.info("Starting Agentic Lyf Telegram Bot")
    logger.info("=" * 60)
    
    # Validate configuration
    try:
        if not TelegramConfig.validate():
            logger.error("Telegram bot is disabled (TELEGRAM_ENABLED=false)")
            logger.info("To enable:")
            logger.info("1. Get a bot token from @BotFather on Telegram")
            logger.info("2. Add to .env file: TELEGRAM_BOT_TOKEN=your_token_here")
            logger.info("3. Set TELEGRAM_ENABLED=true in .env")
            return
    except ValueError as e:
        logger.error("config_error", "Configuration error", error=e)
        return
    
    # Initialize agent factory
    logger.info("Initializing agent factory...")
    agent_factory = AgentFactory()
    await agent_factory.initialize_agent_ecosystem()
    logger.info("agents_initialized", f"Initialized {len(agent_factory.registry.get_all_agents())} agents", {"agent_count": len(agent_factory.registry.get_all_agents())})
    
    # Create and start bot
    token = TelegramConfig.get_bot_token()
    bot = TelegramBot(token=token, agent_factory=agent_factory)
    
    try:
        logger.info("Starting Telegram bot...")
        await bot.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error("bot_error", "Bot error", error=e)
    finally:
        logger.info("Shutting down...")
        await bot.stop()
        logger.info("✅ Bot stopped successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
