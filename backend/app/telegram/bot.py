"""Telegram Bot implementation for Agentic Lyf."""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime


from ..services.knowledge_base import KnowledgeBaseService
from ..agents.factory import AgentFactory
from .formatters import ResponseFormatter
from ..utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.NOTIFICATION)


class TelegramBot:
    """Telegram bot for Agentic Lyf agent system."""
    
    def __init__(self, token: str, agent_factory: AgentFactory):
        """
        Initialize Telegram bot.
        
        Args:
            token: Telegram bot token from BotFather
            agent_factory: Factory for creating and managing agents
        """
        self.token = token
        self.agent_factory = agent_factory
        self.application: Optional[Application] = None
        self.kb_service = KnowledgeBaseService()
        self.formatter = ResponseFormatter()
        
        # User session management
        self.user_sessions: Dict[int, Dict[str, Any]] = {}
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        logger.info("new_user", f"New user started bot: {user.username}", {"username": user.username, "user_id": user.id})
        
        # Initialize user session
        self.user_sessions[chat_id] = {
            "user_id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "started_at": datetime.utcnow().isoformat(),
            "conversation_id": f"telegram_{chat_id}"
        }
        
        welcome_message = f"""
🌟 **Welcome to Agentic Lyf, {user.first_name}!** 🌟

I'm your personal AI assistant ecosystem, ready to help you with:

📊 **Productivity** - Task management, goal tracking, LeetCode practice
💰 **Finance** - Budget planning, expense tracking, financial goals
🏥 **Health** - Meal planning, fitness tracking, wellness
📅 **Scheduling** - Time optimization, appointment booking
📝 **Journaling** - Daily reflections, habit tracking, emotional wellness
💬 **General** - Anything else you need help with!

**Quick Commands:**
/help - Show available commands
/profile - View your profile settings
/settings - Configure your preferences
/status - Check bot status

**Just send me a message to get started!** 🚀

Example: "Can you give me today's LeetCode problems?"
        """
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_message = """
📚 **Agentic Lyf Help Guide** 📚

**Available Commands:**
/start - Initialize the bot
/help - Show this help message
/profile - View your profile and preferences
/settings - Configure your AI mentor style
/status - Check bot and agent status
/reset - Reset conversation context

**How to Use:**
Simply send me a message about what you need help with!

**Examples:**
• "What's my budget for this month?"
• "Help me plan my day"
• "Give me two LeetCode problems to solve"
• "Track my calories for today"
• "How can I improve my time management?"

**Agent Specializations:**
🎯 Productivity - Tasks, goals, learning
💰 Finance - Money management
🏥 Health - Wellness and fitness
📅 Scheduling - Time management
📝 Journal - Reflections and habits
💬 General - Everything else

**Tips:**
✅ Be specific with your requests
✅ Ask follow-up questions
✅ Set up your profile for personalized responses

Need more help? Just ask! 😊
        """
        
        await update.message.reply_text(
            help_message,
            parse_mode='Markdown'
        )
    
    async def profile_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /profile command - show user profile."""
        chat_id = update.effective_chat.id
        
        try:
            # Get user preferences from knowledge base
            user_prefs = await self.kb_service.get_user_preferences()
            
            if not user_prefs:
                await update.message.reply_text(
                    "⚙️ You don't have a profile set up yet.\n\n"
                    "Visit the web app to complete your onboarding and set preferences!"
                )
                return
            
            # Convert Pydantic model to dict if needed
            if hasattr(user_prefs, 'model_dump'):
                user_prefs_dict = user_prefs.model_dump()
            elif hasattr(user_prefs, 'dict'):
                user_prefs_dict = user_prefs.dict()
            else:
                user_prefs_dict = dict(user_prefs) if isinstance(user_prefs, dict) else {}
            
            # Format profile information
            profile_text = self.formatter.format_profile(user_prefs_dict)
            
            await update.message.reply_text(
                profile_text,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error("profile_command_failed", "Profile command failed", error=e)
            await update.message.reply_text(
                "❌ Error loading profile. Please try again later."
            )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command - show bot status."""
        try:
            # Get agent registry status
            agents = self.agent_factory.registry.get_all_agents()
            
            status_text = f"""
🤖 **Bot Status** 🤖

✅ **Connected and Active**

**Registered Agents:** {len(agents)}
{''.join([f'• {agent.agent_type.value.title()}' for agent in agents[:7]])}

**Session Info:**
• Chat ID: `{update.effective_chat.id}`
• User ID: `{update.effective_user.id}`

**System:**
• Response Time: ~2-5 seconds
• Available 24/7

All systems operational! 🚀
            """
            
            await update.message.reply_text(
                status_text,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error("status_command_failed", "Status command failed", error=e)
            await update.message.reply_text("❌ Error checking status.")
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reset command - reset conversation context."""
        chat_id = update.effective_chat.id
        
        # Reset user session
        if chat_id in self.user_sessions:
            old_conv_id = self.user_sessions[chat_id].get("conversation_id")
            self.user_sessions[chat_id]["conversation_id"] = f"telegram_{chat_id}_{datetime.utcnow().timestamp()}"
            
            await update.message.reply_text(
                "🔄 **Conversation Reset!**\n\n"
                "Your conversation context has been cleared. "
                "Starting fresh! 🌟"
            )
        else:
            await update.message.reply_text(
                "ℹ️ No active conversation to reset."
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming user messages."""
        chat_id = update.effective_chat.id
        user_message = update.message.text
        
        # Ensure user session exists
        if chat_id not in self.user_sessions:
            self.user_sessions[chat_id] = {
                "user_id": update.effective_user.id,
                "username": update.effective_user.username,
                "conversation_id": f"telegram_{chat_id}"
            }
        
        try:
            # Show typing indicator
            await context.bot.send_chat_action(
                chat_id=chat_id,
                action="typing"
            )
            
            # Get user preferences
            user_prefs = await self.kb_service.get_user_preferences()
            
            # Convert to dict if it's a Pydantic model
            if user_prefs:
                if hasattr(user_prefs, 'model_dump'):
                    user_prefs_dict = user_prefs.model_dump()
                elif hasattr(user_prefs, 'dict'):
                    user_prefs_dict = user_prefs.dict()
                else:
                    user_prefs_dict = dict(user_prefs) if not isinstance(user_prefs, dict) else user_prefs
            else:
                user_prefs_dict = {}
            
            # Get orchestrator agent
            orchestrator = self.agent_factory.registry.get_agent_by_type("orchestrator")
            
            if not orchestrator:
                await update.message.reply_text(
                    "❌ Agent system is not available. Please try again later."
                )
                return
            
            # Execute orchestrator workflow
            logger.info("processing_message", f"Processing Telegram message", {"chat_id": chat_id, "message_preview": user_message[:50]})
            
            # Call the orchestrator's process_message method
            result = await orchestrator.process_message(
                user_input=user_message,
                session_id=self.user_sessions[chat_id]["conversation_id"],
                context={"user_preferences": user_prefs_dict}
            )
            
            # Extract response
            if isinstance(result, dict):
                response_text = result.get("agent_response", result.get("response", ""))
                # If response_text is still a dict, extract the actual text
                if isinstance(response_text, dict):
                    response_text = response_text.get("response", str(response_text))
                reasoning = result.get("reasoning", {})
            else:
                response_text = str(result)
                reasoning = {}
            
            # Format response for Telegram
            formatted_response = self.formatter.format_response(
                response_text,
                reasoning,
                platform="telegram"
            )
            
            # Send response (split if too long)
            # Use MarkdownV2 with escaping or disable parse_mode if formatting fails
            if len(formatted_response) > 4096:
                # Split into chunks
                chunks = [formatted_response[i:i+4096] for i in range(0, len(formatted_response), 4096)]
                for chunk in chunks:
                    try:
                        await update.message.reply_text(
                            chunk,
                            parse_mode='Markdown'
                        )
                    except Exception as parse_error:
                        # If Markdown parsing fails, send as plain text
                        await update.message.reply_text(chunk)
            else:
                try:
                    await update.message.reply_text(
                        formatted_response,
                        parse_mode='Markdown'
                    )
                except Exception as parse_error:
                    # If Markdown parsing fails, send as plain text
                    await update.message.reply_text(formatted_response)
            
        except Exception as e:
            logger.error("message_handling_failed", "Message handling failed", error=e)
            await update.message.reply_text(
                "❌ Oops! Something went wrong processing your message.\n\n"
                "Please try again or use /help for assistance."
            )
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error("bot_error", "Telegram bot error", {"error": str(context.error)}, error=context.error if isinstance(context.error, Exception) else None)
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again later."
            )
    
    def build_application(self) -> Application:
        """Build and configure the Telegram application."""
        # Create application
        self.application = ApplicationBuilder().token(self.token).build()
        
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("profile", self.profile_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("reset", self.reset_command))
        
        # Add message handler for all text messages
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        
        # Add error handler
        self.application.add_error_handler(self.error_handler)
        
        logger.info("bot_app_built", "Telegram bot application built successfully")
        
        return self.application
    
    async def start(self) -> None:
        """Start the Telegram bot."""
        if not self.application:
            self.build_application()
        
        logger.info("bot_starting", "Starting Telegram bot")
        
        # Initialize and start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
        logger.info("bot_running", "Telegram bot is running")
    
    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self.application:
            logger.info("bot_stopping", "Stopping Telegram bot")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("bot_stopped", "Telegram bot stopped")
    
    async def run(self) -> None:
        """Run the bot (blocking)."""
        await self.start()
        
        # Keep running until interrupted
        try:
            # Run indefinitely
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("stop_signal", "Received stop signal")
        finally:
            await self.stop()
