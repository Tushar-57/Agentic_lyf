#!/usr/bin/env python3
"""
Telegram Bot Test Runner
Helps you test the Telegram bot setup step by step.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n" + "="*60)
    print("STEP 1: Checking Dependencies")
    print("="*60)
    
    try:
        import telegram
        print("✅ python-telegram-bot is installed")
        print(f"   Version: {telegram.__version__}")
        return True
    except ImportError:
        print("❌ python-telegram-bot is NOT installed")
        print("   Run: pip install python-telegram-bot==21.0.1")
        return False


def check_configuration():
    """Check if Telegram bot is configured"""
    print("\n" + "="*60)
    print("STEP 2: Checking Configuration")
    print("="*60)
    
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        print("❌ No .env file found")
        print("\n📝 To create your Telegram bot:")
        print("   1. Open Telegram app")
        print("   2. Search for @BotFather")
        print("   3. Send: /newbot")
        print("   4. Follow the prompts")
        print("   5. Copy the API token")
        print("\n📝 Then create .env file with:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")
        print("   TELEGRAM_ENABLED=true")
        return False
    
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    enabled = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
    
    if not token or token == "your_bot_token_here":
        print("❌ TELEGRAM_BOT_TOKEN not configured in .env")
        print("\n📝 Add to .env file:")
        print("   TELEGRAM_BOT_TOKEN=your_actual_token")
        return False
    
    if not enabled:
        print("⚠️  TELEGRAM_ENABLED is false")
        print("   Set TELEGRAM_ENABLED=true in .env")
        return False
    
    print("✅ Configuration found:")
    print(f"   Token: {token[:10]}...{token[-10:] if len(token) > 20 else ''}")
    print(f"   Enabled: {enabled}")
    return True


def check_agent_system():
    """Check if agent system is available"""
    print("\n" + "="*60)
    print("STEP 3: Checking Agent System")
    print("="*60)
    
    try:
        from app.agents.factory import AgentFactory
        from app.services.knowledge_base import KnowledgeBaseService
        from app.llm.service import LLMService
        
        print("✅ Agent system imports successful")
        
        # Try to initialize (without actually creating agents)
        print("   - AgentFactory available")
        print("   - KnowledgeBaseService available")
        print("   - LLMService available")
        return True
    except Exception as e:
        print(f"❌ Error importing agent system: {e}")
        return False


def test_bot_connection():
    """Test bot connection to Telegram"""
    print("\n" + "="*60)
    print("STEP 4: Testing Bot Connection")
    print("="*60)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        from telegram import Bot
        import asyncio
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        
        async def test():
            bot = Bot(token=token)
            me = await bot.get_me()
            return me
        
        me = asyncio.run(test())
        print("✅ Bot connection successful!")
        print(f"   Bot Name: {me.first_name}")
        print(f"   Username: @{me.username}")
        print(f"   Bot ID: {me.id}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Possible issues:")
        print("   - Invalid bot token")
        print("   - Network connectivity problem")
        print("   - Bot was deleted by @BotFather")
        return False


def show_next_steps():
    """Show next steps to run the bot"""
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED!")
    print("="*60)
    print("\n📱 Your Telegram bot is ready to use!")
    print("\nTo start the bot:")
    print("   python run_telegram_bot.py")
    print("\nOr in the background:")
    print("   nohup python run_telegram_bot.py > telegram_bot.log 2>&1 &")
    print("\nTo test in Telegram:")
    print("   1. Open Telegram app")
    print("   2. Search for your bot by username")
    print("   3. Click 'Start' or send: /start")
    print("   4. Try: 'Give me 2 LeetCode problems'")
    print("   5. Try: 'What's my budget?' (should see honest response!)")
    print("\nAvailable commands:")
    print("   /start  - Welcome message")
    print("   /help   - Show help")
    print("   /profile - View your preferences")
    print("   /status - Bot status")
    print("   /reset  - Clear conversation")


def main():
    """Main test runner"""
    print("\n🤖 TELEGRAM BOT TEST RUNNER")
    print("Testing your Telegram bot setup...\n")
    
    # Run checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_configuration),
        ("Agent System", check_agent_system),
        ("Bot Connection", test_bot_connection),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
            
            # Stop if check fails
            if not result:
                print("\n" + "="*60)
                print(f"❌ {name} check failed!")
                print("="*60)
                print("\n💡 Fix the issue above and run this test again:")
                print("   python test_telegram_setup.py")
                return False
                
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # All checks passed
    show_next_steps()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
