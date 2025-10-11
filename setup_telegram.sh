#!/bin/bash

# Telegram Bot Setup Script for Agentic Lyf

echo "========================================"
echo "Agentic Lyf - Telegram Bot Setup"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo "❌ .env file not found!"
    echo "Creating backend/.env from template..."
    
    cat > backend/.env << 'EOF'
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=
TELEGRAM_ENABLED=false
TELEGRAM_BOT_USERNAME=AgenticLyfBot

# Optional Settings
TELEGRAM_MAX_MESSAGES_PER_MINUTE=20
TELEGRAM_USE_WEBHOOK=false

# LLM Configuration (if not already set)
# OPENAI_API_KEY=your_openai_key_here
# OLLAMA_BASE_URL=http://localhost:11434
EOF

    echo "✅ Created backend/.env file"
    echo ""
fi

# Check if bot token is set
if grep -q "^TELEGRAM_BOT_TOKEN=$" backend/.env; then
    echo "⚠️  Telegram bot token not configured!"
    echo ""
    echo "To set up your bot:"
    echo "1. Open Telegram and search for @BotFather"
    echo "2. Send: /newbot"
    echo "3. Follow prompts to create your bot"
    echo "4. Copy the API token you receive"
    echo "5. Edit backend/.env and set TELEGRAM_BOT_TOKEN=your_token_here"
    echo "6. Set TELEGRAM_ENABLED=true"
    echo ""
    read -p "Have you created your bot and got the token? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your bot token: " bot_token
        
        # Update .env file
        sed -i.bak "s/^TELEGRAM_BOT_TOKEN=.*/TELEGRAM_BOT_TOKEN=$bot_token/" backend/.env
        sed -i.bak "s/^TELEGRAM_ENABLED=.*/TELEGRAM_ENABLED=true/" backend/.env
        rm backend/.env.bak
        
        echo "✅ Bot token configured!"
    else
        echo "Please configure your bot token manually in backend/.env"
        exit 1
    fi
fi

echo ""
echo "Installing dependencies..."
cd backend

# Check if python-telegram-bot is installed
if ! python -c "import telegram" 2>/dev/null; then
    echo "Installing python-telegram-bot..."
    pip install python-telegram-bot==21.0.1
    echo "✅ Telegram bot library installed"
else
    echo "✅ python-telegram-bot already installed"
fi

echo ""
echo "========================================"
echo "Setup Complete! 🎉"
echo "========================================"
echo ""
echo "To start the Telegram bot:"
echo "  cd backend"
echo "  python run_telegram_bot.py"
echo ""
echo "To start both web server AND bot:"
echo "  Terminal 1: cd backend && python start_server.py"
echo "  Terminal 2: cd backend && python run_telegram_bot.py"
echo ""
echo "Read TELEGRAM_BOT_GUIDE.md for full documentation"
echo "========================================"
