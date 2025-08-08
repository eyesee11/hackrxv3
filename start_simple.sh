#!/bin/bash

# Lightning RAG System Startup Script
# Simplified version without ngrok dependencies

echo "🚀 Starting Lightning RAG System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Using default configuration."
    echo "💡 Copy .env.example to .env and configure your API keys for full functionality."
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if API keys are configured
if grep -q "your_google_ai_api_key_here" .env 2>/dev/null; then
    echo "⚠️ Warning: Default API keys detected in .env file"
    echo "💡 Please configure your Google AI API key for full functionality"
fi

# Start the server
echo "🌟 Starting RAG API server..."
echo "📊 Dashboard: http://127.0.0.1:8000/dashboard"
echo "🔍 Health: http://127.0.0.1:8000/health"
echo "📖 API Docs: http://127.0.0.1:8000/docs"
echo "🎯 Hackathon Endpoint: http://127.0.0.1:8000/hackrx/run"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"

python main_simple.py
