#!/bin/bash

# Start the enhanced Lightning RAG system
echo "⚡ Starting Enhanced Lightning RAG System..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "🔑 Loading environment variables..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️ No .env file found. Make sure you have configured your API keys."
fi

# Check for required dependencies
echo "🔍 Checking dependencies..."
python -c "import tiktoken" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📦 Installing tiktoken for accurate token counting..."
    pip install tiktoken
fi

# Start the enhanced server
echo "🚀 Starting enhanced server..."
python main_enhanced.py

# Cleanup on exit
echo "👋 Shutting down..."
