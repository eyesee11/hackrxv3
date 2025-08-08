#!/bin/bash

echo "🚀 Setting up Lightning-Fast RAG System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_info "Detected Python version: $python_version"

# Create virtual environment
print_info "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

print_status "Virtual environment created and activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_info "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_status "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Add missing dependencies for full functionality
print_info "Installing additional performance dependencies..."
pip install psutil  # For memory monitoring

# Check if .env file exists and guide user
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating template..."
    cat > .env << EOF
# Environment variables with defaults
cat > .env << EOL
# Google AI API Configuration
GOOGLE_API_KEY=your_google_ai_api_key_here

# Together.ai API Configuration (Fallback)
TOGETHER_API_KEY=your_together_api_key_here

# Hackathon specific
HACKATHON_TOKEN=09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b

# Performance settings
MAX_EMBEDDING_BATCH_SIZE=50
MAX_QUESTION_CONCURRENCY=5
MAX_CONTEXT_TOKENS=2000
RESPONSE_CACHE_TTL=300

# Server configuration
HOST=127.0.0.1
PORT=8000
EOL
EOF
    print_warning "Please edit .env file and add your API keys!"
else
    print_status ".env file already exists"
fi

# System dependencies check
print_info "Checking system dependencies..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_status "pip3 found"
else
    print_error "pip3 not found. Please install pip3."
    exit 1
fi

# Create startup script
print_info "Creating startup script..."
cat > start_rag.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Lightning-Fast RAG System..."

# Activate virtual environment
source venv/bin/activate

# Check if API keys are set
if grep -q "your-google-api-key-here" .env; then
    echo "❌ Please set your GOOGLE_API_KEY in .env file"
    exit 1
fi

# Start the FastAPI server
echo "🔥 Starting FastAPI server on port 8000..."
python main.py &
SERVER_PID=$!

# Wait a moment for server to start
sleep 3

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo "📍 Server running at: http://localhost:8000"
    echo "📊 Metrics dashboard: http://localhost:8000/metrics"
    echo "🏥 Health check: http://localhost:8000/health"
    echo ""
    echo "To start ngrok tunnel, run in another terminal:"
    echo "ngrok http 8000"
    echo ""
    echo "Press Ctrl+C to stop the server"
    
    # Keep script running
    wait $SERVER_PID
else
    echo "❌ Failed to start server"
    exit 1
fi
EOF

chmod +x start_rag.sh
print_status "Startup script created: ./start_rag.sh"

# Create test script
print_info "Creating test script..."
cat > test_rag.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing RAG System..."

# Test health endpoint
echo "Testing health endpoint..."
curl -s http://localhost:8000/health | jq '.' || echo "Server not running or jq not installed"

echo ""
echo "Testing hackathon endpoint with sample data..."

# Sample test (replace with actual PDF URL and questions)
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b" \
  -d '{
    "documents": "https://example.com/sample.pdf",
    "questions": ["What is this document about?"]
  }' | jq '.' || echo "Test failed - check if server is running and API keys are set"
EOF

chmod +x test_rag.sh
print_status "Test script created: ./test_rag.sh"

# Print setup completion message
echo ""
echo "🎉 Setup completed successfully!"
echo ""
print_info "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - GOOGLE_API_KEY: Get from Google AI Studio"
echo "   - TOGETHER_API_KEY: Get from Together.ai (optional fallback)"
echo ""
echo "2. Start the RAG system:"
echo "   python main_simple.py"
echo ""
echo "3. The server will be available at:"
echo "   http://127.0.0.1:8000"
echo ""
echo "4. To expose externally, run ngrok manually in another terminal:"
echo "   ngrok http 8000"
echo ""
echo "5. Test the system:"
echo "   ./test_rag.sh"
echo ""
print_info "Expected performance: <300ms response time with caching"
print_info "Memory usage: ~500MB-2GB depending on document size"
echo ""
echo "🔧 Troubleshooting:"
echo "- Check logs in terminal for any errors"
echo "- Visit http://127.0.0.1:8000/dashboard for system stats"
echo "- Ensure you have at least 4GB free RAM"
echo "- API keys must be valid and have sufficient quota"
echo ""
print_status "Ready to build lightning-fast RAG! ⚡"
