#!/bin/bash

# Test script for the enhanced Lightning RAG system
echo "🧪 Testing Enhanced Lightning RAG System..."

# Test health endpoint
echo -e "\n📊 Testing health endpoint..."
curl -s http://127.0.0.1:8000/health | python -m json.tool

# Create a test document
echo -e "\n📝 Creating test document..."
cat > test_document.txt << 'EOL'
This is a test document for the Lightning RAG system.

The system uses a multi-level caching approach with the following features:
1. Embedding cache (permanent storage)
2. Query cache (5 minute TTL)
3. LRU cache for frequent lookups

The system also implements circuit breakers for API reliability:
- Gemini breaker with a failure threshold of 3
- Embedding breaker with a timeout of 30 seconds

For parallel processing, the system uses:
- Concurrent question answering with a semaphore of 5
- Batch embedding generation with a batch size of 50
- Thread execution for blocking operations
EOL

# Test document ingestion
echo -e "\n📥 Testing document ingestion..."
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["'"$(cat test_document.txt)"'"]}' | python -m json.tool

# Test question answering
echo -e "\n❓ Testing question answering..."
curl -s -X POST http://127.0.0.1:8000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the cache TTL?"}' | python -m json.tool

# Test batch question answering
echo -e "\n❓ Testing batch question answering..."
curl -s -X POST http://127.0.0.1:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"questions": ["What is the failure threshold?", "What is the semaphore value?"]}' | python -m json.tool

# Test cache stats
echo -e "\n💾 Testing cache stats..."
curl -s http://127.0.0.1:8000/cache/stats | python -m json.tool

echo -e "\n✅ Test complete!"
