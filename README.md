# Lightning-Fast RAG System

A high-performance Retrieval-Augmented Generation (RAG) system designed for <300ms response times with local deployment.

## 🚀 Features

- **Ultra-Fast Performance**: <300ms average response time
- **FAISS Vector Store**: Lightning-fast similarity search
- **Advanced Caching**: Multi-level caching for embeddings and responses
- **Multiple PDF Processors**: PyPDF2, pdfplumber, and pymupdf fallbacks
- **Circuit Breakers**: Robust error handling and API reliability
- **Hackathon Ready**: Compliant with HackRx evaluation endpoint

## 📊 Performance Profile

| Component | Expected Time | Optimization |
|-----------|---------------|--------------|
| Vector search | 1-5ms | ChromaDB RAM-only |
| Google embedding | 30-80ms | Batch processing + cache |
| Gemini generation | 100-200ms | Optimized prompts |
| **Total (cache miss)** | **136-295ms** | ✅ Under target |
| **Total (cache hit)** | **6-17ms** | 🚀 Lightning fast |

## 🛠️ Quick Setup

```bash
# 1. Clone and setup
git clone <repo-url>
cd HackRxV3
chmod +x setup.sh
./setup.sh

# 2. Configure API keys in .env
GOOGLE_API_KEY=your-google-ai-api-key
TOGETHER_API_KEY=your-together-api-key  # Optional fallback

# 3. Start the system
./start_rag.sh

# Or use the enhanced version for maximum performance
./start_enhanced.sh

# 4. Start ngrok tunnel (in another terminal)
ngrok http 8000
```

## 🏗️ Architecture

```
📄 PDF Input → 🔄 Processing → 🧠 Embeddings → 💾 Vector Store
                                                       ↓
🎯 Response ← 🤖 LLM Generation ← 🔍 Context ← 🔎 Similarity Search
```

## 🚀 Enhanced RAG System

The enhanced version (`start_enhanced.sh`) includes additional improvements:

- **Tiktoken-based Token Counting**: Precise token measurement for optimal chunking
- **Advanced Circuit Breakers**: Improved reliability with configurable failure thresholds
- **Multi-level Caching**: Enhanced caching strategy with TTL and LRU policies
- **Non-recursive Chunking**: Memory-safe document processing algorithm
- **Advanced PDF Extraction**: Multiple extraction methods with intelligent fallbacks
- **Concurrent Processing**: Optimized thread and task management
- **Real-time Monitoring**: Enhanced dashboard with detailed metrics

### Usage

```bash
# Start the enhanced version
./start_enhanced.sh

# Access enhanced dashboard
http://localhost:8000/dashboard

# Test the enhanced system
./test_enhanced.sh
```

### Core Components

1. **PDF Processor** (`pdf_processor.py`)
   - Multi-method extraction (PyPDF2, pdfplumber, pymupdf)
   - Automatic fallback and retry logic
   - Async processing with thread executors

2. **Document Processor** (`document_processor.py`)
   - Intelligent chunking with RecursiveCharacterTextSplitter
   - Token-aware splitting (400 tokens optimal)
   - Metadata extraction for better search

3. **Embedding Service** (`embedding_service.py`)
   - Google text-embedding-004 model
   - Batch processing for efficiency
   - Aggressive caching with LRU eviction

4. **Vector Store** (`vector_store.py`)
   - ChromaDB in-memory configuration
   - Optimized HNSW parameters
   - Sub-10ms search times

5. **LLM Service** (`llm_service.py`)
   - Primary: Gemini 2.0 Flash Lite
   - Fallback: Together.ai Llama
   - Circuit breaker protection

6. **RAG Pipeline** (`rag_pipeline.py`)
   - Orchestrates entire workflow
   - Parallel processing where possible
   - Comprehensive error handling

## 🎯 API Endpoints

### Hackathon Endpoint
```http
POST /hackrx/run
Authorization: Bearer 09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b
Content-Type: application/json

{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

### Other Endpoints
- `GET /health` - System health and statistics
- `GET /metrics` - Performance dashboard
- `POST /ask` - Single question endpoint
- `GET /cache/stats` - Cache performance

## ⚡ Performance Optimizations

### 1. Multi-Level Caching
```python
# Embedding cache (永久)
embedding_cache: Dict[str, List[float]]

# Query cache (5分钟TTL)
query_cache: Dict[str, Dict]

# LRU cache for frequent lookups
@lru_cache(maxsize=1000)
```

### 2. Parallel Processing
```python
# Concurrent question answering
semaphore = asyncio.Semaphore(5)

# Batch embedding generation
embeddings = await get_embedding_batch(texts, batch_size=50)

# Thread execution for blocking operations
with ThreadPoolExecutor(max_workers=2) as executor:
```

### 3. Circuit Breakers
```python
# API reliability protection
gemini_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
embedding_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
```

## 🔧 Configuration

### Environment Variables
```bash
GOOGLE_API_KEY=your-api-key           # Required
TOGETHER_API_KEY=your-api-key         # Optional fallback
MAX_EMBEDDING_BATCH_SIZE=50           # Batch size for embeddings
MAX_QUESTION_CONCURRENCY=5            # Parallel question processing
MAX_CONTEXT_TOKENS=2000               # Context window limit
```

### System Requirements
- **RAM**: 4-8GB free (documents stored in memory)
- **CPU**: Multi-core recommended
- **Storage**: SSD for faster model loading
- **Network**: Stable internet for API calls

## 📈 Monitoring

### Metrics Dashboard
Visit `http://localhost:8000/metrics` for real-time system stats:
- Cache hit rates
- Response times
- Memory usage
- Vector store status

### Performance Logs
```bash
✅ PDF downloaded and extracted in 1247.23ms, 45231 characters
✅ Indexed 127 chunks in 3421.45ms
✅ Answered question in 156.78ms
🎯 Complete pipeline finished in 4825.46ms
```

## 🧪 Testing

```bash
# Test system health
curl http://localhost:8000/health

# Test single question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "pdf_url": "https://example.com/doc.pdf"}'

# Run full test suite
./test_rag.sh
```

## 🚨 Troubleshooting

### Common Issues

1. **Slow Response Times**
   - Check cache hit rates (`/cache/stats`)
   - Verify sufficient RAM available
   - Monitor API quota limits

2. **PDF Processing Errors**
   - System tries 3 extraction methods automatically
   - Check PDF URL accessibility
   - Verify PDF is not password protected

3. **API Key Issues**
   - Ensure keys are valid and have quota
   - Check `.env` file configuration
   - Monitor API usage limits

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python main.py
```

## 🏆 Hackathon Compliance

This system is specifically designed for the HackRx evaluation:

- ✅ Exact endpoint specification (`/hackrx/run`)
- ✅ Required authorization token support
- ✅ PDF URL processing capability
- ✅ Batch question answering
- ✅ Proper error handling
- ✅ Performance optimization (<300ms target)

## 🔮 Future Enhancements

- [ ] GPU acceleration for embeddings
- [ ] Distributed vector storage
- [ ] Real-time document updates
- [ ] Advanced relevance scoring
- [ ] Multi-language support

---

**Built with ⚡ for maximum speed and 🧠 for intelligent responses**
