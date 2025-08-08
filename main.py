import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Try to install uvloop for better async performance, but don't fail if not available
try:
    import uvloop
    uvloop.install()
    print("✅ uvloop installed for better async performance")
except ImportError:
    print("⚠️ uvloop not available, using default asyncio event loop")

# Import our RAG components
from rag_pipeline_simple import (
    get_rag_pipeline, 
    answer_question, 
    ingest_documents,
    warmup_pipeline,
    get_pipeline_stats
)
from cache_system import get_cache

load_dotenv()

# Pydantic models for API
class HackathonRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    system_stats: dict

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    # Startup
    print("🚀 Starting RAG system...")
    
    await warmup_pipeline()
    
    yield
    
    # Shutdown
    print("� Shutting down RAG system...")

# Add performance monitoring middleware
@app.middleware("http")
async def add_performance_headers(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Add cache statistics
    cache = get_cache()
    response.headers["X-Cache-Hit-Rate"] = f"{cache.hit_rate():.2%}"
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    stats = get_pipeline_stats()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system": stats
    }

# Create FastAPI application
app = FastAPI(
    title="Lightning-Fast RAG System",
    description="High-performance RAG system for insurance policy Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Store startup time for health checks
app.startup_time = time.time()

# Optimized CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    response.headers["X-Cache-Hit-Rate"] = f"{cache.hit_rate():.2%}"
    return response

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system statistics"""
    
    uptime = time.time() - app.startup_time
    stats = get_system_stats()
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        system_stats=stats
    )

# Main hackathon endpoint
@app.post("/hackrx/run", response_model=HackathonResponse)
async def hackrx_endpoint(
    request: HackathonRequest,
    authorization: str = Header(..., description="Bearer token")
):
    """Main hackathon endpoint - processes PDF and answers questions"""
    
    # Validate authorization
    expected_token = f"Bearer {os.getenv('HACKATHON_TOKEN', '09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b')}"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    # Validate request
    if not request.documents:
        raise HTTPException(status_code=400, detail="PDF URL is required")
    
    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one question is required")
    
    if len(request.questions) > 20:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Too many questions (max 20)")
    
    try:
        start_time = time.time()
        
        # Process the request through RAG pipeline
        answers = await process_hackathon_request(request.documents, request.questions)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"✅ Hackathon request completed in {total_time:.2f}ms")
        print(f"📊 Processed {len(request.questions)} questions")
        print(f"📈 Cache hit rate: {cache.hit_rate():.2%}")
        
        return HackathonResponse(answers=answers)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error processing hackathon request: {error_msg}")
        
        # Return structured error response
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {error_msg}"
        )

# Single question endpoint for testing
@app.post("/ask")
async def ask_question(
    question: str,
    pdf_url: Optional[str] = None
):
    """Single question endpoint for development/testing"""
    
    if pdf_url:
        # Process new PDF
        answers = await process_hackathon_request(pdf_url, [question])
        return {"answer": answers[0] if answers else "No answer generated"}
    else:
        # Use existing indexed documents
        from rag_pipeline import rag_pipeline
        answer = await rag_pipeline.answer_question_optimized(question)
        return {"answer": answer}

# System metrics dashboard
@app.get("/metrics", response_class=HTMLResponse)
async def metrics_dashboard():
    """Simple metrics dashboard"""
    
    stats = get_system_stats()
    uptime = time.time() - app.startup_time
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG System Metrics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
            .status-good {{ color: green; }}
            .status-warning {{ color: orange; }}
            .status-error {{ color: red; }}
        </style>
        <script>
            setTimeout(() => location.reload(), 30000); // Auto-refresh every 30s
        </script>
    </head>
    <body>
        <h1>⚡ Lightning RAG System Metrics</h1>
        
        <div class="metric">
            <strong>System Status:</strong> 
            <span class="status-good">✅ Running</span>
        </div>
        
        <div class="metric">
            <strong>Uptime:</strong> {uptime:.1f} seconds
        </div>
        
        <div class="metric">
            <strong>Cache Hit Rate:</strong> 
            <span class="{'status-good' if stats['cache_hit_rate'] > 0.5 else 'status-warning'}">
                {stats['cache_hit_rate']:.2%}
            </span>
        </div>
        
        <div class="metric">
            <strong>Average Response Time:</strong> {stats['avg_response_time_ms']:.2f}ms
        </div>
        
        <div class="metric">
            <strong>Total Queries:</strong> {stats['total_queries']}
        </div>
        
        <div class="metric">
            <strong>Vector Store Documents:</strong> {stats['vector_store']['document_count']}
        </div>
        
        <div class="metric">
            <strong>Memory Usage:</strong> {stats['memory_usage_mb']:.1f} MB
        </div>
        
        <div class="metric">
            <strong>System Warmed Up:</strong> 
            <span class="{'status-good' if stats['is_warmed_up'] else 'status-warning'}">
                {'Yes' if stats['is_warmed_up'] else 'No'}
            </span>
        </div>
        
        <p><em>Auto-refreshes every 30 seconds</em></p>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Cache management endpoints
@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    cache.query_cache.clear()
    cache.embedding_cache.clear()
    cache.hit_count = 0
    cache.miss_count = 0
    return {"message": "Cache cleared successfully"}

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "hit_rate": cache.hit_rate(),
        "hit_count": cache.hit_count,
        "miss_count": cache.miss_count,
        "query_cache_size": len(cache.query_cache),
        "embedding_cache_size": len(cache.embedding_cache),
        "avg_response_time_ms": cache.avg_response_time()
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "⚡ Lightning-Fast RAG System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "hackathon": "/hackrx/run",
            "metrics": "/metrics",
            "ask": "/ask",
            "cache_stats": "/cache/stats"
        },
        "status": "ready",
        "performance_target": "<300ms response time"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Production-like settings for local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for local development
        loop="uvloop",
        access_log=False,  # Disable for better performance
        reload=False,
        log_level="info"
    )
