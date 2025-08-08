"""
FastAPI RAG Application - Simplified Version
Lightning-fast RAG system using FAISS and optimized components
"""
import asyncio
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Try to install uvloop for better async performance
try:
    import uvloop
    uvloop.install()
    print("✅ uvloop installed for better async performance")
except ImportError:
    print("⚠️ uvloop not available, using default asyncio event loop")

# Import our simplified RAG components
from rag_pipeline_simple import (
    get_rag_pipeline, 
    answer_question, 
    ingest_documents,
    warmup_pipeline,
    get_pipeline_stats
)
from simple_pdf_processor import extract_pdf_text

load_dotenv()

# Pydantic models for API
class HackathonRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

class QuestionRequest(BaseModel):
    question: str

class DocumentRequest(BaseModel):
    documents: List[str]

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    # Startup
    print("🚀 Starting RAG system...")
    
    await warmup_pipeline()
    
    yield
    
    # Shutdown
    print("🛑 Shutting down RAG system...")

# Create FastAPI app
app = FastAPI(
    title="Lightning RAG API",
    description="Ultra-fast RAG system with FAISS vector store",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add performance monitoring middleware
@app.middleware("http")
async def add_performance_headers(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
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

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "message": "Lightning RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "hackathon": "/hackrx/run",
            "question": "/question",
            "ingest": "/ingest",
            "health": "/health",
            "stats": "/stats"
        }
    }

async def download_pdf_from_url(url: str) -> bytes:
    """Download PDF from URL"""
    
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                print(f"⚠️ Content-Type is {content_type}, but proceeding anyway")
            
            return response.content
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="PDF download timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error: {e.response.text}")

@app.post("/hackrx/run", response_model=HackathonResponse)
async def hackrx_endpoint(
    request: HackathonRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Main hackathon endpoint - Download PDF, process, and answer questions
    Expected to complete in <300ms for cached responses
    """
    start_time = time.time()
    
    try:
        print(f"📥 Processing hackathon request with {len(request.questions)} questions")
        
        # Download PDF
        pdf_bytes = await download_pdf_from_url(request.documents)
        
        # Extract text from PDF
        pdf_text = extract_pdf_text(pdf_bytes)
        
        if not pdf_text or len(pdf_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF")
        
        # Ingest document
        ingest_result = await ingest_documents([pdf_text])
        
        if not ingest_result.get("success"):
            raise HTTPException(status_code=500, detail=f"Document ingestion failed: {ingest_result.get('error')}")
        
        # Answer all questions
        answers = []
        for question in request.questions:
            result = await answer_question(question)
            answers.append(result.get("answer", "Sorry, I couldn't process this question."))
        
        processing_time = time.time() - start_time
        
        print(f"✅ Processed {len(request.questions)} questions in {processing_time:.3f}s")
        
        return HackathonResponse(answers=answers)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Hackathon endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/question")
async def ask_question(request: QuestionRequest):
    """Ask a single question against the knowledge base"""
    
    try:
        result = await answer_question(request.question)
        return result
    
    except Exception as e:
        print(f"❌ Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents_endpoint(request: DocumentRequest):
    """Ingest documents into the knowledge base"""
    
    try:
        result = await ingest_documents(request.documents)
        return result
    
    except Exception as e:
        print(f"❌ Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    
    stats = get_pipeline_stats()
    
    return {
        "timestamp": time.time(),
        "pipeline": stats
    }

@app.get("/dashboard")
async def dashboard():
    """Simple dashboard for monitoring"""
    
    stats = get_pipeline_stats()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lightning RAG Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f0f0f0; padding: 20px; margin: 10px 0; border-radius: 5px; }}
            .status {{ color: green; font-weight: bold; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>⚡ Lightning RAG Dashboard</h1>
        <div class="metric">
            <h3>🔥 Pipeline Status</h3>
            <p class="status">Warmed Up: {stats.get('pipeline_warmed_up', False)}</p>
        </div>
        <div class="metric">
            <h3>📊 Vector Store</h3>
            <p>Documents: {stats.get('vector_store', {}).get('total_documents', 0)}</p>
            <p>Index Type: {stats.get('vector_store', {}).get('index_type', 'N/A')}</p>
            <p>Dimension: {stats.get('vector_store', {}).get('dimension', 0)}</p>
        </div>
        <div class="metric">
            <h3>⚙️ Configuration</h3>
            <p>Max Concurrency: {stats.get('max_concurrency', 0)}</p>
            <p>Max Context Tokens: {stats.get('max_context_tokens', 0)}</p>
        </div>
        <p><small>Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "127.0.0.1")  # Default to localhost
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting Lightning RAG API on {host}:{port}")
    print(f"📊 Dashboard available at: http://{host}:{port}/dashboard")
    print(f"🔍 Health check at: http://{host}:{port}/health")
    print(f"📖 API docs at: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main_simple:app",  # Fixed to reference the correct module
        host=host,
        port=port,
        reload=False,
        access_log=True,
        loop="uvloop" if 'uvloop' in globals() else "asyncio"
    )
