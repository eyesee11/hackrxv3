"""
Enhanced RAG Pipeline with tiktoken support
Lightning-fast RAG system following the tech stack specifications
"""
import asyncio
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Import our enhanced services
from embedding_service import get_embedding_batch, get_embedding_single
from faiss_vector_store import get_vector_store, add_documents, search_documents
from enhanced_document_processor import process_documents_enhanced, prepare_context_enhanced
from llm_service import generate_response
from simple_pdf_processor import process_pdf_files
from cache_system import get_cache

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline following the tech stack specifications"""
    
    def __init__(self):
        # Performance settings from tech stack
        self.max_question_concurrency = 5
        self.max_context_tokens = 2000
        self.embedding_batch_size = 50  # Google's recommended batch size
        self.is_warmed_up = False
        
        # Initialize components
        self.cache = get_cache()
        self.vector_store = get_vector_store()
        
        print("🚀 Enhanced RAG Pipeline initialized with tiktoken support")
    
    async def warmup(self):
        """Comprehensive system warmup as per tech stack specs"""
        if self.is_warmed_up:
            return
        
        print("🔥 Warming up enhanced RAG pipeline...")
        
        try:
            # Pre-load and warm up embedding model
            warmup_texts = [
                "What is the main topic?",
                "Sample document content for testing",
                "Insurance policy question example",
                "Medical coverage information"
            ]
            await get_embedding_batch(warmup_texts)
            
            # Warm up LLM service with realistic prompt
            await generate_response(
                "What is covered under this policy?",
                "This is a sample insurance policy document with coverage details."
            )
            
            # Pre-compute common query embeddings (as per tech stack recommendation)
            common_queries = ["what is", "how to", "explain", "summarize", "coverage", "benefits"]
            for query in common_queries:
                embedding = await get_embedding_single(query)
                # Cache the embedding
                self.cache.cache_embedding(query, embedding)
            
            self.is_warmed_up = True
            print("✅ Enhanced RAG pipeline warmed up successfully")
            
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")
    
    async def clear_knowledge_base(self):
        """Clear all documents from vector store"""
        try:
            await self.vector_store.clear()
            print("✅ Knowledge base cleared")
            
        except Exception as e:
            print(f"❌ Failed to clear knowledge base: {e}")
            raise
    
    async def ingest_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Enhanced document ingestion with optimized processing"""
        start_time = time.time()
        
        try:
            # Process documents using enhanced processor with tiktoken
            chunks = await process_documents_enhanced(documents)
            
            if not chunks:
                return {"success": False, "error": "No valid chunks extracted"}
            
            # Get embeddings in optimal batches
            texts = [chunk['content'] for chunk in chunks]
            embeddings = await self._get_embeddings_batched(texts)
            
            # Prepare documents for vector store with enhanced metadata
            docs_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                docs_with_embeddings.append({
                    'content': chunk['content'],
                    'embedding': embedding,
                    'metadata': chunk['metadata']
                })
            
            # Add to FAISS vector store
            doc_ids = await add_documents(docs_with_embeddings)
            
            processing_time = time.time() - start_time
            
            print(f"✅ Enhanced processing: {len(documents)} docs → {len(chunks)} chunks in {processing_time:.2f}s")
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "processing_time": processing_time,
                "total_tokens": sum(chunk['metadata']['token_count'] for chunk in chunks)
            }
            
        except Exception as e:
            print(f"❌ Enhanced document ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_embeddings_batched(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings in optimal batches with caching"""
        
        all_embeddings = []
        
        # Process in batches for optimal performance
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            
            # Check cache first
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                cached = self.cache.get_cached_embedding_direct(text)
                if cached is not None:
                    cached_embeddings.append((i + j, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i + j)
            
            # Get embeddings for uncached texts
            if uncached_texts:
                new_embeddings = await get_embedding_batch(uncached_texts)
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.cache.cache_embedding(text, embedding)
            else:
                new_embeddings = []
            
            # Combine cached and new embeddings in correct order
            batch_embeddings = [None] * len(batch)
            
            # Place cached embeddings
            for idx, embedding in cached_embeddings:
                relative_idx = idx - i
                batch_embeddings[relative_idx] = embedding
            
            # Place new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                relative_idx = idx - i
                batch_embeddings[relative_idx] = embedding
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def ingest_pdf_files(self, pdf_files: List[bytes]) -> Dict[str, Any]:
        """Enhanced PDF ingestion with better text extraction"""
        start_time = time.time()
        
        try:
            # Extract text from PDFs using multiple methods
            pdf_texts = await process_pdf_files(pdf_files)
            
            if not pdf_texts:
                return {"success": False, "error": "No text extracted from PDFs"}
            
            # Process as regular documents with enhanced processing
            result = await self.ingest_documents(pdf_texts)
            result["pdf_files_processed"] = len(pdf_files)
            
            return result
            
        except Exception as e:
            print(f"❌ Enhanced PDF ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Enhanced question answering with caching and optimization"""
        start_time = time.time()
        
        try:
            # Check cache first for complete query results
            cached_result = self.cache.get_cached_query(question)
            if cached_result:
                cached_result["response_time"] = time.time() - start_time
                cached_result["cache_hit"] = True
                return cached_result
            
            # Get question embedding (with caching)
            question_embedding = await get_embedding_single(question)
            
            # Search for relevant documents using FAISS
            search_results = await search_documents(question_embedding, k=5)
            
            if not search_results:
                result = {
                    "answer": "I don't have enough information to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "cache_hit": False
                }
            else:
                # Prepare context using enhanced processor with token awareness
                context = prepare_context_enhanced(search_results, max_tokens=self.max_context_tokens)
                
                # Generate answer with optimized prompt
                answer = await generate_response(question, context)
                
                result = {
                    "answer": answer,
                    "confidence": search_results[0]['similarity'] if search_results else 0.0,
                    "sources": [
                        {
                            "content": r['content'][:200] + "...",
                            "similarity": r['similarity'],
                            "metadata": r.get('metadata', {})
                        } for r in search_results[:3]
                    ],
                    "cache_hit": False
                }
            
            response_time = time.time() - start_time
            result["response_time"] = response_time
            
            # Cache the result
            self.cache.cache_query_result(question, result)
            
            return result
            
        except Exception as e:
            print(f"❌ Enhanced question answering failed: {e}")
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
                "cache_hit": False
            }
    
    async def batch_answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Enhanced batch question processing with optimal concurrency"""
        
        # Limit concurrency as per tech stack recommendations
        semaphore = asyncio.Semaphore(self.max_question_concurrency)
        
        async def answer_with_semaphore(question: str):
            async with semaphore:
                return await self.answer_question(question)
        
        # Process all questions concurrently
        tasks = [answer_with_semaphore(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "answer": f"Error processing question: {result}",
                    "confidence": 0.0,
                    "sources": [],
                    "error": str(result),
                    "cache_hit": False
                })
            else:
                formatted_results.append(result)
        
        return formatted_results
    
    async def search_similar(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced similarity search with metadata filtering"""
        
        try:
            # Get query embedding
            query_embedding = await get_embedding_single(query)
            
            # Search for similar documents
            search_results = await search_documents(query_embedding, k=k)
            
            return search_results
            
        except Exception as e:
            print(f"❌ Enhanced similarity search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Enhanced pipeline statistics"""
        
        vector_stats = self.vector_store.get_stats()
        cache_stats = {
            "hit_rate": self.cache.hit_rate(),
            "avg_response_time_ms": self.cache.avg_response_time(),
            "total_queries": getattr(self.cache, 'total_queries', 0)
        }
        
        return {
            "pipeline_warmed_up": self.is_warmed_up,
            "vector_store": vector_stats,
            "cache": cache_stats,
            "max_concurrency": self.max_question_concurrency,
            "max_context_tokens": self.max_context_tokens,
            "embedding_batch_size": self.embedding_batch_size,
            "tiktoken_enabled": True
        }

# Global enhanced pipeline instance
enhanced_rag_pipeline: EnhancedRAGPipeline = None

def get_enhanced_rag_pipeline() -> EnhancedRAGPipeline:
    """Get or create global enhanced RAG pipeline instance"""
    global enhanced_rag_pipeline
    
    if enhanced_rag_pipeline is None:
        enhanced_rag_pipeline = EnhancedRAGPipeline()
    
    return enhanced_rag_pipeline

# Convenience functions for direct use
async def ingest_documents_enhanced(documents: List[str]) -> Dict[str, Any]:
    """Ingest documents using enhanced pipeline"""
    pipeline = get_enhanced_rag_pipeline()
    return await pipeline.ingest_documents(documents)

async def ingest_pdf_files_enhanced(pdf_files: List[bytes]) -> Dict[str, Any]:
    """Ingest PDF files using enhanced pipeline"""
    pipeline = get_enhanced_rag_pipeline()
    return await pipeline.ingest_pdf_files(pdf_files)

async def answer_question_enhanced(question: str) -> Dict[str, Any]:
    """Answer a single question using enhanced pipeline"""
    pipeline = get_enhanced_rag_pipeline()
    return await pipeline.answer_question(question)

async def batch_answer_questions_enhanced(questions: List[str]) -> List[Dict[str, Any]]:
    """Answer multiple questions using enhanced pipeline"""
    pipeline = get_enhanced_rag_pipeline()
    return await pipeline.batch_answer_questions(questions)

async def search_similar_enhanced(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Search for similar documents using enhanced pipeline"""
    pipeline = get_enhanced_rag_pipeline()
    return await pipeline.search_similar(query, k)

async def warmup_enhanced_pipeline():
    """Warmup the enhanced RAG pipeline"""
    pipeline = get_enhanced_rag_pipeline()
    await pipeline.warmup()

async def clear_enhanced_knowledge_base():
    """Clear the enhanced knowledge base"""
    pipeline = get_enhanced_rag_pipeline()
    await pipeline.clear_knowledge_base()

def get_enhanced_pipeline_stats() -> Dict[str, Any]:
    """Get enhanced pipeline statistics"""
    pipeline = get_enhanced_rag_pipeline()
    return pipeline.get_stats()
