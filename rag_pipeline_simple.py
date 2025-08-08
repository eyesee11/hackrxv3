"""
Enhanced RAG Pipeline with High Impact, Low Effort Optimizations
- Improved chunking with overlap and metadata
- Enhanced prompts with few-shot examples
- Hybrid search (vector + keyword)
- Response validation
"""
import asyncio
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Import our enhanced services
from embedding_service import get_embedding_batch, get_embedding_single
from faiss_vector_store import get_vector_store, add_documents, search_documents
from simple_document_processor import process_documents, prepare_context
from llm_service import generate_response
from simple_pdf_processor import process_pdf_files
from cache_system import get_cache
from hybrid_search import fit_hybrid_search, perform_hybrid_search
from response_validator import validate_response

class RAGPipeline:
    """Enhanced RAG pipeline with accuracy optimizations"""
    
    def __init__(self):
        self.max_question_concurrency = 5
        self.max_context_tokens = 2500  # Increased for better context
        self.is_warmed_up = False
        self.cache = get_cache()
        self.vector_store = get_vector_store()
        self.documents_indexed = 0
        self.hybrid_search_fitted = False
    
    async def warmup(self):
        """Warmup all services for optimal performance"""
        if self.is_warmed_up:
            return
        
        print("🔥 Warming up RAG pipeline...")
        
        try:
            # Warmup embeddings service
            warmup_texts = ["What is the main topic?", "Sample document content"]
            await get_embedding_batch(warmup_texts)
            
            # Warmup LLM service  
            await generate_response(
                "What is AI?",
                "AI is artificial intelligence"
            )
            
            self.is_warmed_up = True
            print("✅ RAG pipeline warmed up successfully")
            
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
        """Ingest documents into the knowledge base"""
        start_time = time.time()
        
        try:
            # Process documents into chunks
            chunks = await process_documents(documents)
            
            if not chunks:
                return {"success": False, "error": "No valid chunks extracted"}
            
            # Get embeddings for all chunks
            texts = [chunk['content'] for chunk in chunks]
            embeddings = await get_embedding_batch(texts)
            
            # Prepare documents for vector store
            docs_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                docs_with_embeddings.append({
                    'content': chunk['content'],
                    'embedding': embedding,
                    'metadata': chunk['metadata']
                })
            
            # Add to vector store
            doc_ids = await add_documents(docs_with_embeddings)
            
            # Fit hybrid search on the processed chunks
            if not self.hybrid_search_fitted:
                fit_hybrid_search(chunks)
                self.hybrid_search_fitted = True
                print("✅ Hybrid search fitted on document corpus")
            
            self.documents_indexed += len(chunks)
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "total_indexed": self.documents_indexed,
                "processing_time": processing_time,
                "hybrid_search_ready": self.hybrid_search_fitted
            }
            
        except Exception as e:
            print(f"❌ Document ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def ingest_pdf_files(self, pdf_files: List[bytes]) -> Dict[str, Any]:
        """Ingest PDF files into the knowledge base"""
        start_time = time.time()
        
        try:
            # Extract text from PDFs
            pdf_texts = await process_pdf_files(pdf_files)
            
            if not pdf_texts:
                return {"success": False, "error": "No text extracted from PDFs"}
            
            # Process as regular documents
            result = await self.ingest_documents(pdf_texts)
            result["pdf_files_processed"] = len(pdf_files)
            
            return result
            
        except Exception as e:
            print(f"❌ PDF ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using enhanced retrieval and validation"""
        start_time = time.time()
        
        try:
            # Get question embedding
            question_embedding = await get_embedding_single(question)
            
            # Search for relevant documents (get more candidates for hybrid search)
            search_results = await search_documents(question_embedding, k=15)
            
            if not search_results:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "validation": {"is_valid": False, "issues": ["No relevant documents found"]}
                }
            
            # Apply hybrid search if available
            if self.hybrid_search_fitted:
                # Convert search results to format expected by hybrid search
                vector_scores = [(i, result['similarity']) for i, result in enumerate(search_results)]
                hybrid_results = await perform_hybrid_search(question, vector_scores, top_k=8)
                
                # Convert back to original format
                final_results = []
                for result in hybrid_results:
                    if result['index'] < len(search_results):
                        original_result = search_results[result['index']]
                        original_result['similarity'] = result['similarity']  # Update with hybrid score
                        original_result['hybrid_scores'] = {
                            'vector': result.get('vector_score', 0),
                            'keyword': result.get('keyword_score', 0)
                        }
                        final_results.append(original_result)
                
                print(f"🔍 Hybrid search returned {len(final_results)} results")
            else:
                final_results = search_results[:8]  # Use top vector results
                print(f"🔍 Vector search returned {len(final_results)} results")
            
            # Prepare enhanced context
            context = prepare_context(final_results, max_chars=self.max_context_tokens)
            
            # Generate answer with enhanced prompts
            answer = await generate_response(question, context)
            
            # Validate response
            validation_result = await validate_response(question, answer, context)
            
            # Calculate enhanced confidence score
            base_confidence = final_results[0]['similarity'] if final_results else 0.0
            validation_confidence = validation_result.confidence_score
            
            # Combined confidence (weighted average)
            combined_confidence = (0.6 * base_confidence + 0.4 * validation_confidence)
            
            response_time = time.time() - start_time
            
            return {
                "answer": answer,
                "confidence": combined_confidence,
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "confidence_score": validation_result.confidence_score,
                    "factual_accuracy": validation_result.factual_accuracy,
                    "completeness_score": validation_result.completeness_score,
                    "issues": validation_result.issues,
                    "suggestions": validation_result.suggestions
                },
                "sources": [
                    {
                        "content": r['content'][:300] + "...",
                        "similarity": r['similarity'],
                        "metadata": r.get('metadata', {}),
                        "hybrid_scores": r.get('hybrid_scores', {})
                    } for r in final_results[:3]
                ],
                "response_time": response_time,
                "search_method": "hybrid" if self.hybrid_search_fitted else "vector_only"
            }
            
        except Exception as e:
            print(f"❌ Question answering failed: {e}")
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "confidence": 0.0,
                "sources": [],
                "validation": {"is_valid": False, "issues": [f"Processing error: {str(e)}"]},
                "error": str(e)
            }
    
    async def batch_answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Answer multiple questions concurrently"""
        
        # Limit concurrency to avoid overwhelming services
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
                    "error": str(result)
                })
            else:
                formatted_results.append(result)
        
        return formatted_results
    
    async def search_similar(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents without generating an answer"""
        
        try:
            # Get query embedding
            query_embedding = await get_embedding_single(query)
            
            # Search for similar documents
            search_results = await search_documents(query_embedding, k=k)
            
            return search_results
            
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced pipeline statistics"""
        
        vector_stats = self.vector_store.get_stats()
        
        return {
            "pipeline_warmed_up": self.is_warmed_up,
            "vector_store": vector_stats,
            "max_concurrency": self.max_question_concurrency,
            "max_context_tokens": self.max_context_tokens,
            "documents_indexed": self.documents_indexed,
            "hybrid_search_fitted": self.hybrid_search_fitted,
            "optimizations": {
                "enhanced_chunking": True,
                "overlapping_chunks": True,
                "metadata_extraction": True,
                "few_shot_prompts": True,
                "hybrid_search": self.hybrid_search_fitted,
                "response_validation": True,
                "smart_overlap": True,
                "policy_term_detection": True
            }
        }

# Global pipeline instance
rag_pipeline: RAGPipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or create global RAG pipeline instance"""
    global rag_pipeline
    
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    
    return rag_pipeline

# Convenience functions for direct use
async def ingest_documents(documents: List[str]) -> Dict[str, Any]:
    """Ingest documents into the knowledge base"""
    pipeline = get_rag_pipeline()
    return await pipeline.ingest_documents(documents)

async def ingest_pdf_files(pdf_files: List[bytes]) -> Dict[str, Any]:
    """Ingest PDF files into the knowledge base"""
    pipeline = get_rag_pipeline()
    return await pipeline.ingest_pdf_files(pdf_files)

async def answer_question(question: str) -> Dict[str, Any]:
    """Answer a single question"""
    pipeline = get_rag_pipeline()
    return await pipeline.answer_question(question)

async def batch_answer_questions(questions: List[str]) -> List[Dict[str, Any]]:
    """Answer multiple questions"""
    pipeline = get_rag_pipeline()
    return await pipeline.batch_answer_questions(questions)

async def search_similar(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Search for similar documents"""
    pipeline = get_rag_pipeline()
    return await pipeline.search_similar(query, k)

async def warmup_pipeline():
    """Warmup the RAG pipeline"""
    pipeline = get_rag_pipeline()
    await pipeline.warmup()

async def clear_knowledge_base():
    """Clear the knowledge base"""
    pipeline = get_rag_pipeline()
    await pipeline.clear_knowledge_base()

def get_pipeline_stats() -> Dict[str, Any]:
    """Get pipeline statistics"""
    pipeline = get_rag_pipeline()
    return pipeline.get_stats()
