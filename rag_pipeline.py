import asyncio
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Import our simplified services
from embedding_service import get_embeddings
from faiss_vector_store import get_vector_store, add_documents, search_documents
from simple_document_processor import process_documents, prepare_context
from llm_service import generate_response
from simple_pdf_processor import process_pdf_files
from cache_system import get_cache

class RAGPipeline:
    """Optimized RAG pipeline for ultra-fast processing"""
    
    def __init__(self):
        self.max_question_concurrency = 5
        self.max_context_tokens = 2000
        self.is_warmed_up = False
    
    async def warm_up_system(self):
        """Pre-load models and warm up caches for faster first request"""
        
        if self.is_warmed_up:
            return
        
        print("🔥 Warming up RAG system...")
        start_time = time.time()
        
        try:
            # Warm up embedding model with common queries
            warmup_texts = [
                "what is the grace period",
                "waiting period for coverage", 
                "maternity benefits",
                "no claim discount",
                "hospital definition"
            ]
            
            await get_embedding_batch(warmup_texts, task_type="retrieval_query")
            
            # Warm up LLM with a test query
            await generate_answer_with_context(
                "What is this policy about?",
                "This is a sample insurance policy document for testing purposes."
            )
            
            self.is_warmed_up = True
            warmup_time = (time.time() - start_time) * 1000
            print(f"✅ System warmed up in {warmup_time:.2f}ms")
            
        except Exception as e:
            print(f"❌ Warmup failed: {str(e)}")
    
    async def process_and_index_documents(self, content: str):
        """Process PDF content and build vector index efficiently"""
        
        start_time = time.time()
        
        try:
            # Clear existing collection
            await clear_vector_store()
            
            # Process documents into chunks
            chunks = await process_documents([content])
            
            if not chunks:
                raise ValueError("No valid chunks extracted from document")
            
            # Extract texts for embedding
            texts = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = await get_embedding_batch(texts, task_type="retrieval_document")
            
            # Add to vector store
            await add_documents(texts, embeddings, metadatas)
            
            processing_time = (time.time() - start_time) * 1000
            print(f"✅ Indexed {len(chunks)} chunks in {processing_time:.2f}ms")
            
        except Exception as e:
            print(f"❌ Error in document processing: {str(e)}")
            raise
    
    async def answer_question_optimized(self, question: str) -> str:
        """Answer a single question with optimized pipeline"""
        
        try:
            start_time = time.time()
            
            # Step 1: Get question embedding
            question_embedding = await get_embedding_single(question, task_type="retrieval_query")
            
            # Step 2: Search vector store
            search_results = await search_vectors(question_embedding, top_k=5)
            
            if not search_results:
                return "I couldn't find relevant information to answer this question."
            
            # Step 3: Prepare context from top results
            context = prepare_context(search_results, max_tokens=self.max_context_tokens)
            
            # Step 4: Generate answer
            answer = await generate_answer_with_context(question, context)
            
            response_time = (time.time() - start_time) * 1000
            cache.record_response_time(response_time)
            
            print(f"✅ Answered question in {response_time:.2f}ms")
            return answer.strip()
            
        except Exception as e:
            print(f"❌ Error answering question: {str(e)}")
            return "I encountered an error while processing your question. Please try again."
    
    async def answer_questions_batch(self, questions: List[str]) -> List[str]:
        """Answer multiple questions efficiently with controlled concurrency"""
        
        if not questions:
            return []
        
        start_time = time.time()
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.max_question_concurrency)
        
        async def answer_with_semaphore(question: str) -> str:
            async with semaphore:
                return await self.answer_question_optimized(question)
        
        try:
            # Process all questions in parallel with limited concurrency
            tasks = [answer_with_semaphore(q) for q in questions]
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    print(f"❌ Error answering question {i}: {str(answer)}")
                    final_answers.append("Unable to answer this question due to processing error.")
                else:
                    final_answers.append(answer)
            
            total_time = (time.time() - start_time) * 1000
            print(f"✅ Answered {len(questions)} questions in {total_time:.2f}ms")
            
            return final_answers
            
        except Exception as e:
            print(f"❌ Error in batch processing: {str(e)}")
            # Return error messages for all questions
            return ["Error processing questions. Please try again."] * len(questions)
    
    async def process_hackathon_request(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Main pipeline for hackathon endpoint"""
        
        overall_start = time.time()
        
        try:
            # Ensure system is warmed up
            await self.warm_up_system()
            
            # Step 1: Download and extract PDF (parallel with warmup if not done)
            print(f"📄 Processing PDF: {pdf_url}")
            pdf_content = await download_and_extract_pdf(pdf_url)
            
            if not pdf_content or len(pdf_content.strip()) < 100:
                raise ValueError("PDF content is too short or empty")
            
            # Step 2: Process and index documents
            await self.process_and_index_documents(pdf_content)
            
            # Step 3: Answer all questions
            answers = await self.answer_questions_batch(questions)
            
            total_time = (time.time() - overall_start) * 1000
            print(f"🎯 Complete pipeline finished in {total_time:.2f}ms")
            
            return answers
            
        except Exception as e:
            print(f"❌ Pipeline error: {str(e)}")
            error_message = f"Pipeline processing error: {str(e)}"
            return [error_message] * len(questions)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        
        vector_info = vector_store.get_collection_info()
        
        return {
            "cache_hit_rate": cache.hit_rate(),
            "avg_response_time_ms": cache.avg_response_time(),
            "total_queries": cache.total_queries,
            "vector_store": vector_info,
            "is_warmed_up": self.is_warmed_up,
            "memory_usage_mb": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()

# Convenience functions
async def process_hackathon_request(pdf_url: str, questions: List[str]) -> List[str]:
    """Process hackathon request"""
    return await rag_pipeline.process_hackathon_request(pdf_url, questions)

async def warm_up_system():
    """Warm up the system"""
    await rag_pipeline.warm_up_system()

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics"""
    return rag_pipeline.get_system_stats()
