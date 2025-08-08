import google.generativeai as genai
import asyncio
import time
from typing import List, Optional
from cache_system import cache, CircuitBreaker
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Circuit breaker for embedding API
embedding_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

class EmbeddingService:
    """Optimized embedding service with caching and batching"""
    
    def __init__(self):
        self.model_name = "models/text-embedding-004"
        self.max_batch_size = int(os.getenv("MAX_EMBEDDING_BATCH_SIZE", "50"))
    
    async def get_embedding_single(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Get embedding for a single text with caching"""
        
        # Check cache first
        cached_embedding = cache.get_cached_embedding_direct(text)
        if cached_embedding:
            return cached_embedding
        
        try:
            # Use circuit breaker for reliability
            result = await embedding_breaker.call(
                self._generate_embedding, 
                text, 
                task_type
            )
            
            # Cache the result
            cache.cache_embedding(text, result)
            return result
            
        except Exception as e:
            print(f"❌ Embedding error for text: {str(e)[:100]}...")
            # Return zero vector as fallback
            return [0.0] * 768  # text-embedding-004 has 768 dimensions
    
    async def get_embedding_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Get embeddings for multiple texts with optimal batching"""
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = cache.get_cached_embedding_direct(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batches
        if uncached_texts:
            try:
                uncached_embeddings = await self._generate_embeddings_batch(uncached_texts, task_type)
                
                # Fill in the placeholders and cache results
                for idx, embedding in zip(uncached_indices, uncached_embeddings):
                    embeddings[idx] = embedding
                    cache.cache_embedding(texts[idx], embedding)
                    
            except Exception as e:
                print(f"❌ Batch embedding error: {str(e)}")
                # Fill with zero vectors as fallback
                for idx in uncached_indices:
                    embeddings[idx] = [0.0] * 768
        
        return embeddings
    
    async def _generate_embedding(self, text: str, task_type: str) -> List[float]:
        """Generate single embedding using Google API"""
        result = await genai.embed_content_async(
            model=self.model_name,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    
    async def _generate_embeddings_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Generate batch embeddings using Google API"""
        all_embeddings = []
        
        # Process in smaller batches to respect API limits
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            result = await genai.embed_content_async(
                model=self.model_name,
                content=batch,
                task_type=task_type
            )
            
            all_embeddings.extend(result['embedding'])
            
            # Small delay to avoid rate limiting
            if i + self.max_batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings

# Global embedding service instance
embedding_service = EmbeddingService()

# Convenience functions for backward compatibility
async def get_embedding_batch(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Get embeddings for multiple texts"""
    return await embedding_service.get_embedding_batch(texts, task_type)

async def get_embedding_single(text: str, task_type: str = "retrieval_query") -> List[float]:
    """Get embedding for a single text (typically queries)"""
    return await embedding_service.get_embedding_single(text, task_type)
