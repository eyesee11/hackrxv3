import asyncio
import time
import hashlib
import io
from functools import lru_cache
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pickle

# Try to install uvloop for better async performance, but don't fail if not available
try:
    import uvloop
    uvloop.install()
    print("✅ uvloop installed for better async performance")
except ImportError:
    print("⚠️ uvloop not available, using default asyncio event loop")

class AdvancedCache:
    """Multi-level caching system for RAG components"""
    
    def __init__(self):
        self.embedding_cache: Dict[str, Any] = {}
        self.query_cache: Dict[str, Any] = {}
        self.context_cache: Dict[str, Any] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.total_queries = 0
        self.response_times = []
    
    def cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_cached_embedding(self, text: str):
        """LRU cache for embeddings - these never change"""
        # This will be populated by the actual embedding function
        return None
    
    def cache_query_result(self, query: str, result: dict):
        """Cache complete query results with timestamp"""
        key = self.cache_key(query)
        self.query_cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def get_cached_query(self, query: str, max_age: int = 300) -> Optional[dict]:
        """Get cached query if not expired (5 minutes default)"""
        key = self.cache_key(query)
        if key in self.query_cache:
            cached = self.query_cache[key]
            if time.time() - cached['timestamp'] < max_age:
                self.hit_count += 1
                return cached['result']
        
        self.miss_count += 1
        return None
    
    def cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding result"""
        key = self.cache_key(text)
        self.embedding_cache[key] = embedding
    
    def get_cached_embedding_direct(self, text: str) -> Optional[List[float]]:
        """Get cached embedding directly"""
        key = self.cache_key(text)
        return self.embedding_cache.get(key)
    
    def record_response_time(self, response_time: float):
        """Record response time for metrics"""
        self.response_times.append(response_time)
        if len(self.response_times) > 1000:  # Keep only last 1000
            self.response_times = self.response_times[-1000:]
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def avg_response_time(self) -> float:
        """Calculate average response time"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    def clear_expired(self, max_age: int = 300):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.query_cache.items()
            if current_time - data['timestamp'] > max_age
        ]
        for key in expired_keys:
            del self.query_cache[key]

class CircuitBreaker:
    """Circuit breaker pattern for API reliability"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self.last_failure_time and datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.last_failure_time = datetime.now()
            raise e

# Global cache instance
cache = AdvancedCache()

def get_cache() -> AdvancedCache:
    """Get the global cache instance"""
    return cache

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024
