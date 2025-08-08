import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class VectorStore:
    """Optimized ChromaDB vector store for ultra-fast search"""
    
    def __init__(self):
        # Ultra-fast in-memory configuration
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=None,  # In-memory mode for speed
            anonymized_telemetry=False
        ))
        
        self.collection = None
        self.collection_name = "documents"
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or recreate the collection with optimized settings"""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass  # Collection doesn't exist, that's fine
            
            # Create new collection with optimized HNSW settings
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": 16,  # Optimized for speed vs accuracy balance
                    "hnsw:ef_construction": 200,
                    "hnsw:ef": 10
                }
            )
            print("✅ Vector store initialized with optimized settings")
            
        except Exception as e:
            print(f"❌ Error initializing vector store: {str(e)}")
            raise
    
    async def add_documents_batch(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ):
        """Add documents to the vector store in batches"""
        
        if not documents:
            return
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]
        
        # Generate metadata if not provided
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(documents))]
        
        try:
            # Use thread executor for blocking ChromaDB operations
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(
                    executor,
                    self._add_documents_sync,
                    documents, embeddings, metadatas, ids
                )
            
            print(f"✅ Added {len(documents)} documents to vector store")
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise
    
    def _add_documents_sync(self, documents, embeddings, metadatas, ids):
        """Synchronous document addition for thread executor"""
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        where: Optional[Dict] = None,
        distance_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with optimized performance"""
        
        try:
            # Use thread executor for blocking ChromaDB operations
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                results = await loop.run_in_executor(
                    executor,
                    self._search_sync,
                    query_embedding, top_k, where
                )
            
            # Filter by distance threshold and format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0.0] * len(results['documents'][0])
                )):
                    if distance < distance_threshold:
                        formatted_results.append({
                            'content': doc,
                            'metadata': metadata or {},
                            'distance': distance,
                            'similarity': 1 - distance  # Convert distance to similarity
                        })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching vector store: {str(e)}")
            return []
    
    def _search_sync(self, query_embedding, top_k, where):
        """Synchronous search for thread executor"""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )
    
    async def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(
                    executor,
                    self._clear_collection_sync
                )
            print("✅ Vector store cleared")
            
        except Exception as e:
            print(f"❌ Error clearing vector store: {str(e)}")
    
    def _clear_collection_sync(self):
        """Synchronous collection clearing"""
        # Recreate collection to clear all data
        self._initialize_collection()
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "status": "ready"
            }
        except Exception as e:
            return {
                "document_count": 0,
                "collection_name": self.collection_name,
                "status": f"error: {str(e)}"
            }

# Global vector store instance
vector_store = VectorStore()

# Convenience functions
async def add_documents(documents: List[str], embeddings: List[List[float]], metadatas: List[Dict] = None):
    """Add documents to the vector store"""
    await vector_store.add_documents_batch(documents, embeddings, metadatas)

async def search_vectors(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Search for similar vectors"""
    return await vector_store.search_similar(query_embedding, top_k)

async def clear_vector_store():
    """Clear the vector store"""
    await vector_store.clear_collection()
