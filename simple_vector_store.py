import numpy as np
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import os

class SimpleVectorStore:
    """Simple in-memory vector store with basic similarity search"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.dimension = None
        
        # Initialize SQLite for persistence if needed
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database for optional persistence"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            self.conn.commit()
            print("✅ Vector store database initialized")
        except Exception as e:
            print(f"❌ Error initializing database: {str(e)}")
    
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
        
        # Validate embeddings
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        # Set dimension from first embedding
        if self.dimension is None:
            self.dimension = len(embeddings[0])
        
        # Generate metadata if not provided
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(documents))]
        
        try:
            # Convert embeddings to numpy arrays for faster computation
            embeddings_np = [np.array(emb, dtype=np.float32) for emb in embeddings]
            
            # Add to in-memory storage
            start_idx = len(self.documents)
            self.documents.extend(documents)
            self.embeddings.extend(embeddings_np)
            self.metadatas.extend(metadatas)
            
            # Optionally persist to SQLite
            if self.db_path != ":memory:":
                await self._persist_documents(documents, embeddings_np, metadatas)
            
            print(f"✅ Added {len(documents)} documents to vector store (total: {len(self.documents)})")
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise
    
    async def _persist_documents(self, documents, embeddings, metadatas):
        """Persist documents to SQLite database"""
        loop = asyncio.get_event_loop()
        
        def persist_sync():
            for doc, emb, meta in zip(documents, embeddings, metadatas):
                # Convert embedding to bytes for storage
                emb_bytes = emb.tobytes()
                meta_json = json.dumps(meta)
                
                self.conn.execute(
                    "INSERT INTO documents (content, embedding, metadata) VALUES (?, ?, ?)",
                    (doc, emb_bytes, meta_json)
                )
            self.conn.commit()
        
        await loop.run_in_executor(None, persist_sync)
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        distance_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        
        if not self.embeddings:
            return []
        
        try:
            # Convert query to numpy array
            query_np = np.array(query_embedding, dtype=np.float32)
            
            # Compute cosine similarities
            similarities = []
            for i, doc_emb in enumerate(self.embeddings):
                # Cosine similarity
                dot_product = np.dot(query_np, doc_emb)
                norms = np.linalg.norm(query_np) * np.linalg.norm(doc_emb)
                
                if norms > 0:
                    similarity = dot_product / norms
                    distance = 1 - similarity  # Convert to distance
                    
                    if distance < distance_threshold:
                        similarities.append({
                            'index': i,
                            'distance': distance,
                            'similarity': similarity
                        })
            
            # Sort by similarity (descending) and take top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            similarities = similarities[:top_k]
            
            # Format results
            formatted_results = []
            for sim in similarities:
                idx = sim['index']
                formatted_results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'distance': sim['distance'],
                    'similarity': sim['similarity']
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching vector store: {str(e)}")
            return []
    
    async def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.documents.clear()
            self.embeddings.clear()
            self.metadatas.clear()
            self.dimension = None
            
            # Clear database
            self.conn.execute("DELETE FROM documents")
            self.conn.commit()
            
            print("✅ Vector store cleared")
            
        except Exception as e:
            print(f"❌ Error clearing vector store: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            return {
                "document_count": len(self.documents),
                "dimension": self.dimension,
                "status": "ready"
            }
        except Exception as e:
            return {
                "document_count": 0,
                "dimension": None,
                "status": f"error: {str(e)}"
            }
    
    def __del__(self):
        """Clean up database connection"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except:
            pass

# Global vector store instance
vector_store = SimpleVectorStore()

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
