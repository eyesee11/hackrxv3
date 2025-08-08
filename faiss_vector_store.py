"""
FAISS-based vector store for high-performance similarity search
Much faster than ChromaDB and easier to install
"""
import json
import pickle
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """High-performance vector store using FAISS"""
    
    def __init__(self, dimension: int = 768, index_type: str = "IVF", storage_path: str = "faiss_index"):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Vector dimension (768 for text-embedding-004)
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')
            storage_path: Path to store index and metadata
        """
        if not faiss:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize index
        self.index = None
        self.metadata = []  # Store document metadata
        self.id_mapping = {}  # Map internal IDs to document IDs
        self.next_id = 0
        
        # Performance settings
        self.batch_size = 100
        self.search_threads = 2
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"✅ FAISS Vector Store initialized with {index_type} index, dimension={dimension}")
    
    def _create_index(self) -> faiss.Index:
        """Create appropriate FAISS index based on type"""
        
        if self.index_type == "Flat":
            # Exact search - slower but most accurate
            index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        elif self.index_type == "IVF":
            # Inverted file index - good balance of speed/accuracy
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(100, max(1, len(self.metadata) // 50))  # Adaptive nlist
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.nprobe = min(10, nlist)  # Search 10 clusters
        
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World - fastest search
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 connections per node
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 32
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        return index
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents with embeddings to the vector store"""
        
        if not documents:
            return []
        
        # Extract embeddings and metadata
        embeddings = []
        doc_ids = []
        
        for doc in documents:
            if 'embedding' not in doc:
                raise ValueError("Document must contain 'embedding' field")
            
            embeddings.append(doc['embedding'])
            
            # Generate ID if not provided
            doc_id = doc.get('id', f"doc_{self.next_id}")
            doc_ids.append(doc_id)
            self.next_id += 1
            
            # Store metadata
            metadata = {
                'id': doc_id,
                'content': doc.get('content', ''),
                'metadata': doc.get('metadata', {}),
                'timestamp': doc.get('timestamp')
            }
            self.metadata.append(metadata)
            self.id_mapping[len(self.metadata) - 1] = doc_id
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_array = embeddings_array / norms
        
        # Create or update index
        if self.index is None:
            self.index = self._create_index()
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(embeddings_array) >= 100:  # Need enough data to train
                logger.info("🔄 Training FAISS index...")
                self.index.train(embeddings_array)
            else:
                # Fall back to Flat index for small datasets
                logger.info("📊 Dataset too small for IVF, using Flat index")
                self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors to index
        self.index.add(embeddings_array)
        
        # Save index and metadata
        await self._save_index()
        
        logger.info(f"✅ Added {len(documents)} documents to FAISS index")
        return doc_ids
    
    async def search(self, query_embedding: List[float], k: int = 5, 
                    filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_array = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_array)
        if norm > 0:
            query_array = query_array / norm
        
        # Perform search
        try:
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
        
        # Process results
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            if idx >= len(self.metadata):
                logger.warning(f"Index {idx} out of range for metadata")
                continue
            
            metadata = self.metadata[idx]
            
            # Apply metadata filtering if specified
            if filter_metadata:
                if not self._matches_filter(metadata.get('metadata', {}), filter_metadata):
                    continue
            
            result = {
                'id': metadata['id'],
                'content': metadata['content'],
                'metadata': metadata['metadata'],
                'similarity': float(score),  # Cosine similarity
                'timestamp': metadata.get('timestamp')
            }
            
            results.append(result)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"🔍 Found {len(results)} similar documents")
        return results
    
    def _matches_filter(self, doc_metadata: Dict, filter_metadata: Dict) -> bool:
        """Check if document metadata matches filter"""
        
        for key, value in filter_metadata.items():
            if key not in doc_metadata:
                return False
            
            if isinstance(value, list):
                if doc_metadata[key] not in value:
                    return False
            else:
                if doc_metadata[key] != value:
                    return False
        
        return True
    
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by ID (requires rebuilding index)"""
        
        if not doc_ids:
            return True
        
        # Find indices to remove
        indices_to_remove = []
        
        for i, metadata in enumerate(self.metadata):
            if metadata['id'] in doc_ids:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return True
        
        # Remove metadata
        for idx in sorted(indices_to_remove, reverse=True):
            del self.metadata[idx]
        
        # Rebuild index (FAISS doesn't support efficient deletion)
        await self._rebuild_index()
        
        logger.info(f"🗑️ Deleted {len(indices_to_remove)} documents")
        return True
    
    async def _rebuild_index(self):
        """Rebuild index from scratch (needed for deletions)"""
        
        if not self.metadata:
            self.index = self._create_index()
            return
        
        logger.info("🔄 Rebuilding FAISS index...")
        
        # Re-create index
        self.index = self._create_index()
        
        # Note: This would require re-embedding all documents
        # For now, we'll just create an empty index
        # In a full implementation, you'd store embeddings separately
        
        await self._save_index()
    
    async def _save_index(self):
        """Save index and metadata to disk"""
        
        if self.index is None:
            return
        
        try:
            # Save FAISS index
            index_path = self.storage_path / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.storage_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'id_mapping': self.id_mapping,
                    'next_id': self.next_id,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }, f, indent=2)
            
            logger.debug("💾 Saved FAISS index and metadata")
        
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _load_index(self):
        """Load existing index and metadata from disk"""
        
        try:
            index_path = self.storage_path / "faiss.index"
            metadata_path = self.storage_path / "metadata.json"
            
            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                
                self.metadata = data.get('metadata', [])
                self.id_mapping = data.get('id_mapping', {})
                self.next_id = data.get('next_id', 0)
                
                # Convert string keys back to int for id_mapping
                self.id_mapping = {int(k): v for k, v in self.id_mapping.items()}
                
                logger.info(f"📂 Loaded FAISS index with {len(self.metadata)} documents")
            
            else:
                logger.info("🆕 No existing FAISS index found, will create new one")
        
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.index = None
            self.metadata = []
            self.id_mapping = {}
            self.next_id = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        
        stats = {
            'total_documents': len(self.metadata),
            'index_type': self.index_type,
            'dimension': self.dimension,
            'index_size': self.index.ntotal if self.index else 0,
            'is_trained': getattr(self.index, 'is_trained', True) if self.index else False
        }
        
        return stats
    
    async def clear(self):
        """Clear all documents from the vector store"""
        
        self.index = self._create_index()
        self.metadata = []
        self.id_mapping = {}
        self.next_id = 0
        
        await self._save_index()
        logger.info("🧹 Cleared FAISS vector store")

# Global vector store instance
vector_store: Optional[FAISSVectorStore] = None

def get_vector_store() -> FAISSVectorStore:
    """Get or create global vector store instance"""
    global vector_store
    
    if vector_store is None:
        vector_store = FAISSVectorStore(
            dimension=768,  # text-embedding-004 dimension
            index_type="IVF",  # Good balance of speed/accuracy
            storage_path="faiss_index"
        )
    
    return vector_store

# Convenience functions
async def add_documents(documents: List[Dict[str, Any]]) -> List[str]:
    """Add documents to vector store"""
    store = get_vector_store()
    return await store.add_documents(documents)

async def search_documents(query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar documents"""
    store = get_vector_store()
    return await store.search(query_embedding, k)

async def delete_documents(doc_ids: List[str]) -> bool:
    """Delete documents by ID"""
    store = get_vector_store()
    return await store.delete_documents(doc_ids)

def get_store_stats() -> Dict[str, Any]:
    """Get vector store statistics"""
    store = get_vector_store()
    return store.get_stats()
