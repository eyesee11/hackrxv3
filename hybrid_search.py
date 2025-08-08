"""
Hybrid Search Implementation: Vector Similarity + Keyword Search (BM25)
High Impact, Low Effort Optimization
"""

import re
import math
import asyncio
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np

class BM25:
    """Optimized BM25 implementation for keyword search"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.avgdl = 0.0
        
    def fit(self, corpus: List[str]):
        """Fit BM25 on document corpus"""
        self.corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        
        # Calculate document frequencies
        self.doc_freqs = []
        for doc in self.corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
        
        # Calculate IDF values
        all_terms = set()
        for doc in self.corpus:
            all_terms.update(doc)
        
        for term in all_terms:
            df = sum(1 for doc in self.corpus if term in doc)
            self.idf[term] = math.log((len(self.corpus) - df + 0.5) / (df + 0.5))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple but effective tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Remove very short tokens
        return [token for token in tokens if len(token) > 2]
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query against all documents"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc_freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lens[i]
            
            for token in query_tokens:
                if token in doc_freqs:
                    tf = doc_freqs[token]
                    idf = self.idf.get(token, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores

class HybridSearch:
    """Hybrid search combining vector similarity and BM25 keyword search"""
    
    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.bm25 = BM25()
        self.documents = []
        self.is_fitted = False
        
        # Query preprocessing patterns
        self.expansion_terms = {
            'cover': ['coverage', 'covered', 'includes', 'protection', 'benefit'],
            'exclude': ['exclusion', 'excluded', 'not covered', 'limitation', 'restriction'],
            'cost': ['premium', 'price', 'fee', 'charge', 'amount', 'payment'],
            'claim': ['claims', 'file claim', 'claim process', 'reimbursement'],
            'deductible': ['deductible', 'out of pocket', 'copay', 'coinsurance'],
            'limit': ['maximum', 'limit', 'cap', 'ceiling', 'up to']
        }
    
    def fit(self, documents: List[Dict[str, Any]]):
        """Fit hybrid search on documents"""
        self.documents = documents
        
        # Extract text content for BM25
        texts = [doc['content'] for doc in documents]
        self.bm25.fit(texts)
        self.is_fitted = True
        
        print(f"✅ Hybrid search fitted on {len(documents)} documents")
    
    def expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        expanded_terms = []
        query_lower = query.lower()
        
        for base_term, expansions in self.expansion_terms.items():
            if base_term in query_lower:
                expanded_terms.extend(expansions)
        
        # Add original query terms
        expanded_terms.append(query)
        
        return ' '.join(expanded_terms)
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        # Expand abbreviations
        abbreviations = {
            'deduct': 'deductible',
            'max': 'maximum',
            'min': 'minimum',
            'prem': 'premium',
            'copay': 'copayment',
            'coinsurance': 'co-insurance'
        }
        
        query_lower = query.lower()
        for abbrev, full_term in abbreviations.items():
            query_lower = query_lower.replace(abbrev, full_term)
        
        return query_lower
    
    async def search(self, query: str, vector_scores: List[Tuple[int, float]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword scores"""
        
        if not self.is_fitted:
            print("⚠️ Hybrid search not fitted, using vector scores only")
            # Return top results based on vector scores only
            top_indices = sorted(vector_scores, key=lambda x: x[1], reverse=True)[:top_k]
            return [{'index': idx, 'content': self.documents[idx]['content'], 'metadata': self.documents[idx]['metadata'], 'similarity': score} 
                   for idx, score in top_indices if idx < len(self.documents)]
        
        # Preprocess and expand query
        processed_query = self.preprocess_query(query)
        expanded_query = self.expand_query(processed_query)
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(expanded_query)
        
        # Normalize scores to 0-1 range
        vector_scores_dict = {idx: score for idx, score in vector_scores}
        
        # Normalize vector scores
        if vector_scores_dict:
            max_vector_score = max(vector_scores_dict.values())
            min_vector_score = min(vector_scores_dict.values())
            vector_range = max_vector_score - min_vector_score
            
            if vector_range > 0:
                normalized_vector = {idx: (score - min_vector_score) / vector_range 
                                   for idx, score in vector_scores_dict.items()}
            else:
                normalized_vector = {idx: 1.0 for idx in vector_scores_dict.keys()}
        else:
            normalized_vector = {}
        
        # Normalize BM25 scores
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            bm25_range = max_bm25 - min_bm25
            
            if bm25_range > 0:
                normalized_bm25 = [(score - min_bm25) / bm25_range for score in bm25_scores]
            else:
                normalized_bm25 = [1.0] * len(bm25_scores)
        else:
            normalized_bm25 = [0.0] * len(self.documents)
        
        # Combine scores
        combined_results = []
        
        for idx in range(len(self.documents)):
            vector_score = normalized_vector.get(idx, 0.0)
            keyword_score = normalized_bm25[idx] if idx < len(normalized_bm25) else 0.0
            
            # Weighted combination
            combined_score = (self.vector_weight * vector_score + 
                            self.keyword_weight * keyword_score)
            
            # Boost score for exact matches
            if self._has_exact_match(query, self.documents[idx]['content']):
                combined_score *= 1.2
            
            # Boost score for metadata matches
            metadata_boost = self._get_metadata_boost(query, self.documents[idx]['metadata'])
            combined_score *= metadata_boost
            
            combined_results.append({
                'index': idx,
                'content': self.documents[idx]['content'],
                'metadata': self.documents[idx]['metadata'],
                'similarity': combined_score,
                'vector_score': vector_score,
                'keyword_score': keyword_score
            })
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x['similarity'], reverse=True)
        return combined_results[:top_k]
    
    def _has_exact_match(self, query: str, text: str) -> bool:
        """Check for exact phrase matches"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Check for exact phrase match
        if query_lower in text_lower:
            return True
        
        # Check for exact word matches
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        # If more than 70% of query words are in text
        if query_words and len(query_words & text_words) / len(query_words) > 0.7:
            return True
        
        return False
    
    def _get_metadata_boost(self, query: str, metadata: Dict[str, Any]) -> float:
        """Get boost factor based on metadata matches"""
        boost = 1.0
        query_lower = query.lower()
        
        # Boost based on section type
        section_type = metadata.get('section_type', 'general')
        if section_type == 'coverage' and any(word in query_lower for word in ['cover', 'benefit', 'include']):
            boost *= 1.15
        elif section_type == 'exclusions' and any(word in query_lower for word in ['exclude', 'not cover', 'exception']):
            boost *= 1.15
        elif section_type == 'claims' and any(word in query_lower for word in ['claim', 'file', 'process']):
            boost *= 1.15
        elif section_type == 'payment' and any(word in query_lower for word in ['cost', 'premium', 'payment']):
            boost *= 1.15
        
        # Boost for relevant metadata flags
        if metadata.get('has_amounts', False) and any(word in query_lower for word in ['amount', 'cost', 'premium', 'deductible']):
            boost *= 1.1
        
        if metadata.get('has_dates', False) and any(word in query_lower for word in ['when', 'date', 'time', 'period']):
            boost *= 1.1
        
        if metadata.get('has_policy_terms', False):
            boost *= 1.05
        
        return boost

# Global hybrid search instance
hybrid_search = HybridSearch()

async def perform_hybrid_search(query: str, vector_results: List[Tuple[int, float]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Perform hybrid search with query and vector results"""
    return await hybrid_search.search(query, vector_results, top_k)

def fit_hybrid_search(documents: List[Dict[str, Any]]):
    """Fit hybrid search on documents"""
    hybrid_search.fit(documents)
