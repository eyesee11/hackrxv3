"""
Enhanced Document Processor with tiktoken
Uses proper token counting and optimized chunking strategies
"""
import re
import asyncio
import tiktoken
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class EnhancedDocumentProcessor:
    """Advanced document processing with tiktoken-based chunking"""
    
    def __init__(self):
        # Initialize tiktoken encoder
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
        
        # Optimized chunk settings for Google's text-embedding-004
        self.chunk_size = 400  # tokens (optimal for embedding model)
        self.chunk_overlap = 50  # tokens
        self.min_chunk_size = 50  # tokens
        self.max_chunks_per_doc = 100  # Limit chunks per document
        
        # Simple separators for efficient chunking
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", " "]
    
    def count_tokens(self, text: str) -> int:
        """Accurate token count using tiktoken"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback to character-based estimation
        return len(text) // 4
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for better chunking"""
        
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace from lines
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)  # Windows to Unix
        text = re.sub(r'\r', '\n', text)    # Mac to Unix
        
        # Reduce excessive line breaks but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control characters
        text = re.sub(r'(?:\n|^)\s*\d+\s*(?:\n|$)', '\n', text)  # Standalone page numbers
        
        # Fix common tokenization issues
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentences
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)     # Fix missing spaces between words
        
        # Remove lines that are mostly special characters or very short
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip lines that are too short or mostly non-alphanumeric
            if len(line) < 10:
                continue
            
            # Skip lines that are mostly special characters
            alpha_ratio = sum(c.isalnum() or c.isspace() for c in line) / len(line)
            if alpha_ratio < 0.6:
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def split_text_simple(self, text: str) -> List[str]:
        """Simple, efficient text splitting without recursion"""
        
        token_count = self.count_tokens(text)
        
        # If text is already small enough, return it
        if token_count <= self.chunk_size:
            if token_count >= self.min_chunk_size:
                return [text]
            else:
                return []
        
        chunks = []
        
        # Try each separator to find the best one
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                
                current_chunk = ""
                current_tokens = 0
                
                for split in splits:
                    # Re-add separator if it's not empty
                    if separator and split:
                        split_text = split + separator
                    else:
                        split_text = split
                    
                    split_tokens = self.count_tokens(split_text)
                    
                    # Check if we can add this split to current chunk
                    if current_tokens + split_tokens <= self.chunk_size:
                        current_chunk += split_text
                        current_tokens += split_tokens
                    else:
                        # Save current chunk if it's large enough
                        if current_tokens >= self.min_chunk_size:
                            chunks.append(current_chunk.strip())
                        
                        # Start new chunk
                        current_chunk = split_text
                        current_tokens = split_tokens
                
                # Add final chunk
                if current_tokens >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # If we got good chunks, return them
                if chunks:
                    return [chunk for chunk in chunks if chunk.strip()]
        
        # Fallback: split by token count
        return self._split_by_tokens_simple(text)
    
    def _split_by_tokens_simple(self, text: str) -> List[str]:
        """Simple token-based splitting"""
        
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                chunks = []
                
                for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                    chunk_tokens = tokens[i:i + self.chunk_size]
                    
                    if len(chunk_tokens) >= self.min_chunk_size:
                        try:
                            chunk_text = self.tokenizer.decode(chunk_tokens)
                            chunks.append(chunk_text)
                        except Exception:
                            continue
                
                return chunks
            except Exception:
                pass
        
        # Character-based fallback
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        char_min_size = self.min_chunk_size * 4
        
        chunks = []
        for i in range(0, len(text), char_chunk_size - char_overlap):
            chunk = text[i:i + char_chunk_size]
            if len(chunk) >= char_min_size:
                chunks.append(chunk)
        
        return chunks
    
    async def process_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Process documents into optimized chunks"""
        
        if not documents:
            return []
        
        processed_chunks = []
        
        # Use thread executor for CPU-intensive processing
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks = []
            
            for doc_id, content in enumerate(documents):
                task = loop.run_in_executor(
                    executor,
                    self._process_single_document,
                    content, doc_id
                )
                tasks.append(task)
            
            # Wait for all documents to be processed
            document_chunks = await asyncio.gather(*tasks)
            
            # Flatten the results
            for chunks in document_chunks:
                processed_chunks.extend(chunks)
        
        print(f"✅ Processed {len(documents)} documents into {len(processed_chunks)} chunks")
        return processed_chunks
    
    def _process_single_document(self, content: str, doc_id: int) -> List[Dict[str, Any]]:
        """Process a single document into chunks"""
        
        # Preprocess the content
        cleaned_content = self.preprocess_text(content)
        
        if self.count_tokens(cleaned_content) < self.min_chunk_size:
            return []
        
        # Split into chunks with simple, safe method
        chunks = self.split_text_simple(cleaned_content)
        
        # Enforce absolute maximum to prevent recursion issues
        if len(chunks) > self.max_chunks_per_doc:
            chunks = chunks[:self.max_chunks_per_doc]
            print(f"⚠️ Limited document {doc_id} to {self.max_chunks_per_doc} chunks")
        
        processed_chunks = []
        
        for chunk_id, chunk in enumerate(chunks):
            chunk = chunk.strip()
            
            # Skip chunks that are too small
            token_count = self.count_tokens(chunk)
            if token_count < self.min_chunk_size:
                continue
            
            # Extract metadata for better search
            chunk_metadata = self._extract_chunk_metadata(chunk)
            
            processed_chunks.append({
                'content': chunk,
                'metadata': {
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'token_count': token_count,
                    'char_count': len(chunk),
                    **chunk_metadata
                }
            })
        
        return processed_chunks
    
    def _extract_chunk_metadata(self, chunk: str) -> Dict[str, Any]:
        """Extract useful metadata from chunk content"""
        
        # Detect content patterns
        has_numbers = bool(re.search(r'\d+', chunk))
        has_percentages = bool(re.search(r'\d+\s*%', chunk))
        has_dates = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', chunk))
        has_currency = bool(re.search(r'[$€£¥₹]\s*\d+|Rs\.?\s*\d+|\d+\s*dollars?', chunk, re.IGNORECASE))
        
        # Count sentences and paragraphs
        sentence_count = len(re.findall(r'[.!?]+', chunk))
        paragraph_count = len(re.split(r'\n\s*\n', chunk))
        
        # Detect question patterns
        has_questions = bool(re.search(r'\?', chunk))
        
        # Check for list patterns
        has_lists = bool(re.search(r'^\s*[-•*]\s+', chunk, re.MULTILINE))
        
        # Detect headings or titles (simple heuristic)
        lines = chunk.split('\n')
        potential_heading = any(
            len(line.strip()) < 100 and len(line.strip()) > 5 and not line.strip().endswith('.')
            for line in lines[:2]  # Check first two lines
        )
        
        return {
            'has_numbers': has_numbers,
            'has_percentages': has_percentages,
            'has_dates': has_dates,
            'has_currency': has_currency,
            'has_questions': has_questions,
            'has_lists': has_lists,
            'has_potential_heading': potential_heading,
            'sentence_count': max(1, sentence_count),
            'paragraph_count': max(1, paragraph_count),
            'language_complexity': self._estimate_complexity(chunk)
        }
    
    def _estimate_complexity(self, chunk: str) -> str:
        """Estimate the linguistic complexity of the chunk"""
        
        words = chunk.split()
        if not words:
            return "low"
        
        # Simple metrics for complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = long_words / len(words)
        
        if avg_word_length > 6 and long_word_ratio > 0.3:
            return "high"
        elif avg_word_length > 4.5 and long_word_ratio > 0.2:
            return "medium"
        else:
            return "low"
    
    def prepare_context(self, search_results: List[Dict], max_tokens: int = 1500) -> str:
        """Prepare context from search results with token limit"""
        
        if not search_results:
            return "No relevant information found."
        
        context_parts = []
        total_tokens = 0
        
        # Sort by similarity (highest first)
        sorted_results = sorted(search_results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for result in sorted_results:
            content = result['content']
            content_tokens = self.count_tokens(content)
            
            # Check if adding this content would exceed token limit
            if total_tokens + content_tokens > max_tokens:
                # Try to fit partial content if we have significant space left
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:
                    # Truncate content to fit
                    try:
                        # Encode, truncate, and decode
                        tokens = self.tokenizer.encode(content)
                        truncated_tokens = tokens[:remaining_tokens]
                        truncated_content = self.tokenizer.decode(truncated_tokens) + "..."
                        context_parts.append(truncated_content)
                    except Exception:
                        # Fallback to character truncation
                        char_limit = remaining_tokens * 4
                        truncated_content = content[:char_limit] + "..."
                        context_parts.append(truncated_content)
                break
            
            context_parts.append(content)
            total_tokens += content_tokens
        
        return "\n\n".join(context_parts)

# Global enhanced document processor instance
enhanced_processor = EnhancedDocumentProcessor()

# Convenience functions
async def process_documents_enhanced(documents: List[str]) -> List[Dict]:
    """Process documents using enhanced processor"""
    return await enhanced_processor.process_documents(documents)

def prepare_context_enhanced(search_results: List[Dict], max_tokens: int = 1500) -> str:
    """Prepare context using enhanced processor"""
    return enhanced_processor.prepare_context(search_results, max_tokens)

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    return enhanced_processor.count_tokens(text)
