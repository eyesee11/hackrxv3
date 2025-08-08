from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from typing import List, Dict, Any
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedDocumentProcessor:
    """Optimized document processing with intelligent chunking"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Optimized splitter configuration
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Optimal for Google's embedding model
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
            length_function=self.count_tokens,
            keep_separator=False
        )
        
        # Minimum chunk size to avoid tiny meaningless chunks
        self.min_chunk_size = 50
        
        # Maximum tokens per chunk for context window
        self.max_tokens_per_chunk = 500
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to character-based estimation
            return len(text) // 4
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better chunking"""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page breaks and form feeds
        text = re.sub(r'[\f\r]', '\n', text)
        
        # Normalize newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove lines with only special characters
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 3 and not re.match(r'^[^\w]*$', line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    async def process_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Process documents into optimized chunks"""
        
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
        
        if len(cleaned_content.strip()) < self.min_chunk_size:
            return []
        
        # Split into chunks
        chunks = self.splitter.split_text(cleaned_content)
        
        processed_chunks = []
        
        for chunk_id, chunk in enumerate(chunks):
            chunk = chunk.strip()
            
            # Skip tiny chunks
            if len(chunk) < self.min_chunk_size:
                continue
            
            # Check token count
            token_count = self.count_tokens(chunk)
            
            # Skip chunks that are too large (shouldn't happen with our settings)
            if token_count > self.max_tokens_per_chunk:
                continue
            
            # Extract key information for better search
            chunk_info = self._extract_chunk_info(chunk)
            
            processed_chunks.append({
                'content': chunk,
                'metadata': {
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'token_count': token_count,
                    'char_count': len(chunk),
                    'has_numbers': chunk_info['has_numbers'],
                    'has_percentages': chunk_info['has_percentages'],
                    'has_dates': chunk_info['has_dates'],
                    'sentence_count': chunk_info['sentence_count']
                }
            })
        
        return processed_chunks
    
    def _extract_chunk_info(self, chunk: str) -> Dict[str, Any]:
        """Extract useful information from chunk for metadata"""
        
        # Check for numbers and percentages
        has_numbers = bool(re.search(r'\d+', chunk))
        has_percentages = bool(re.search(r'\d+\s*%', chunk))
        
        # Check for dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
            r'\b\d{1,2}\s+(months?|years?|days?)\b',  # "2 months", "1 year"
        ]
        has_dates = any(re.search(pattern, chunk, re.IGNORECASE) for pattern in date_patterns)
        
        # Count sentences (approximate)
        sentence_count = len(re.findall(r'[.!?]+', chunk))
        
        return {
            'has_numbers': has_numbers,
            'has_percentages': has_percentages,
            'has_dates': has_dates,
            'sentence_count': max(1, sentence_count)
        }
    
    def prepare_context(self, search_results: List[Dict], max_tokens: int = 2000) -> str:
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
                # Try to fit partial content
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only if we have reasonable space left
                    # Truncate content to fit
                    words = content.split()
                    truncated_words = []
                    temp_tokens = 0
                    
                    for word in words:
                        word_tokens = self.count_tokens(word + " ")
                        if temp_tokens + word_tokens > remaining_tokens:
                            break
                        truncated_words.append(word)
                        temp_tokens += word_tokens
                    
                    if truncated_words:
                        truncated_content = " ".join(truncated_words) + "..."
                        context_parts.append(truncated_content)
                break
            
            context_parts.append(content)
            total_tokens += content_tokens
        
        return "\n\n".join(context_parts)

# Global document processor instance
document_processor = OptimizedDocumentProcessor()

# Convenience functions
async def process_documents(documents: List[str]) -> List[Dict]:
    """Process documents into chunks"""
    return await document_processor.process_documents(documents)

def prepare_context(search_results: List[Dict], max_tokens: int = 2000) -> str:
    """Prepare context from search results"""
    return document_processor.prepare_context(search_results, max_tokens)
