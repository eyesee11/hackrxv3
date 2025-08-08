import re
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class SimpleDocumentProcessor:
    """Enhanced document processing with overlapping chunks and metadata"""
    
    def __init__(self):
        # Improved chunk settings for better accuracy
        self.chunk_size = 500  # characters - slightly larger for better context
        self.chunk_overlap = 150  # 30% overlap for better continuity
        self.min_chunk_size = 100  # larger minimum for meaningful chunks
        
        # Hierarchical separators for better chunking
        self.separators = [
            "\n\n\n",  # Major section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence endings with space
            "! ",      # Exclamation with space
            "? ",      # Question with space
            "; ",      # Semicolon with space
            ", ",      # Comma with space
            " ",       # Word boundaries
            ""         # Character level (fallback)
        ]
    
    def count_tokens_approx(self, text: str) -> int:
        """Approximate token count (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better chunking"""
        
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)  # Clean up paragraph breaks
        
        # Remove page breaks and form feeds
        text = re.sub(r'[\f\r]', '\n', text)
        
        # Normalize multiple newlines (but keep double newlines for paragraphs)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Clean up common document artifacts
        text = re.sub(r'^\s*page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\s*-+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove lines with only special characters but preserve structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            # Keep the line if it has meaningful content or is a paragraph break
            if (len(stripped_line) > 2 and not re.match(r'^[^\w\s]*$', stripped_line)) or stripped_line == '':
                cleaned_lines.append(line)  # Keep original spacing
        
        return '\n'.join(cleaned_lines)
    
    def split_text_simple(self, text: str) -> List[str]:
        """Enhanced text splitting with better overlap and boundary detection"""
        
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        chunks = []
        
        # Try each separator in hierarchical order
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                
                current_chunk = ""
                for i, split in enumerate(splits):
                    # Add separator back except for the last split
                    if separator and i < len(splits) - 1:
                        split = split + separator
                    
                    # Check if adding this split would exceed chunk size
                    test_chunk = current_chunk + split if current_chunk else split
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if it's large enough
                        if len(current_chunk.strip()) >= self.min_chunk_size:
                            chunks.append(current_chunk.strip())
                        
                        # Start new chunk with smart overlap
                        if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                            # Try to create overlap at word/sentence boundaries
                            overlap = self._create_smart_overlap(current_chunk, self.chunk_overlap)
                            current_chunk = overlap + split
                        else:
                            current_chunk = split
                
                # Add final chunk
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                # If we got good chunks, return them
                if chunks and len(chunks) > 1:
                    return chunks
        
        # Fallback: split by character count with overlap
        return self._split_by_chars_with_overlap(text)
    
    def _create_smart_overlap(self, text: str, overlap_size: int) -> str:
        """Create overlap at natural boundaries (sentences, words)"""
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundaries first
        sentences = re.split(r'[.!?]+\s+', text)
        if len(sentences) > 1:
            overlap = ""
            for sentence in reversed(sentences[:-1]):  # Exclude the last incomplete sentence
                test_overlap = sentence + ". " + overlap
                if len(test_overlap) <= overlap_size:
                    overlap = test_overlap
                else:
                    break
            if overlap:
                return overlap
        
        # Fallback to word boundaries
        words = text.split()
        overlap_words = []
        overlap_length = 0
        
        for word in reversed(words):
            test_length = overlap_length + len(word) + 1  # +1 for space
            if test_length <= overlap_size:
                overlap_words.insert(0, word)
                overlap_length = test_length
            else:
                break
        
        return " ".join(overlap_words) + " " if overlap_words else text[-overlap_size:]
    
    def _split_by_chars_with_overlap(self, text: str) -> List[str]:
        """Split text by character count with smart overlap"""
        chunks = []
        
        if len(text) < self.min_chunk_size:
            return []
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if len(chunk.strip()) >= self.min_chunk_size:
                chunks.append(chunk.strip())
            
            # Calculate next start position with overlap
            if end >= len(text):
                break
            
            # Try to break at word boundary for better overlap
            overlap_start = max(start, end - self.chunk_overlap)
            next_space = text.find(' ', overlap_start)
            
            if next_space != -1 and next_space < end:
                start = next_space + 1
            else:
                start = end - self.chunk_overlap
            
            # Safety check
            if len(chunks) > 1000:
                print(f"⚠️ Too many chunks generated, stopping at {len(chunks)}")
                break
        
        return chunks
    
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
        
        try:
            # Preprocess the content
            cleaned_content = self.preprocess_text(content)
            
            if len(cleaned_content.strip()) < self.min_chunk_size:
                return []
            
            # Split into chunks with error handling
            try:
                chunks = self.split_text_simple(cleaned_content)
            except Exception as e:
                print(f"⚠️ Error splitting document {doc_id}: {e}, using simple character splitting")
                chunks = self._split_by_chars(cleaned_content)
            
            processed_chunks = []
            
            for chunk_id, chunk in enumerate(chunks):
                chunk = chunk.strip()
                
                # Skip tiny chunks
                if len(chunk) < self.min_chunk_size:
                    continue
                
                # Extract key information for better search
                try:
                    chunk_info = self._extract_chunk_info(chunk)
                except Exception as e:
                    print(f"⚠️ Error extracting chunk info: {e}")
                    chunk_info = {
                        'has_numbers': False,
                        'has_percentages': False,
                        'has_dates': False,
                        'sentence_count': 1
                    }
                
                processed_chunks.append({
                    'content': chunk,
                    'metadata': {
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'token_count': self.count_tokens_approx(chunk),
                        'char_count': len(chunk),
                        'has_numbers': chunk_info['has_numbers'],
                        'has_percentages': chunk_info['has_percentages'],
                        'has_dates': chunk_info['has_dates'],
                        'sentence_count': chunk_info['sentence_count'],
                        # Enhanced metadata for better retrieval
                        'has_policy_terms': chunk_info.get('has_policy_terms', False),
                        'has_amounts': chunk_info.get('has_amounts', False),
                        'section_type': chunk_info.get('section_type', 'general'),
                        'chunk_position': chunk_id / len(chunks) if len(chunks) > 0 else 0,  # relative position
                    }
                })
                
                # Safety limit to prevent excessive memory usage
                if len(processed_chunks) > 500:
                    print(f"⚠️ Document {doc_id} generated too many chunks, stopping at {len(processed_chunks)}")
                    break
            
            return processed_chunks
            
        except Exception as e:
            print(f"❌ Error processing document {doc_id}: {e}")
            return []
        
        return processed_chunks
    
    def _extract_chunk_info(self, chunk: str) -> Dict[str, Any]:
        """Extract enhanced information from chunk for metadata"""
        
        # Check for numbers and percentages
        has_numbers = bool(re.search(r'\d+', chunk))
        has_percentages = bool(re.search(r'\d+\s*%', chunk))
        
        # Check for monetary amounts
        has_amounts = bool(re.search(r'[\$₹€£]\s*\d+|amount|premium|deductible|limit', chunk, re.IGNORECASE))
        
        # Check for dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
            r'\b\d{1,2}\s+(months?|years?|days?)\b',  # "2 months", "1 year"
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b'
        ]
        has_dates = any(re.search(pattern, chunk, re.IGNORECASE) for pattern in date_patterns)
        
        # Check for policy-specific terms
        policy_terms = [
            'coverage', 'premium', 'deductible', 'exclusion', 'benefit', 'claim',
            'policy', 'insured', 'insurer', 'liability', 'limit', 'condition',
            'renewal', 'cancellation', 'endorsement', 'rider'
        ]
        has_policy_terms = any(term in chunk.lower() for term in policy_terms)
        
        # Determine section type based on content
        section_type = 'general'
        if any(term in chunk.lower() for term in ['coverage', 'benefit', 'covered']):
            section_type = 'coverage'
        elif any(term in chunk.lower() for term in ['exclusion', 'not covered', 'except']):
            section_type = 'exclusions'
        elif any(term in chunk.lower() for term in ['premium', 'payment', 'due']):
            section_type = 'payment'
        elif any(term in chunk.lower() for term in ['claim', 'procedure', 'process']):
            section_type = 'claims'
        
        # Count sentences (improved)
        sentence_count = len(re.findall(r'[.!?]+\s+[A-Z]', chunk)) + 1
        
        return {
            'has_numbers': has_numbers,
            'has_percentages': has_percentages,
            'has_dates': has_dates,
            'has_policy_terms': has_policy_terms,
            'has_amounts': has_amounts,
            'section_type': section_type,
            'sentence_count': max(1, sentence_count)
        }
    
    def prepare_context(self, search_results: List[Dict], max_chars: int = 2000) -> str:
        """Prepare context from search results with character limit"""
        
        if not search_results:
            return "No relevant information found."
        
        context_parts = []
        total_chars = 0
        
        # Sort by similarity (highest first)
        sorted_results = sorted(search_results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for result in sorted_results:
            content = result['content']
            content_chars = len(content)
            
            # Check if adding this content would exceed character limit
            if total_chars + content_chars > max_chars:
                # Try to fit partial content
                remaining_chars = max_chars - total_chars
                if remaining_chars > 100:  # Only if we have reasonable space left
                    # Truncate content to fit
                    truncated_content = content[:remaining_chars] + "..."
                    context_parts.append(truncated_content)
                break
            
            context_parts.append(content)
            total_chars += content_chars
        
        return "\n\n".join(context_parts)

# Global document processor instance
document_processor = SimpleDocumentProcessor()

# Convenience functions
async def process_documents(documents: List[str]) -> List[Dict]:
    """Process documents into chunks"""
    return await document_processor.process_documents(documents)

def prepare_context(search_results: List[Dict], max_chars: int = 2000) -> str:
    """Prepare context from search results"""
    return document_processor.prepare_context(search_results, max_chars)
