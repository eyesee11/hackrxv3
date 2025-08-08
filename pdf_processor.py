import httpx
import PyPDF2
import pdfplumber
import io
import asyncio
from typing import Optional
import time

class PDFProcessor:
    """Optimized PDF processing with multiple extraction methods"""
    
    def __init__(self):
        self.timeout = 30.0
        self.max_retries = 3
    
    async def download_and_extract_pdf(self, pdf_url: str) -> str:
        """Download PDF from URL and extract text with fallback methods"""
        
        start_time = time.time()
        
        try:
            # Download PDF content
            pdf_content = await self._download_pdf_with_retry(pdf_url)
            
            # Try multiple extraction methods for best results
            text_content = await self._extract_text_with_fallbacks(pdf_content)
            
            if not text_content or len(text_content.strip()) < 100:
                raise ValueError("Extracted text is too short or empty")
            
            download_time = (time.time() - start_time) * 1000
            print(f"✅ PDF downloaded and extracted in {download_time:.2f}ms, {len(text_content)} characters")
            
            return text_content
            
        except Exception as e:
            print(f"❌ Error processing PDF from {pdf_url}: {str(e)}")
            raise
    
    async def _download_pdf_with_retry(self, pdf_url: str) -> bytes:
        """Download PDF with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(pdf_url)
                    response.raise_for_status()
                    
                    if len(response.content) < 1000:  # Suspiciously small PDF
                        raise ValueError("Downloaded PDF is too small")
                    
                    return response.content
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"❌ Download attempt {attempt + 1} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    
    async def _extract_text_with_fallbacks(self, pdf_content: bytes) -> str:
        """Try multiple PDF extraction methods for best results"""
        
        extraction_methods = [
            ("PyPDF2", self._extract_with_pypdf2),
            ("pdfplumber", self._extract_with_pdfplumber)
        ]
        
        for method_name, method_func in extraction_methods:
            try:
                text = await method_func(pdf_content)
                
                if text and len(text.strip()) > 100:
                    print(f"✅ Successfully extracted text using {method_name}")
                    return text
                else:
                    print(f"⚠️ {method_name} extracted insufficient text")
                    
            except Exception as e:
                print(f"❌ {method_name} extraction failed: {str(e)}")
                continue
        
        raise ValueError("All PDF extraction methods failed")
    
    async def _extract_with_pypdf2(self, pdf_content: bytes) -> str:
        """Extract text using PyPDF2 (fastest method)"""
        
        def extract_sync():
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"
                except Exception as e:
                    print(f"❌ Error extracting page {page_num}: {str(e)}")
                    continue
            
            return text_content
        
        # Run in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract_sync)
    
    async def _extract_with_pdfplumber(self, pdf_content: bytes) -> str:
        """Extract text using pdfplumber (more accurate)"""
        
        def extract_sync():
            pdf_file = io.BytesIO(pdf_content)
            text_content = ""
            
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n\n"
                    except Exception as e:
                        print(f"❌ Error extracting page {page_num} with pdfplumber: {str(e)}")
                        continue
            
            return text_content
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, extract_sync)

# Global PDF processor instance
pdf_processor = PDFProcessor()

# Convenience function
async def download_and_extract_pdf(pdf_url: str) -> str:
    """Download and extract text from PDF URL"""
    return await pdf_processor.download_and_extract_pdf(pdf_url)

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes - synchronous version for direct API use"""
    try:
        # First try PyPDF2 (faster)
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n\n"
            except Exception:
                continue
        
        # If PyPDF2 extracted sufficient text, return it
        if text_content and len(text_content.strip()) > 100:
            print(f"✅ PDF text extracted with PyPDF2: {len(text_content)} characters")
            return text_content
        
        # Fall back to pdfplumber
        pdf_file.seek(0)  # Reset file pointer
        text_content = ""
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n"
                except Exception:
                    continue
        
        print(f"✅ PDF text extracted with pdfplumber: {len(text_content)} characters")
        return text_content
        
    except Exception as e:
        print(f"❌ Error extracting PDF text: {str(e)}")
        return ""
