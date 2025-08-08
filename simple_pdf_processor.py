"""
Simple PDF extraction without PyMuPDF dependency
Uses only PyPDF2 and pdfplumber
"""
import io
import asyncio
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

class SimplePDFProcessor:
    """Simple PDF processor using available libraries"""
    
    def __init__(self):
        self.available_methods = []
        
        if PyPDF2:
            self.available_methods.append("pypdf2")
        if pdfplumber:
            self.available_methods.append("pdfplumber")
        
        if not self.available_methods:
            raise ImportError("No PDF processing libraries available. Install PyPDF2 or pdfplumber.")
        
        print(f"📄 PDF processor initialized with methods: {self.available_methods}")
    
    def extract_with_pypdf2(self, pdf_bytes: bytes) -> str:
        """Extract text using PyPDF2"""
        if not PyPDF2:
            return ""
        
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text.strip()
        
        except Exception as e:
            print(f"⚠️ PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_with_pdfplumber(self, pdf_bytes: bytes) -> str:
        """Extract text using pdfplumber"""
        if not pdfplumber:
            return ""
        
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                return text.strip()
        
        except Exception as e:
            print(f"⚠️ pdfplumber extraction failed: {e}")
            return ""
    
    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extract text using best available method"""
        
        # Try methods in order of preference
        methods = {
            "pdfplumber": self.extract_with_pdfplumber,
            "pypdf2": self.extract_with_pypdf2
        }
        
        for method_name in self.available_methods:
            if method_name in methods:
                text = methods[method_name](pdf_bytes)
                if text and len(text.strip()) > 10:
                    print(f"✅ Extracted {len(text)} characters using {method_name}")
                    return text
        
        print("❌ All PDF extraction methods failed")
        return ""
    
    async def process_pdf_files(self, pdf_files: List[bytes]) -> List[str]:
        """Process multiple PDF files asynchronously"""
        
        if not pdf_files:
            return []
        
        loop = asyncio.get_event_loop()
        
        # Use ThreadPoolExecutor for CPU-intensive PDF processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks = []
            
            for pdf_bytes in pdf_files:
                task = loop.run_in_executor(
                    executor,
                    self.extract_text,
                    pdf_bytes
                )
                tasks.append(task)
            
            # Wait for all PDFs to be processed
            texts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and empty texts
            valid_texts = []
            for i, text in enumerate(texts):
                if isinstance(text, Exception):
                    print(f"⚠️ PDF {i} processing failed: {text}")
                elif text and len(text.strip()) > 10:
                    valid_texts.append(text)
                else:
                    print(f"⚠️ PDF {i} extraction returned empty text")
            
            print(f"📄 Successfully processed {len(valid_texts)}/{len(pdf_files)} PDF files")
            return valid_texts
    
    def get_pdf_info(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Get basic information about a PDF"""
        
        info = {
            "size_bytes": len(pdf_bytes),
            "pages": 0,
            "extractable": False,
            "method_used": None
        }
        
        # Try to get page count with PyPDF2
        if PyPDF2:
            try:
                pdf_file = io.BytesIO(pdf_bytes)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                info["pages"] = len(pdf_reader.pages)
                
                # Test if text is extractable
                if info["pages"] > 0:
                    first_page_text = pdf_reader.pages[0].extract_text()
                    if first_page_text and len(first_page_text.strip()) > 10:
                        info["extractable"] = True
                        info["method_used"] = "pypdf2"
                
            except Exception as e:
                print(f"⚠️ Could not get PDF info: {e}")
        
        return info

# Global PDF processor instance
try:
    pdf_processor = SimplePDFProcessor()
except ImportError as e:
    print(f"⚠️ PDF processing not available: {e}")
    pdf_processor = None

# Convenience functions
async def process_pdf_files(pdf_files: List[bytes]) -> List[str]:
    """Process PDF files and extract text"""
    if not pdf_processor:
        print("❌ PDF processor not available")
        return []
    
    return await pdf_processor.process_pdf_files(pdf_files)

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a single PDF"""
    if not pdf_processor:
        print("❌ PDF processor not available")
        return ""
    
    return pdf_processor.extract_text(pdf_bytes)

def get_pdf_info(pdf_bytes: bytes) -> Dict[str, Any]:
    """Get information about a PDF"""
    if not pdf_processor:
        return {"error": "PDF processor not available"}
    
    return pdf_processor.get_pdf_info(pdf_bytes)
