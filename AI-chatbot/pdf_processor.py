import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple
import tempfile
import os

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_with_pages(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page numbers and metadata
        Returns list of dictionaries with text, page_num, and metadata
        """
        doc = fitz.open(pdf_path)
        extracted_content = []
        
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Clean and preprocess text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    extracted_content.append({
                        'text': text,
                        'page_num': page_num + 1,  # 1-indexed page numbers
                        'char_count': len(text),
                        'word_count': len(text.split())
                    })
        finally:
            doc.close()
        
        return extracted_content
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-$$$$]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def chunk_content(self, extracted_content: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Split content into chunks while preserving page information
        """
        chunks = []
        
        for content in extracted_content:
            text = content['text']
            page_num = content['page_num']
            
            # If text is shorter than chunk_size, keep as single chunk
            if len(text) <= self.chunk_size:
                chunks.append({
                    'text': text,
                    'page_num': page_num,
                    'chunk_id': len(chunks),
                    'char_count': len(text),
                    'word_count': len(text.split())
                })
            else:
                # Split into overlapping chunks
                start = 0
                chunk_id = 0
                
                while start < len(text):
                    end = start + self.chunk_size
                    
                    # Try to break at word boundary
                    if end < len(text):
                        # Find last space within chunk
                        last_space = text.rfind(' ', start, end)
                        if last_space > start:
                            end = last_space
                    
                    chunk_text = text[start:end].strip()
                    
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'page_num': page_num,
                            'chunk_id': len(chunks),
                            'char_count': len(chunk_text),
                            'word_count': len(chunk_text.split()),
                            'start_pos': start,
                            'end_pos': end
                        })
                    
                    # Move start position with overlap
                    start = end - self.chunk_overlap
                    if start >= len(text):
                        break
        
        return chunks
    
    def process_uploaded_files(self, uploaded_files) -> Tuple[List[Dict[str, any]], Dict[str, str]]:
        """
        Process multiple uploaded PDF files
        Returns chunks and file mapping
        """
        all_chunks = []
        file_mapping = {}
        
        for uploaded_file in uploaded_files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Extract content with page numbers
                extracted_content = self.extract_text_with_pages(tmp_path)
                
                # Chunk the content
                file_chunks = self.chunk_content(extracted_content)
                
                # Add file information to each chunk
                for chunk in file_chunks:
                    chunk['filename'] = uploaded_file.name
                    chunk['file_id'] = len(file_mapping)
                
                all_chunks.extend(file_chunks)
                file_mapping[uploaded_file.name] = {
                    'file_id': len(file_mapping),
                    'total_pages': len(extracted_content),
                    'total_chunks': len(file_chunks)
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except (OSError, PermissionError):
                    # Handle Windows file locking issues
                    pass
        
        return all_chunks, file_mapping
    
    def get_chunk_reference(self, chunk: Dict[str, any]) -> str:
        """
        Generate a readable reference for a chunk
        """
        filename = chunk.get('filename', 'Unknown')
        page_num = chunk.get('page_num', 'Unknown')
        return f"{filename} (Page {page_num})"
