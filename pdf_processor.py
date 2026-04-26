"""
pdf_processor.py
Handles PDF loading, text extraction, and chunking.
"""
import pdfplumber
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> str:
    """
    Load and extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(f"Successfully extracted text from PDF: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: Size of each chunk
        overlap: Overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def process_pdf(file_path: str) -> List[str]:
    """
    Complete pipeline: load PDF and split into chunks.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of text chunks
    """
    text = load_pdf(file_path)
    chunks = chunk_text(text)
    return chunks
