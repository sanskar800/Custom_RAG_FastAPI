import PyPDF2
from typing import BinaryIO

def extract_text_from_pdf(file_content: BinaryIO) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_content: Binary file content
        
    Returns:
        Extracted text as string
    """
    try:
        reader = PyPDF2.PdfReader(file_content)
        text_parts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        full_text = "\n".join(text_parts)
        return full_text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_txt(file_content: BinaryIO) -> str:
    """
    Extract text from a TXT file.
    
    Args:
        file_content: Binary file content
        
    Returns:
        Extracted text as string
    """
    try:
        content = file_content.read()
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='ignore')
        else:
            text = content
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from TXT: {str(e)}")