import re
from typing import List
from app.config import settings

def chunk_text_fixed(
    text: str, 
    chunk_size: int = None, 
    overlap: int = None
) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if overlap is None:
        overlap = settings.CHUNK_OVERLAP
    
    chunks = []
    text_length = len(text)
    position = 0
    
    while position < text_length:
        end_position = min(position + chunk_size, text_length)
        chunk = text[position:end_position].strip()
        
        if chunk:
            chunks.append(chunk)
        
        position += chunk_size - overlap
        
        if position >= text_length:
            break
    
    return chunks

def chunk_text_semantic(
    text: str, 
    chunk_size: int = None, 
    overlap: int = None
) -> List[str]:
    """
    Split text into semantic chunks based on paragraphs and sentences.
    
    Args:
        text: Input text
        chunk_size: Target size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if overlap is None:
        overlap = settings.CHUNK_OVERLAP
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    
    # Split paragraphs into sentences
    sentences = []
    for paragraph in paragraphs:
        sentence_list = re.split(r'(?<=[\.\?\!])\s+', paragraph)
        for sentence in sentence_list:
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    
    if not sentences:
        return [text] if text.strip() else []
    
    # Build chunks from sentences
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        potential_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        
        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # Handle very long sentences
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - overlap):
                    part = sentence[i:i + (chunk_size - overlap)].strip()
                    if part:
                        chunks.append(part)
                current_chunk = ""
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Apply overlap if needed
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                previous_chunk = overlapped_chunks[-1]
                tail = previous_chunk[-overlap:] if len(previous_chunk) >= overlap else previous_chunk
                new_chunk = (tail + " " + chunk).strip()
                overlapped_chunks.append(new_chunk)
        chunks = overlapped_chunks
    
    return chunks

def chunk_text(text: str, method: str = "semantic") -> List[str]:
    """
    Split text into chunks using specified method.
    
    Args:
        text: Input text
        method: Chunking method ('semantic' or 'fixed')
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    if method == "semantic":
        return chunk_text_semantic(text)
    elif method == "fixed":
        return chunk_text_fixed(text)
    else:
        raise ValueError(f"Unknown chunking method: {method}")