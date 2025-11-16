import os
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional

from app.utils.file_utils import save_upload_file, cleanup_file
from app.utils.pdf_to_text import extract_text_from_pdf, extract_text_from_txt
from app.services.chunking import chunk_text
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store_service
from app.services.rag import rag_service

router = APIRouter()

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    message: str
    doc_id: str
    filename: str
    num_chunks: int
    chunking_method: str
    timestamp: str

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or TXT file to ingest"),
    chunking_method: str = Form("semantic", description="Chunking method: 'semantic' or 'fixed'")
):
    """
    Ingest a document into the RAG system.
    
    This endpoint:
    1. Accepts PDF or TXT file uploads
    2. Extracts text from the file
    3. Chunks the text using the specified method
    4. Generates embeddings for each chunk
    5. Stores vectors in Qdrant
    6. Saves metadata in MongoDB
    
    Args:
        file: Uploaded file (PDF or TXT)
        chunking_method: Method for chunking ('semantic' or 'fixed')
        
    Returns:
        Ingestion result with document metadata
    """
    file_path = None
    
    try:
        # Validate chunking method
        if chunking_method not in ["semantic", "fixed"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid chunking method. Must be 'semantic' or 'fixed'."
            )
        
        # Save uploaded file
        file_path, unique_filename = save_upload_file(file)
        
        # Extract text based on file type
        filename_lower = file.filename.lower()
        if filename_lower.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                text = extract_text_from_pdf(f)
        elif filename_lower.endswith('.txt'):
            with open(file_path, 'rb') as f:
                text = extract_text_from_txt(f)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only PDF and TXT files are supported."
            )
        
        # Validate extracted text
        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract sufficient text from the file. File may be empty or corrupted."
            )
        
        # Chunk the text
        chunks = chunk_text(text, method=chunking_method)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to create chunks from the text. File may be too short."
            )
        
        # Generate embeddings
        vectors = embedding_service.embed_texts(chunks)
        
        if len(chunks) != len(vectors):
            raise HTTPException(
                status_code=500,
                detail="Mismatch between number of chunks and vectors."
            )
        
        # Generate unique document ID
        doc_id = f"{file.filename}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        
        # Store vectors in Qdrant
        chunk_ids = vector_store_service.store_chunks(
            doc_id=doc_id,
            filename=file.filename,
            chunks=chunks,
            vectors=vectors
        )
        
        # Save metadata to MongoDB
        rag_service.save_document_metadata(
            doc_id=doc_id,
            filename=file.filename,
            num_chunks=len(chunks),
            chunk_ids=chunk_ids,
            chunking_method=chunking_method
        )
        
        return IngestResponse(
            success=True,
            message="Document ingested successfully",
            doc_id=doc_id,
            filename=file.filename,
            num_chunks=len(chunks),
            chunking_method=chunking_method,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )
    finally:
        # Clean up uploaded file
        if file_path:
            cleanup_file(file_path)