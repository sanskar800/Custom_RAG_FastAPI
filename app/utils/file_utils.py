import os
import uuid
from pathlib import Path
from typing import BinaryIO
from fastapi import UploadFile, HTTPException
from app.config import settings

def validate_file_extension(filename: str) -> bool:
    """
    Validate if file has an allowed extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if valid, False otherwise
    """
    ext = Path(filename).suffix.lower()
    return ext in settings.ALLOWED_EXTENSIONS

def validate_file_size(file: UploadFile) -> bool:
    """
    Validate file size.
    
    Args:
        file: Uploaded file
        
    Returns:
        True if valid size
    """
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    return file_size <= settings.MAX_FILE_SIZE

def save_upload_file(file: UploadFile) -> tuple[str, str]:
    """
    Save uploaded file to disk.
    
    Args:
        file: Uploaded file
        
    Returns:
        Tuple of (file_path, unique_filename)
    """
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    if not validate_file_size(file):
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
        file.file.seek(0)
        return file_path, unique_filename
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def cleanup_file(file_path: str):
    """
    Remove file from disk.
    
    Args:
        file_path: Path to file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass