import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: str
    QDRANT_API_KEY: str
    
    # Database URLs
    MONGODB_URI: str
    QDRANT_URL: str
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 11046
    REDIS_PASSWORD: Optional[str] = None
    REDIS_TTL: int = 86400
    
    # Vector Store Configuration
    QDRANT_COLLECTION: str = "rag_documents"
    EMBEDDING_DIM: int = 384
    
    # LLM Configuration
    MODEL_NAME: str = "llama-3.1-8b-instant"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    
    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt"}
    
    # MongoDB Collections
    MONGO_DB_NAME: str = "rag_database"
    MONGO_DOCS_COLLECTION: str = "documents"
    MONGO_BOOKINGS_COLLECTION: str = "bookings"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()