from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from app.config import settings
from app.routes import ingest, chat

# Create uploads directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    print("=" * 60)
    print("RAG Backend Starting...")
    print("=" * 60)
    print(f"Upload Directory: {settings.UPLOAD_DIR}")
    print(f"Qdrant URL: {settings.QDRANT_URL}")
    print(f"MongoDB URI: {'Connected' if settings.MONGODB_URI else 'Not configured'}")
    print(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Embedding Dimension: {settings.EMBEDDING_DIM}")
    print(f"Chunk Size: {settings.CHUNK_SIZE}")
    print("=" * 60)
    
    # Initialize services (this ensures connections are established)
    try:
        from app.services.vector_store import vector_store_service
        from app.services.redis_memory import redis_memory_service
        from app.services.embeddings import embedding_service
        from app.services.rag import rag_service
        from app.services.booking_engine import booking_engine
        
        print("✓ Vector Store Service initialized")
        print("✓ Redis Memory Service initialized")
        print("✓ Embedding Service initialized")
        print("✓ RAG Service initialized")
        print("✓ Booking Engine initialized")
        print("=" * 60)
        print("Server ready to accept requests!")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Failed to initialize services: {e}")
        print("=" * 60)
    
    yield
    
    # Shutdown
    print("\nShutting down RAG Backend...")

# Create FastAPI app
app = FastAPI(
    title="RAG Backend API",
    description=(
        "Production-level RAG backend with document ingestion, "
        "conversational AI, and interview booking capabilities."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/api", tags=["Document Ingestion"])
app.include_router(chat.router, prefix="/api", tags=["Conversational RAG & Booking"])

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Backend API",
        "version": "1.0.0",
        "endpoints": {
            "ingest": "/api/ingest - Upload and process documents",
            "chat": "/api/chat - Conversational RAG with booking",
            "history": "/api/chat/history/{session_id} - Get conversation history",
            "clear": "/api/chat/session/{session_id} - Clear session data",
            "health": "/health - Health check",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "services": {}
    }
    
    # Check Qdrant
    try:
        from app.services.vector_store import vector_store_service
        collection_info = vector_store_service.get_collection_info()
        health_status["services"]["qdrant"] = {
            "status": "connected",
            "collection": collection_info
        }
    except Exception as e:
        health_status["services"]["qdrant"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        from app.services.redis_memory import redis_memory_service
        redis_memory_service.client.ping()
        health_status["services"]["redis"] = {"status": "connected"}
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check MongoDB
    try:
        from app.services.rag import rag_service
        rag_service.mongo_client.admin.command('ping')
        health_status["services"]["mongodb"] = {"status": "connected"}
    except Exception as e:
        health_status["services"]["mongodb"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Groq
    try:
        from app.services.booking_engine import booking_engine
        if booking_engine.groq_client:
            health_status["services"]["groq"] = {"status": "configured"}
        else:
            health_status["services"]["groq"] = {"status": "not_configured"}
    except Exception as e:
        health_status["services"]["groq"] = {
            "status": "error",
            "error": str(e)
        }
    
    if health_status["status"] == "degraded":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )