# Custom RAG Backend

A custom Retrieval-Augmented Generation (RAG) backend system that enables intelligent document Q&A and seamless interview booking through conversational AI.

## Features

### Document Processing
- Document ingestion: Upload and process PDF and TXT files  
- Two chunking strategies: **Semantic (context-aware)** and **Fixed (size-based)**  
- Automatic embedding generation with SentenceTransformers  
- Vector storage in Qdrant for lightning-fast semantic search  

### Conversational RAG
- Multi-turn conversations with session memory  
- Context-aware responses using document knowledge  
- Retrieves top-K relevant chunks for accurate answers  
- Powered by **Groq Llama 3.1 (8B)** for fast inference  

### Intelligent Interview Booking System
- Automated interview scheduling through natural conversation  
- Intent detection using hybrid (keyword + LLM) approach  
- Progressive data collection: **Name → Email → Date → Time**  
- Input validation at each step with helpful error messages  
- Booking confirmation saved to **MongoDB**

---

## Quick Start

### Prerequisites
- Python 3.11+
- MongoDB instance (local or Atlas)
- Qdrant instance (local or Cloud)
- Redis instance (local or Cloud)
- Groq API key

---

## Installation


```bash
git clone https://github.com/ysanskar800/Custom_RAG_FastAPI.git
cd Custom_RAG_FastAPI

pip install -r requirements.txt

#Create a .env file in the project root
# Groq API
GROQ_API_KEY=your_groq_api_key_here

# MongoDB
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net

# Qdrant
QDRANT_URL=https://your-instance.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Redis
REDIS_HOST=your-redis-host.com
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

#Run the Server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

#Access the API
http://localhost:8000/docs
```

---

## Technology Stack

| Component            | Technology              | Purpose                                      |
|----------------------|--------------------------|----------------------------------------------|
| Backend Framework    | FastAPI                  | Async REST API server                        |
| Vector Database      | Qdrant                   | Semantic search & similarity matching        |
| Document Database    | MongoDB                  | Metadata & booking storage                   |
| Cache & Sessions     | Redis                    | Conversation memory & state                  |
| LLM                  | Groq (Llama 3.1 8B)      | Fast text generation                         |
| Embeddings           | SentenceTransformers     | Local embedding generation                   |
| PDF Processing       | PyPDF2                   | Extract text from PDFs                       |
| Validation           | Pydantic                 | Request/response validation                  |



