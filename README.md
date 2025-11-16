A production-ready Retrieval-Augmented Generation (RAG) backend system that enables intelligent document Q&A and seamless interview booking through conversational AI.

🌟 Features
📄 Document Ingestion

Upload and process PDF and TXT files
Two chunking strategies: Semantic (context-aware) and Fixed (size-based)
Automatic embedding generation with SentenceTransformers
Vector storage in Qdrant for lightning-fast semantic search

💬 Conversational RAG

Multi-turn conversations with session memory
Context-aware responses using document knowledge
Retrieves top-K relevant chunks for accurate answers
Powered by Groq's Llama 3.1 (8B) for fast inference

🗓️ Intelligent Booking System

Automated interview scheduling through natural conversation
Intent detection using hybrid (keyword + LLM) approach
Progressive data collection: Name → Email → Date → Time
Input validation at each step with helpful error messages
Booking confirmation saved to MongoDB


🚀 Quick Start
Prerequisites

Python 3.11+
MongoDB instance (local or Atlas)
Qdrant instance (local or Cloud)
Redis instance (local or Cloud)
Groq API key

Installation

Clone the repository

bashgit clone https://github.com/yourusername/rag-backend.git
cd rag-backend

Install dependencies

bashpip install -r requirements.txt

Configure environment variables

Create a .env file in the project root:
env# Groq API
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net
QDRANT_URL=https://your-instance.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
REDIS_HOST=your-redis-host.com
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_TTL=86400

Run the server

bashuvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Access the API: http://localhost:8000/docs
