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

### 1. Clone the Repository
```bash
git clone https://github.com/ysanskar800/Custom_RAG_FastAPI.git
cd rag-backend

### 2. Install Dependencies
```bash
pip install -r requirements.txt
