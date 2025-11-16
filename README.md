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


