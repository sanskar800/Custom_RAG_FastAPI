from typing import List, Dict, Optional
from groq import Groq
from pymongo import MongoClient
from datetime import datetime, timezone
from app.config import settings
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store_service
from app.services.redis_memory import redis_memory_service

class RAGService:
    """Service for RAG pipeline operations."""
    
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.mongo_client = MongoClient(settings.MONGODB_URI)
        self.db = self.mongo_client[settings.MONGO_DB_NAME]
        self.docs_collection = self.db[settings.MONGO_DOCS_COLLECTION]
    
    def save_document_metadata(
        self, 
        doc_id: str, 
        filename: str, 
        num_chunks: int, 
        chunk_ids: List[str],
        chunking_method: str
    ):
        """
        Save document metadata to MongoDB.
        
        Args:
            doc_id: Document identifier
            filename: Original filename
            num_chunks: Number of chunks created
            chunk_ids: List of chunk IDs
            chunking_method: Method used for chunking
        """
        record = {
            "doc_id": doc_id,
            "filename": filename,
            "num_chunks": num_chunks,
            "chunk_ids": chunk_ids,
            "chunking_method": chunking_method,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self.docs_collection.insert_one(record)
    
    def search_documents(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_vector = embedding_service.embed_text(query)
        
        # Search in vector store
        results = vector_store_service.search_similar(query_vector, top_k=top_k)
        
        return results
    
    def build_context(self, search_results: List[Dict]) -> str:
        """
        Build context string from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for idx, result in enumerate(search_results, 1):
            filename = result.get("filename", "Unknown")
            chunk_idx = result.get("chunk_idx", "?")
            text = result.get("text", "")
            score = result.get("score", 0.0)
            
            context_parts.append(
                f"[Source {idx}: {filename}, Chunk {chunk_idx}, Relevance: {score:.2f}]\n{text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(
        self, 
        question: str, 
        context: str, 
        conversation_history: List[Dict]
    ) -> str:
        """
        Generate answer using LLM with context and history.
        
        Args:
            question: User's question
            context: Retrieved context
            conversation_history: Recent conversation history
            
        Returns:
            Generated answer
        """
        system_prompt = (
            "You are a helpful and knowledgeable assistant. Your task is to answer questions "
            "based on the provided context from documents.\n\n"
            "Guidelines:\n"
            "- Use the context to provide accurate and relevant answers\n"
            "- If the exact answer isn't in the context but related information is present, "
            "use that information to provide a helpful response\n"
            "- Only say 'I don't know' or 'The information is not available' if the context "
            "is completely unrelated to the question\n"
            "- Be concise, clear, and accurate\n"
            "- Cite sources when appropriate (e.g., 'According to the document...')\n"
            "- If asked about information not in the context, be honest about it"
        )
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 12 messages for context)
        if conversation_history:
            messages.extend(conversation_history[-12:])
        
        # Add current query with context
        user_message = f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.groq_client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def process_query(
        self, 
        session_id: str, 
        query: str, 
        top_k: int = 5
    ) -> str:
        """
        Process a RAG query end-to-end.
        
        Args:
            session_id: Session identifier
            query: User's query
            top_k: Number of chunks to retrieve
            
        Returns:
            Generated answer
        """
        # Search for relevant chunks
        search_results = self.search_documents(query, top_k=top_k)
        
        if not search_results:
            return (
                "I don't have any documents in my knowledge base yet. "
                "Please upload documents first using the /ingest endpoint."
            )
        
        # Build context from results
        context = self.build_context(search_results)
        
        # Get conversation history
        conversation_history = redis_memory_service.get_conversation_context(session_id)
        
        # Generate answer
        answer = self.generate_answer(query, context, conversation_history)
        
        return answer
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get all document metadata.
        
        Returns:
            List of document records
        """
        documents = list(self.docs_collection.find({}, {"_id": 0}))
        
        # Convert datetime to string
        for doc in documents:
            if "timestamp" in doc:
                doc["timestamp"] = doc["timestamp"].isoformat()
        
        return documents

# Singleton instance
rag_service = RAGService()