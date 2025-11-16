import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.config import settings

class VectorStoreService:
    """Service for managing Qdrant vector store operations."""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists, create if it doesn't."""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
    
    def store_chunks(
        self, 
        doc_id: str, 
        filename: str, 
        chunks: List[str], 
        vectors: List[List[float]]
    ) -> List[str]:
        """
        Store document chunks with their embeddings in Qdrant.
        
        Args:
            doc_id: Unique document identifier
            filename: Original filename
            chunks: List of text chunks
            vectors: List of embedding vectors
            
        Returns:
            List of chunk IDs
        """
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors")
        
        points = []
        chunk_ids = []
        
        for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            point = PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_idx": idx,
                    "filename": filename
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return chunk_ids
    
    def search_similar(
        self, 
        query_vector: List[float], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar chunks in the vector store.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of search results with text, score, and metadata
        """
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        results = []
        for point in search_result.points:
            payload = point.payload or {}
            results.append({
                "text": payload.get("text", ""),
                "score": getattr(point, "score", None),
                "filename": payload.get("filename"),
                "doc_id": payload.get("doc_id"),
                "chunk_id": getattr(point, "id", None),
                "chunk_idx": payload.get("chunk_idx")
            })
        
        return results
    
    def delete_document_chunks(self, doc_id: str):
        """
        Delete all chunks belonging to a document.
        
        Args:
            doc_id: Document identifier
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"value": doc_id}
                        }
                    ]
                }
            }
        )
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.
        
        Returns:
            Collection information
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection.vectors_count,
                "points_count": collection.points_count,
                "status": collection.status
            }
        except Exception as e:
            return {"error": str(e)}

# Singleton instance
vector_store_service = VectorStoreService()