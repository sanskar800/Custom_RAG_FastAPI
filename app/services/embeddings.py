from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    """Service for generating text embeddings."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        embedding = self._model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            return []
        
        embeddings = self._model.encode(valid_texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        return self._model.get_sentence_embedding_dimension()

# Singleton instance
embedding_service = EmbeddingService()