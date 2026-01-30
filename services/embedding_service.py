"""
Embedding service for generating text embeddings using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using sentence-transformers models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings or list of numpy arrays
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} text(s)")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.model:
            return {"model_name": self.model_name, "loaded": False}
        
        return {
            "model_name": self.model_name,
            "loaded": True,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'Unknown')
        }
