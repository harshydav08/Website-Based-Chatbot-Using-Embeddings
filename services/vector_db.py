"""
Vector database service using ChromaDB for persistent storage and retrieval.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Vector database service using ChromaDB for storing and querying embeddings.
    """
    
    def __init__(self, persist_directory: str, collection_name: str = "website_content"):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to store embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._initialize_client()
        self._get_or_create_collection()
    
    def _initialize_client(self):
        """Initialize the ChromaDB client."""
        try:
            logger.info(f"Initializing ChromaDB client with persist directory: {self.persist_directory}")
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def _get_or_create_collection(self):
        """Get or create the collection."""
        try:
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Retrieved existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Website content embeddings for chatbot"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to get or create collection: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List, embeddings: List[List[float]]):
        """
        Add text chunks and their embeddings to the database.
        
        Args:
            chunks: List of TextChunk objects
            embeddings: List of embedding vectors
        """
        if not chunks or len(embeddings) == 0:
            logger.warning("No chunks or embeddings provided")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) doesn't match number of embeddings ({len(embeddings)})")
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings_list = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Generate unique ID if chunk doesn't have one
                chunk_id = getattr(chunk, 'chunk_id', None) or str(uuid.uuid4())
                ids.append(chunk_id)
                
                # Add content
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB doesn't support nested objects)
                metadata = {}
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    else:
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
                embeddings_list.append(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            
            logger.info(f"Added {len(chunks)} chunks to the vector database")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector database: {str(e)}")
            raise
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, **filters) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            **filters: Additional filters for metadata
            
        Returns:
            List of search results with content and metadata
        """
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = value
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:  # Check if we have results
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "similarity": 1 - distance  # Convert distance to similarity
                    })
            
            logger.debug(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            stats = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": str(self.persist_directory)
            }
            
            # Get some sample metadata if available
            if count > 0:
                sample_results = self.collection.peek(limit=1)
                if sample_results and sample_results.get("metadatas"):
                    sample_metadata = sample_results["metadatas"][0]
                    stats["sample_metadata_keys"] = list(sample_metadata.keys())
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self):
        """Clear all data from the collection."""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Website content embeddings for chatbot"}
            )
            
            logger.info("Collection cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise
    
    def delete_by_source(self, source_url: str):
        """
        Delete all chunks from a specific source URL.
        
        Args:
            source_url: The source URL to delete chunks for
        """
        try:
            # Get all items with the specified source URL
            results = self.collection.get(
                where={"source_url": source_url},
                include=["documents", "metadatas"]
            )
            
            if results["ids"]:
                # Delete the items
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks from {source_url}")
            else:
                logger.info(f"No chunks found for source URL: {source_url}")
                
        except Exception as e:
            logger.error(f"Failed to delete chunks by source: {str(e)}")
            raise
    
    def check_source_exists(self, source_url: str) -> bool:
        """
        Check if chunks from a source URL already exist in the database.
        
        Args:
            source_url: The source URL to check
            
        Returns:
            True if chunks exist, False otherwise
        """
        try:
            results = self.collection.get(
                where={"source_url": source_url},
                limit=1
            )
            
            return len(results["ids"]) > 0
            
        except Exception as e:
            logger.error(f"Failed to check if source exists: {str(e)}")
            return False
