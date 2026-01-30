"""
RAG (Retrieval-Augmented Generation) service that combines retrieval and generation.
"""

from typing import List, Dict, Any, Optional
import logging
from services.embedding_service import EmbeddingService
from services.vector_db import VectorDatabase
from services.llm_service import SimpleQALLM
from config import config

logger = logging.getLogger(__name__)

class RAGService:
    """
    RAG service that combines retrieval from vector database with LLM generation.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        vector_db: VectorDatabase = None,
        llm_service = None
    ):
        """
        Initialize the RAG service.
        
        Args:
            embedding_service: Embedding service for query encoding
            vector_db: Vector database for retrieval
            llm_service: LLM service for response generation
        """
        self.embedding_service = embedding_service or EmbeddingService(config.EMBEDDING_MODEL_NAME)
        self.vector_db = vector_db or VectorDatabase(
            config.CHROMADB_PERSIST_DIRECTORY,
            config.COLLECTION_NAME
        )
        
        # Use SimpleQALLM for reliable, fast responses without requiring heavy models
        self.llm_service = llm_service or SimpleQALLM()
        
        logger.info("RAG service initialized")
    
    def ask_question(self, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Ask a question and get an answer using RAG.
        
        Args:
            question: The user's question
            conversation_history: Previous conversation for context
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Step 1: Create query embedding
            query_embedding = self._create_query_embedding(question, conversation_history)
            
            # Step 2: Retrieve relevant chunks
            retrieved_chunks = self._retrieve_relevant_chunks(query_embedding)
            
            if not retrieved_chunks:
                logger.info("No relevant content found in database")
                return {
                    "answer": config.NOT_FOUND_RESPONSE,
                    "sources": [],
                    "confidence": 0.0,
                    "chunks_used": 0
                }
            
            # Step 3: Generate response using retrieved context
            response = self._generate_response(question, retrieved_chunks, conversation_history)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question.",
                "sources": [],
                "confidence": 0.0,
                "chunks_used": 0,
                "error": str(e)
            }
    
    def _create_query_embedding(self, question: str, conversation_history: List[Dict] = None) -> List[float]:
        """
        Create an embedding for the query, potentially enhanced with conversation context.
        
        Args:
            question: The current question
            conversation_history: Previous conversation context
            
        Returns:
            Query embedding vector
        """
        try:
            # For now, just embed the current question
            # In a more sophisticated system, we might combine with recent conversation context
            query_text = question
            
            # Optionally enhance with recent context
            if conversation_history and len(conversation_history) > 0:
                # Take the last user message as additional context
                recent_messages = conversation_history[-2:]  # Last 2 messages
                context_text = " ".join([msg.get("content", "") for msg in recent_messages])
                if context_text.strip():
                    query_text = f"{context_text} {question}"
            
            embedding = self.embedding_service.generate_embeddings(query_text)
            
            # Convert to list if it's a numpy array
            # embedding_service returns shape (1, 384) for single text, so we need the first element
            if hasattr(embedding, 'tolist'):
                # If it's 2D (shape: (1, 384)), get the first row
                if len(embedding.shape) > 1:
                    return embedding[0].tolist()
                else:
                    return embedding.tolist()
            else:
                # Already a list
                return embedding[0] if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list) else embedding
                
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            raise
    
    def _retrieve_relevant_chunks(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from the vector database.
        
        Args:
            query_embedding: The query embedding vector
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Search for similar chunks
            results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                top_k=config.TOP_K_RESULTS
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result.get("similarity", 0) >= config.SIMILARITY_THRESHOLD
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} relevant chunks (threshold: {config.SIMILARITY_THRESHOLD})")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            raise
    
    def _generate_response(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the LLM based on retrieved chunks.
        
        Args:
            question: The user's question
            retrieved_chunks: Retrieved relevant chunks
            conversation_history: Previous conversation context
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            if not retrieved_chunks:
                return {
                    "answer": config.NOT_FOUND_RESPONSE,
                    "sources": [],
                    "confidence": 0.0,
                    "chunks_used": 0
                }
            
            # Prepare context from retrieved chunks
            context_parts = []
            sources = set()
            
            for chunk in retrieved_chunks:
                context_parts.append(chunk["content"])
                source_url = chunk["metadata"].get("source_url", "Unknown")
                if source_url != "Unknown":
                    sources.add(source_url)
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for the LLM
            prompt = self._create_prompt(question, context, conversation_history)
            
            # Generate response using LLM
            answer = self.llm_service.generate_response(prompt)
            
            # Clean up the response
            answer = self._clean_response(answer)
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(chunk.get("similarity", 0) for chunk in retrieved_chunks) / len(retrieved_chunks)
            
            return {
                "answer": answer,
                "sources": list(sources),
                "confidence": avg_similarity,
                "chunks_used": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks[:3]  # Include top 3 for debugging
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while generating the response.",
                "sources": [],
                "confidence": 0.0,
                "chunks_used": 0,
                "error": str(e)
            }
    
    def _create_prompt(self, question: str, context: str, conversation_history: List[Dict] = None) -> str:
        """
        Create a prompt for the LLM that encourages answering only from the provided context.
        
        Args:
            question: The user's question
            context: Retrieved context from the website
            conversation_history: Previous conversation context
            
        Returns:
            Formatted prompt for the LLM
        """
        # Base prompt that emphasizes answering only from the provided context
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided website content.

IMPORTANT INSTRUCTIONS:
- Answer ONLY using information from the provided context below
- If the answer is not in the context, respond with EXACTLY: "The answer is not available on the provided website."
- Do not use any external knowledge or information not in the context
- Keep your answer concise and directly relevant to the question
- Do not make up or infer information not explicitly stated in the context

Context:
{context}

Question: {question}

Answer:"""

        return prompt
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and validate the LLM response.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response
        """
        if not response or not response.strip():
            return config.NOT_FOUND_RESPONSE
        
        # Remove common LLM artifacts
        response = response.strip()
        
        # Remove "Answer:" prefix if it exists
        if response.lower().startswith("answer:"):
            response = response[7:].strip()
        
        # If response is too short or seems invalid, return not found
        if len(response) < 10:
            return config.NOT_FOUND_RESPONSE
        
        # Ensure the response ends with proper punctuation
        if response and not response[-1] in '.!?':
            response += '.'
        
        return response
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        try:
            return self.vector_db.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_database(self):
        """Clear all data from the vector database."""
        try:
            self.vector_db.clear_collection()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            raise
