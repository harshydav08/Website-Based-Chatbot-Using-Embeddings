"""
Core chatbot service that orchestrates all components.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from utils import WebCrawler, CrawledPage
from services import (
    TextProcessor, EmbeddingService, VectorDatabase, 
    RAGService, ConversationMemory
)
from config import config

logger = logging.getLogger(__name__)

class WebsiteChatbot:
    """
    Main chatbot class that orchestrates website crawling, indexing, and questioning.
    """
    
    def __init__(self):
        """Initialize the chatbot with all necessary services."""
        # Ensure directories exist
        config.ensure_directories()
        
        # Initialize core services
        self.text_processor = TextProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.embedding_service = EmbeddingService(config.EMBEDDING_MODEL_NAME)
        
        self.vector_db = VectorDatabase(
            config.CHROMADB_PERSIST_DIRECTORY,
            config.COLLECTION_NAME
        )
        
        self.rag_service = RAGService(
            embedding_service=self.embedding_service,
            vector_db=self.vector_db
        )
        
        self.conversation_memory = ConversationMemory()
        
        self.web_crawler = WebCrawler(
            max_pages=config.MAX_PAGES_TO_CRAWL,
            timeout=config.CRAWL_TIMEOUT
        )
        
        logger.info("Website Chatbot initialized successfully")
    
    def index_website(self, url: str) -> Dict[str, Any]:
        """
        Index a website by crawling, processing, and storing its content.
        
        Args:
            url: The website URL to index
            
        Returns:
            Dictionary with indexing results and statistics
        """
        try:
            logger.info(f"Starting website indexing for: {url}")
            
            # Step 1: Crawl the website
            logger.info("Crawling website...")
            crawled_pages = self.web_crawler.crawl_website(url)
            
            if not crawled_pages:
                return {
                    "success": False,
                    "error": "No content could be extracted from the website",
                    "stats": {"pages_crawled": 0, "chunks_created": 0}
                }
            
            logger.info(f"Crawled {len(crawled_pages)} pages")
            
            # Step 2: Process and chunk the content
            logger.info("Processing and chunking content...")
            text_chunks = self.text_processor.process_crawled_pages(crawled_pages)
            
            if not text_chunks:
                return {
                    "success": False,
                    "error": "No meaningful content could be processed from the crawled pages",
                    "stats": {"pages_crawled": len(crawled_pages), "chunks_created": 0}
                }
            
            logger.info(f"Created {len(text_chunks)} text chunks")
            
            # Step 3: Generate embeddings
            logger.info("Generating embeddings...")
            chunk_contents = [chunk.content for chunk in text_chunks]
            embeddings = self.embedding_service.generate_embeddings(chunk_contents)
            
            # Step 4: Store in vector database
            logger.info("Storing embeddings in vector database...")
            
            # Check if content from this URL already exists
            if self.vector_db.check_source_exists(url):
                logger.info(f"Removing existing content for URL: {url}")
                self.vector_db.delete_by_source(url)
            
            # Add new content
            self.vector_db.add_chunks(text_chunks, embeddings)
            
            # Get statistics
            chunk_summary = self.text_processor.get_chunk_summary(text_chunks)
            db_stats = self.vector_db.get_collection_stats()
            
            logger.info("Website indexing completed successfully")
            
            return {
                "success": True,
                "message": f"Successfully indexed {len(crawled_pages)} pages with {len(text_chunks)} chunks",
                "stats": {
                    "url": url,
                    "pages_crawled": len(crawled_pages),
                    "chunks_created": len(text_chunks),
                    "total_words": chunk_summary["total_words"],
                    "average_chunk_size": chunk_summary["average_chunk_size"],
                    "total_chunks_in_db": db_stats["total_chunks"],
                    "embedding_model": config.EMBEDDING_MODEL_NAME
                }
            }
            
        except Exception as e:
            logger.error(f"Error indexing website {url}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to index website: {str(e)}",
                "stats": {"pages_crawled": 0, "chunks_created": 0}
            }
    
    def ask_question(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question about the indexed website content.
        
        Args:
            question: The user's question
            session_id: Optional session ID for conversation context
            
        Returns:
            Dictionary with the answer and metadata
        """
        try:
            logger.info(f"Processing question in session {session_id}: {question[:100]}...")
            
            # Get conversation context if session provided
            conversation_history = []
            if session_id and self.conversation_memory.session_exists(session_id):
                conversation_history = self.conversation_memory.get_recent_context(session_id)
                
                # Add user question to conversation memory
                self.conversation_memory.add_message(session_id, "user", question)
            
            # Get answer using RAG
            response = self.rag_service.ask_question(question, conversation_history)
            
            # Add assistant response to conversation memory
            if session_id and self.conversation_memory.session_exists(session_id):
                self.conversation_memory.add_message(
                    session_id, 
                    "assistant", 
                    response["answer"],
                    {
                        "confidence": response.get("confidence", 0.0),
                        "sources": response.get("sources", []),
                        "chunks_used": response.get("chunks_used", 0)
                    }
                )
            
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
    
    def create_conversation_session(self) -> str:
        """
        Create a new conversation session.
        
        Returns:
            New session ID
        """
        return self.conversation_memory.create_session()
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of messages in the conversation
        """
        return self.conversation_memory.get_conversation_history(session_id)
    
    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear a conversation session.
        
        Args:
            session_id: The session ID to clear
            
        Returns:
            True if successful, False otherwise
        """
        return self.conversation_memory.clear_session(session_id)
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the current database state.
        
        Returns:
            Database statistics and information
        """
        try:
            db_stats = self.vector_db.get_collection_stats()
            memory_stats = self.conversation_memory.get_memory_stats()
            
            return {
                "database": db_stats,
                "memory": memory_stats,
                "models": {
                    "embedding_model": config.EMBEDDING_MODEL_NAME,
                    "llm_model": "Simple Rule-based QA",
                    "chunk_size": config.CHUNK_SIZE,
                    "chunk_overlap": config.CHUNK_OVERLAP
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
            return {"error": str(e)}
    
    def clear_database(self) -> bool:
        """
        Clear all data from the vector database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_db.clear_collection()
            logger.info("Database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a URL for crawling.
        
        Args:
            url: The URL to validate
            
        Returns:
            Tuple of (is_valid, normalized_url, error_message)
        """
        return self.web_crawler.url_validator.validate_url(url)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status and health.
        
        Returns:
            System status information
        """
        try:
            # Check if all services are working
            embedding_info = self.embedding_service.get_model_info()
            db_stats = self.vector_db.get_collection_stats()
            memory_stats = self.conversation_memory.get_memory_stats()
            
            return {
                "status": "healthy",
                "services": {
                    "embedding_service": embedding_info,
                    "vector_database": {
                        "status": "connected" if "error" not in db_stats else "error",
                        "total_chunks": db_stats.get("total_chunks", 0)
                    },
                    "conversation_memory": {
                        "status": "active",
                        "active_sessions": memory_stats["total_sessions"]
                    }
                },
                "config": config.get_config_dict()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
