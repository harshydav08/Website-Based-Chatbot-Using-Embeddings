"""
Configuration settings for the Website Chatbot application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Configuration class that manages all application settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "microsoft/DialoGPT-medium")
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    
    # Embedding Configuration
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Vector Database Configuration
    CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", str(CHROMA_DB_DIR))
    COLLECTION_NAME = "website_content"
    
    # Text Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Crawling Configuration
    MAX_PAGES_TO_CRAWL = int(os.getenv("MAX_PAGES_TO_CRAWL", "50"))
    CRAWL_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "30"))
    
    # Response Configuration
    NOT_FOUND_RESPONSE = "The answer is not available on the provided website."
    
    # UI Configuration
    PAGE_TITLE = "Website Chatbot"
    PAGE_ICON = "🤖"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return configuration as dictionary for debugging."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "embedding_model": cls.EMBEDDING_MODEL_NAME,
            "llm_model": cls.LLM_MODEL_NAME,
            "top_k_results": cls.TOP_K_RESULTS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "max_pages_to_crawl": cls.MAX_PAGES_TO_CRAWL,
            "crawl_timeout": cls.CRAWL_TIMEOUT
        }

# Global config instance
config = Config()
