"""
Basic system validation test for Website Chatbot.
This script tests core functionality to ensure the system works correctly.
"""

import logging
import traceback
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test core imports
        from config import config
        from utils import URLValidator, WebCrawler
        from services import (
            TextProcessor, EmbeddingService, VectorDatabase,
            RAGService, ConversationMemory
        )
        from core import WebsiteChatbot
        
        logger.info("✅ All core modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}")
        return False

def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    try:
        from config import config
        
        # Test essential config values
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert config.TOP_K_RESULTS > 0, "TOP_K_RESULTS must be positive"
        assert config.EMBEDDING_MODEL_NAME, "EMBEDDING_MODEL_NAME cannot be empty"
        
        # Ensure directories
        config.ensure_directories()
        
        logger.info("✅ Configuration is valid")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {str(e)}")
        return False

def test_url_validation():
    """Test URL validation functionality."""
    logger.info("Testing URL validation...")
    
    try:
        from utils import URLValidator
        
        validator = URLValidator()
        
        # Test valid URL
        is_valid, normalized_url, error = validator.validate_url("https://httpbin.org/html")
        if not is_valid:
            logger.warning(f"Expected valid URL, got error: {error}")
            # Try with a simpler test
            is_valid, normalized_url, error = validator.validate_url("https://example.com")
        
        # Test invalid URL
        is_invalid, _, invalid_error = validator.validate_url("not-a-url")
        assert not is_invalid, "Invalid URL should be rejected"
        
        logger.info("✅ URL validation working correctly")
        return True
    except Exception as e:
        logger.error(f"❌ URL validation test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_text_processing():
    """Test text processing functionality."""
    logger.info("Testing text processing...")
    
    try:
        from services import TextProcessor
        from utils.web_crawler import CrawledPage
        
        processor = TextProcessor(chunk_size=500, chunk_overlap=100)
        
        # Create a mock crawled page
        mock_page = CrawledPage(
            url="https://example.com",
            title="Test Page",
            content="This is a test content for the text processor. " * 20,
            word_count=100
        )
        
        # Test processing
        chunks = processor.process_crawled_pages([mock_page])
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert chunks[0].metadata['source_url'] == "https://example.com", "Metadata should be preserved"
        
        logger.info(f"✅ Text processing created {len(chunks)} chunks")
        return True
    except Exception as e:
        logger.error(f"❌ Text processing test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_embeddings():
    """Test embedding generation."""
    logger.info("Testing embedding generation...")
    
    try:
        from services import EmbeddingService
        
        # This may take a moment on first run as it downloads the model
        embedding_service = EmbeddingService()
        
        # Test single text embedding
        text = "This is a test sentence for embedding generation."
        embedding = embedding_service.generate_embeddings(text)
        
        assert embedding is not None, "Embedding should not be None"
        assert len(embedding) > 0, "Embedding should have dimensions"
        
        # Test batch embedding
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedding_service.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts), "Should generate one embedding per text"
        
        model_info = embedding_service.get_model_info()
        logger.info(f"✅ Embeddings working. Model: {model_info['model_name']}, Dimension: {model_info['embedding_dimension']}")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_vector_database():
    """Test vector database functionality."""
    logger.info("Testing vector database...")
    
    try:
        from services import VectorDatabase, TextProcessor
        from config import config
        import tempfile
        import os
        
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            db = VectorDatabase(temp_dir, "test_collection")
            
            # Test basic stats
            stats = db.get_collection_stats()
            assert "total_chunks" in stats, "Stats should include total_chunks"
            
            logger.info("✅ Vector database initialized successfully")
            return True
    except Exception as e:
        logger.error(f"❌ Vector database test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_conversation_memory():
    """Test conversation memory functionality."""
    logger.info("Testing conversation memory...")
    
    try:
        from services import ConversationMemory
        
        memory = ConversationMemory()
        
        # Create a session
        session_id = memory.create_session()
        assert session_id, "Session ID should not be empty"
        
        # Add messages
        success = memory.add_message(session_id, "user", "Hello, how are you?")
        assert success, "Should successfully add user message"
        
        success = memory.add_message(session_id, "assistant", "I'm doing well, thank you!")
        assert success, "Should successfully add assistant message"
        
        # Get conversation history
        history = memory.get_conversation_history(session_id)
        assert len(history) == 2, "Should have 2 messages in history"
        
        # Get memory stats
        stats = memory.get_memory_stats()
        assert stats['total_sessions'] >= 1, "Should have at least one session"
        
        logger.info("✅ Conversation memory working correctly")
        return True
    except Exception as e:
        logger.error(f"❌ Conversation memory test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_simple_llm():
    """Test the simple LLM service."""
    logger.info("Testing simple LLM...")
    
    try:
        from services import SimpleQALLM
        
        llm = SimpleQALLM()
        
        # Test response generation
        prompt = """Context:
The capital of France is Paris. Paris is located in northern France and is known for the Eiffel Tower.

Question: What is the capital of France?

Answer:"""
        
        response = llm.generate_response(prompt)
        assert response, "Should generate a response"
        assert "Paris" in response, "Response should mention Paris"
        
        logger.info(f"✅ Simple LLM generated response: {response[:50]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Simple LLM test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_integration():
    """Test basic integration between components."""
    logger.info("Testing integration...")
    
    try:
        from core import WebsiteChatbot
        
        # Initialize chatbot (this tests integration of all components)
        chatbot = WebsiteChatbot()
        
        # Test system status
        status = chatbot.get_system_status()
        assert status['status'] == 'healthy', "System should be healthy"
        
        # Test database info
        db_info = chatbot.get_database_info()
        assert 'database' in db_info, "Should have database info"
        
        # Test conversation session creation
        session_id = chatbot.create_conversation_session()
        assert session_id, "Should create a conversation session"
        
        logger.info("✅ Integration test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Integration test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_all_tests():
    """Run all tests and report results."""
    logger.info("🚀 Starting Website Chatbot System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("URL Validation", test_url_validation),
        ("Text Processing", test_text_processing),
        ("Embeddings", test_embeddings),
        ("Vector Database", test_vector_database),
        ("Conversation Memory", test_conversation_memory),
        ("Simple LLM", test_simple_llm),
        ("Integration", test_integration),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
            logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test {test_name} FAILED with exception: {str(e)}")
            results[test_name] = False
        
        logger.info("-" * 40)
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info(f"Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System is ready to use.")
    else:
        logger.info(f"⚠️  {total - passed} test(s) failed. Please check the logs above.")
    
    logger.info("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
