"""
Streamlit application for the Website Chatbot.
"""

import streamlit as st
import logging
from typing import Dict, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our chatbot
from core import WebsiteChatbot
from config import config

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize chatbot (with caching to avoid reloading)
@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot with caching."""
    try:
        return WebsiteChatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return None

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
    
    if 'conversation_session_id' not in st.session_state:
        if st.session_state.chatbot:
            st.session_state.conversation_session_id = st.session_state.chatbot.create_conversation_session()
        else:
            st.session_state.conversation_session_id = None
    
    if 'indexed_urls' not in st.session_state:
        st.session_state.indexed_urls = []
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_header():
    """Display the application header."""
    st.title("🤖 Website Chatbot")
    st.markdown("Ask questions about any website's content using AI-powered embeddings and retrieval.")
    
    # Display system status in the sidebar
    with st.sidebar:
        st.header("System Status")
        if st.session_state.chatbot:
            try:
                status = st.session_state.chatbot.get_system_status()
                if status["status"] == "healthy":
                    st.success("✅ System is healthy")
                    
                    # Display basic stats
                    db_info = st.session_state.chatbot.get_database_info()
                    if "database" in db_info:
                        st.metric("Indexed Chunks", db_info["database"].get("total_chunks", 0))
                    
                    if "memory" in db_info:
                        st.metric("Active Sessions", db_info["memory"].get("total_sessions", 0))
                        
                else:
                    st.error(f"❌ System error: {status.get('error', 'Unknown')}")
            except Exception as e:
                st.error(f"❌ Status check failed: {str(e)}")
        else:
            st.error("❌ Chatbot not initialized")

def display_url_indexing_section():
    """Display the URL indexing section."""
    st.header("📚 Index Website Content")
    
    with st.form("url_form"):
        url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter a website URL to crawl and index its content"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            index_button = st.form_submit_button("🔍 Index Website", type="primary")
        with col2:
            if st.form_submit_button("🗑️ Clear Database"):
                if st.session_state.chatbot:
                    with st.spinner("Clearing database..."):
                        success = st.session_state.chatbot.clear_database()
                    if success:
                        st.success("Database cleared successfully!")
                        st.session_state.indexed_urls = []
                        st.rerun()
                    else:
                        st.error("Failed to clear database")
    
    if index_button and url and st.session_state.chatbot:
        # Validate URL first
        with st.spinner("Validating URL..."):
            is_valid, normalized_url, error = st.session_state.chatbot.validate_url(url)
        
        if not is_valid:
            st.error(f"❌ Invalid URL: {error}")
            return
        
        # Index the website
        with st.spinner(f"Indexing website: {normalized_url}"):
            start_time = time.time()
            result = st.session_state.chatbot.index_website(normalized_url)
            end_time = time.time()
        
        if result["success"]:
            st.success(f"✅ {result['message']}")
            
            # Display statistics
            stats = result["stats"]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pages Crawled", stats["pages_crawled"])
            with col2:
                st.metric("Chunks Created", stats["chunks_created"])
            with col3:
                st.metric("Processing Time", f"{end_time - start_time:.1f}s")
            
            # Additional stats
            with st.expander("📊 Detailed Statistics"):
                st.json(stats)
            
            # Add to indexed URLs
            if normalized_url not in st.session_state.indexed_urls:
                st.session_state.indexed_urls.append(normalized_url)
        else:
            st.error(f"❌ Indexing failed: {result['error']}")
            
            # Show partial stats if available
            if "stats" in result:
                st.info(f"Pages crawled: {result['stats']['pages_crawled']}")

def display_indexed_urls():
    """Display the list of indexed URLs."""
    if st.session_state.indexed_urls:
        st.subheader("📋 Indexed Websites")
        for i, url in enumerate(st.session_state.indexed_urls, 1):
            st.write(f"{i}. {url}")

def display_chat_interface():
    """Display the chat interface."""
    st.header("💬 Ask Questions")
    
    if not st.session_state.chatbot:
        st.error("Chatbot not available. Please refresh the page.")
        return
    
    # Check if we have indexed content
    db_info = st.session_state.chatbot.get_database_info()
    total_chunks = db_info.get("database", {}).get("total_chunks", 0)
    
    if total_chunks == 0:
        st.info("ℹ️ Please index a website first before asking questions.")
        return
    
    st.info(f"Ready to answer questions about {total_chunks} indexed content chunks.")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Create columns for metadata
                    if metadata.get("sources") or metadata.get("confidence", 0) > 0:
                        with st.expander("📊 Response Details"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                confidence = metadata.get("confidence", 0)
                                st.metric("Confidence", f"{confidence:.2%}")
                            
                            with col2:
                                chunks_used = metadata.get("chunks_used", 0)
                                st.metric("Chunks Used", chunks_used)
                            
                            with col3:
                                sources = metadata.get("sources", [])
                                st.metric("Sources", len(sources))
                            
                            # Display sources
                            if sources:
                                st.write("**Sources:**")
                                for i, source in enumerate(sources, 1):
                                    st.write(f"{i}. {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the indexed website content..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from chatbot
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.ask_question(
                        prompt, 
                        st.session_state.conversation_session_id
                    )
                
                answer = response["answer"]
                st.write(answer)
                
                # Display metadata
                if response.get("sources") or response.get("confidence", 0) > 0:
                    with st.expander("📊 Response Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            confidence = response.get("confidence", 0)
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        with col2:
                            chunks_used = response.get("chunks_used", 0)
                            st.metric("Chunks Used", chunks_used)
                        
                        with col3:
                            sources = response.get("sources", [])
                            st.metric("Sources", len(sources))
                        
                        # Display sources
                        if sources:
                            st.write("**Sources:**")
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source}")
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metadata": {
                        "confidence": response.get("confidence", 0),
                        "sources": response.get("sources", []),
                        "chunks_used": response.get("chunks_used", 0)
                    }
                })

def display_conversation_controls():
    """Display conversation control buttons."""
    with st.sidebar:
        st.header("💬 Conversation Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New Chat"):
                # Clear messages and create new session
                st.session_state.messages = []
                if st.session_state.chatbot:
                    st.session_state.conversation_session_id = st.session_state.chatbot.create_conversation_session()
                st.rerun()
        
        with col2:
            if st.button("📥 Export Chat"):
                if st.session_state.messages:
                    chat_text = "\n\n".join([
                        f"{msg['role'].title()}: {msg['content']}" 
                        for msg in st.session_state.messages
                    ])
                    st.download_button(
                        label="💾 Download Chat",
                        data=chat_text,
                        file_name=f"website_chat_{int(time.time())}.txt",
                        mime="text/plain"
                    )

def display_system_info():
    """Display system information in sidebar."""
    with st.sidebar:
        if st.expander("⚙️ System Information"):
            if st.session_state.chatbot:
                system_status = st.session_state.chatbot.get_system_status()
                if "config" in system_status:
                    st.json(system_status["config"])
                else:
                    st.write("Config information not available")
            else:
                st.write("Chatbot not initialized")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # URL indexing section
        display_url_indexing_section()
        
        # Chat interface
        display_chat_interface()
    
    with col2:
        # Indexed URLs
        display_indexed_urls()
        
        # Conversation controls
        display_conversation_controls()
        
        # System info
        display_system_info()

if __name__ == "__main__":
    main()
