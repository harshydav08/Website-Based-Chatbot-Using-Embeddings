# Website-Based Chatbot Using Embeddings

A production-ready AI-powered chatbot that crawls websites, processes their content into embeddings, and provides intelligent question-answering capabilities using Retrieval-Augmented Generation (RAG).

## 🌟 Features

- **Website Crawling**: Intelligent crawling with content extraction, removing headers, footers, navigation, and advertisements
- **Text Processing**: Advanced text cleaning and semantic chunking with configurable parameters
- **Embeddings**: Uses sentence-transformers for high-quality text embeddings
- **Vector Database**: Persistent storage using ChromaDB for efficient similarity search
- **RAG System**: Combines retrieval and generation for accurate, context-aware responses
- **Conversational Memory**: Maintains short-term conversation context within sessions
- **Streamlit UI**: User-friendly web interface for indexing websites and asking questions
- **Error Handling**: Robust error handling and validation throughout the system

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Crawler   │────│  Text Processor │────│ Embedding Service│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  URL Validator  │    │  Text Chunks    │    │   Vector DB     │
└─────────────────┘    └─────────────────┘    │   (ChromaDB)    │
                                               └─────────────────┘
                                                       │
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◄───│   RAG Service   │◄───│  Conversation   │
│                 │    │                 │    │    Memory       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         └──────────────│   LLM Service   │
                        │  (Rule-based)   │
                        └─────────────────┘
```

### Core Components

1. **Web Crawler** (`utils/web_crawler.py`)
   - Validates and normalizes URLs
   - Crawls websites with rate limiting
   - Extracts meaningful content while removing unwanted elements
   - Supports same-domain link following

2. **Text Processor** (`services/text_processor.py`)
   - Cleans and normalizes extracted text
   - Creates semantic chunks with configurable size and overlap
   - Maintains metadata for each chunk (source URL, title, etc.)

3. **Embedding Service** (`services/embedding_service.py`)
   - Uses sentence-transformers for generating embeddings
   - Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
   - Efficient batch processing for multiple texts

4. **Vector Database** (`services/vector_db.py`)
   - ChromaDB for persistent vector storage
   - Efficient similarity search with configurable parameters
   - Supports metadata filtering and source-based operations

5. **RAG Service** (`services/rag_service.py`)
   - Combines retrieval and generation
   - Uses similarity search to find relevant content
   - Generates responses using retrieved context

6. **LLM Service** (`services/llm_service.py`)
   - Simple rule-based QA system for reliable responses
   - Designed to prevent hallucinations by only using provided context
   - Fallback to "not available" when answer not found in content

7. **Conversation Memory** (`services/conversation_memory.py`)
   - Session-based memory management
   - Maintains conversation context within sessions
   - Automatic cleanup of old sessions

## 🛠️ Technology Stack

### Frameworks and Libraries
- **Streamlit**: Web interface framework
- **ChromaDB**: Vector database for embeddings storage
- **Sentence Transformers**: Text embedding generation
- **Beautiful Soup**: HTML parsing and content extraction
- **Requests**: HTTP client for web crawling

### Models Used
- **Embedding Model**: `all-MiniLM-L6-v2` by sentence-transformers
  - 384-dimensional embeddings
  - Good balance of quality and performance
  - Optimized for semantic similarity tasks

- **LLM Model**: Simple Rule-based QA System
  - Designed to prevent hallucinations
  - Uses keyword matching and sentence extraction
  - Reliable fallback to "not available" responses
  - Can be easily replaced with more advanced models (GPT, BERT, etc.)

### Vector Database Choice: ChromaDB
- **Why ChromaDB**: 
  - Simple setup and maintenance
  - Excellent Python integration
  - Built-in persistence
  - Good performance for small to medium datasets
  - No external dependencies

## 📦 Installation and Setup

### Prerequisites
- Python 3.8 or higher
- 2GB+ RAM (for embedding models)
- Internet connection for model downloads

### Step 1: Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd website_chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration (Optional)

Copy and modify environment variables:

```bash
cp .env.example .env
```

Edit `.env` file to customize:
- Embedding model settings
- Chunk size and overlap parameters
- Vector database configuration
- Crawling limits

### Step 3: Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 💻 Usage Guide

### 1. Index a Website
1. Open the Streamlit interface
2. Enter a website URL in the "Index Website Content" section
3. Click "Index Website" to start crawling and processing
4. Wait for the indexing to complete (processing time depends on website size)

### 2. Ask Questions
1. Once indexing is complete, use the chat interface
2. Ask questions about the website content
3. The chatbot will provide answers based only on the indexed content
4. View response confidence scores and sources

### 3. Manage Database
- **Clear Database**: Remove all indexed content
- **View Statistics**: See number of indexed chunks and active sessions
- **New Chat**: Start a fresh conversation session

## 🔧 Configuration Options

Key configuration parameters in `config/settings.py`:

```python
# Text Processing
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# Embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval
TOP_K_RESULTS = 5          # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score

# Crawling
MAX_PAGES_TO_CRAWL = 50    # Maximum pages per website
CRAWL_TIMEOUT = 30         # Timeout per request (seconds)
```

## 🚀 Embedding Strategy

### Model Selection
- **Primary**: `all-MiniLM-L6-v2`
  - Fast inference
  - Good semantic understanding
  - 384-dimensional embeddings
  - Suitable for diverse content types

### Chunking Strategy
- **Semantic chunking**: Splits on paragraph boundaries
- **Overlap**: 200 characters to maintain context
- **Size**: 1000 characters for optimal balance
- **Metadata**: Preserves source URL and page title

### Vector Storage
- **Distance Metric**: Cosine similarity (default in ChromaDB)
- **Indexing**: Automatic HNSW indexing for fast retrieval
- **Persistence**: All embeddings stored persistently

## ⚠️ Limitations

1. **Content Types**: Only supports HTML web pages (no PDFs, images, videos)
2. **Language**: Optimized for English content
3. **Scale**: Suitable for small to medium websites (< 10,000 pages)
4. **LLM Capabilities**: Simple rule-based responses may miss complex reasoning
5. **Crawling**: Respects robots.txt but may not handle all dynamic content

## 🔮 Future Improvements

### Short Term
- [ ] Support for PDF and document crawling
- [ ] Advanced LLM integration (GPT-4, Claude, Llama)
- [ ] Multi-language support
- [ ] Better handling of dynamic content (JavaScript rendering)

### Medium Term  
- [ ] User authentication and multi-tenancy
- [ ] Advanced conversation management
- [ ] Integration with external APIs
- [ ] Batch processing for multiple URLs

### Long Term
- [ ] Fine-tuned domain-specific models
- [ ] Advanced reasoning capabilities
- [ ] Integration with knowledge graphs
- [ ] Real-time website monitoring and updates

## 🛡️ Assumptions and Design Decisions

### Assumptions
1. **Static Content**: Websites contain primarily static HTML content
2. **English Language**: Primary language is English
3. **Resource Constraints**: System runs on limited computational resources
4. **Trust**: Users provide legitimate website URLs for indexing

### Design Decisions
1. **Simple LLM**: Used rule-based approach to prevent hallucinations
2. **ChromaDB**: Chosen for simplicity over more complex solutions like Pinecone
3. **Sentence Transformers**: Balanced performance and quality for embeddings
4. **Streamlit**: Rapid prototyping and deployment over complex web frameworks
5. **Session-based Memory**: Temporary memory to avoid privacy concerns

## 🔒 Security Considerations

- URL validation prevents malicious inputs
- No persistent user data storage
- Content sanitization during crawling
- Rate limiting to prevent abuse
- No external API keys required for basic operation

## 📊 Performance Characteristics

### Typical Performance
- **Indexing**: ~2-5 seconds per page
- **Query Response**: <2 seconds
- **Memory Usage**: ~1-2GB for embedding models
- **Storage**: ~10MB per 1000 chunks

### Scalability
- **Websites**: Up to 50 pages per website (configurable)
- **Concurrent Users**: Supports multiple simultaneous users
- **Database Size**: Efficient up to ~100,000 chunks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers** for excellent embedding models
- **ChromaDB** for user-friendly vector database
- **Streamlit** for rapid web app development
- **Beautiful Soup** for robust HTML parsing

---

## 🆘 Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```
   Solution: Ensure stable internet connection and sufficient disk space
   ```

2. **Website Not Indexing**
   ```
   Check: URL is valid and accessible
   Check: Website contains meaningful HTML content
   Check: Not blocked by robots.txt
   ```

3. **Memory Issues**
   ```
   Reduce: CHUNK_SIZE and MAX_PAGES_TO_CRAWL in config
   ```

4. **No Responses to Questions**
   ```
   Check: Website is properly indexed
   Lower: SIMILARITY_THRESHOLD in config
   ```

### Getting Help
- Check the logs in the Streamlit interface
- Review system status in the sidebar
- Verify configuration parameters
- Ensure all dependencies are properly installed

---

**Built with ❤️ using AI and open-source technologies**
