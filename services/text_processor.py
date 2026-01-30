"""
Text processing service for cleaning and chunking website content.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of processed text with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    word_count: int

class TextProcessor:
    """
    Processes raw text content into clean, semantically meaningful chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patterns for text normalization
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']+')
        self.multiple_periods_pattern = re.compile(r'\.{3,}')
        self.multiple_dashes_pattern = re.compile(r'-{3,}')
        
    def process_crawled_pages(self, crawled_pages: List) -> List[TextChunk]:
        """
        Process multiple crawled pages into text chunks.
        
        Args:
            crawled_pages: List of CrawledPage objects
            
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        
        for page in crawled_pages:
            try:
                # Clean the content
                cleaned_content = self.clean_text(page.content)
                
                if not cleaned_content.strip():
                    logger.warning(f"No content after cleaning for {page.url}")
                    continue
                
                # Create chunks
                chunks = self.create_chunks(
                    cleaned_content, 
                    {
                        'source_url': page.url,
                        'page_title': page.title,
                        'original_word_count': page.word_count
                    }
                )
                
                all_chunks.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {page.url}")
                
            except Exception as e:
                logger.error(f"Error processing page {page.url}: {str(e)}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove special characters but keep punctuation
        text = self.special_chars_pattern.sub(' ', text)
        
        # Normalize multiple periods and dashes
        text = self.multiple_periods_pattern.sub('...', text)
        text = self.multiple_dashes_pattern.sub('---', text)
        
        # Remove common web artifacts
        web_artifacts = [
            r'\\n', r'\\t', r'\\r',  # Escaped newlines/tabs
            r'\[.*?\]',  # Content in square brackets (often metadata)
            r'\(.*?\)',  # Content in parentheses if too long
        ]
        
        for pattern in web_artifacts:
            text = re.sub(pattern, ' ', text)
        
        # Clean up sentences
        text = self._clean_sentences(text)
        
        # Final cleanup
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def _clean_sentences(self, text: str) -> str:
        """Clean individual sentences and remove incomplete ones."""
        sentences = re.split(r'[.!?]+', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short or incomplete sentences
            if len(sentence) < 10:
                continue
                
            # Skip sentences with too many numbers (likely metadata)
            if len(re.findall(r'\d', sentence)) > len(sentence.split()) * 0.3:
                continue
            
            # Skip sentences that are mostly uppercase (likely headers)
            if len([c for c in sentence if c.isupper()]) > len(sentence) * 0.7:
                continue
            
            cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences) + '.' if cleaned_sentences else ""
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Create semantic chunks from cleaned text.
        
        Args:
            text: Cleaned text content
            metadata: Metadata for the text (URL, title, etc.)
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        # First, try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            paragraphs = [text]
        
        chunks = []
        current_chunk = ""
        chunk_counter = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk.strip():
                    chunk = self._create_text_chunk(
                        current_chunk.strip(), 
                        metadata, 
                        chunk_counter
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_text(current_chunk) + paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = self._create_text_chunk(
                current_chunk.strip(), 
                metadata, 
                chunk_counter
            )
            chunks.append(chunk)
        
        # If no chunks were created but we have text, create one chunk
        if not chunks and text.strip():
            chunk = self._create_text_chunk(text.strip(), metadata, 0)
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get the last part of the text for chunk overlap.
        
        Args:
            text: The text to extract overlap from
            
        Returns:
            Overlap text
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a good sentence boundary for overlap
        overlap_text = text[-self.chunk_overlap:]
        
        # Find the first sentence boundary after the start
        sentence_end = re.search(r'[.!?]\s+', overlap_text)
        if sentence_end:
            return overlap_text[sentence_end.end():]
        
        # If no sentence boundary, find a word boundary
        space_idx = overlap_text.find(' ')
        if space_idx > 0:
            return overlap_text[space_idx + 1:]
        
        return overlap_text
    
    def _create_text_chunk(self, content: str, metadata: Dict[str, Any], chunk_index: int) -> TextChunk:
        """
        Create a TextChunk object with proper metadata.
        
        Args:
            content: The chunk content
            metadata: Original metadata
            chunk_index: Index of this chunk
            
        Returns:
            TextChunk object
        """
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_index': chunk_index,
            'chunk_length': len(content),
            'chunk_word_count': len(content.split())
        })
        
        # Generate a unique chunk ID
        chunk_id = f"{hash(metadata.get('source_url', ''))}_chunk_{chunk_index}"
        
        word_count = len(content.split())
        
        return TextChunk(
            content=content,
            metadata=chunk_metadata,
            chunk_id=chunk_id,
            word_count=word_count
        )
    
    def get_chunk_summary(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Get a summary of the chunks created.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Summary dictionary
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_words': 0,
                'average_chunk_size': 0,
                'sources': []
            }
        
        total_words = sum(chunk.word_count for chunk in chunks)
        sources = list(set(chunk.metadata.get('source_url', 'Unknown') for chunk in chunks))
        
        return {
            'total_chunks': len(chunks),
            'total_words': total_words,
            'average_chunk_size': total_words // len(chunks),
            'average_chunk_length': sum(len(chunk.content) for chunk in chunks) // len(chunks),
            'sources': sources,
            'source_count': len(sources)
        }
