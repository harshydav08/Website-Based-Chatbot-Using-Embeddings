"""
Web crawler for extracting meaningful content from websites.
"""

import requests
from bs4 import BeautifulSoup, Comment
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import logging
import time
from dataclasses import dataclass
from utils.url_validator import URLValidator

logger = logging.getLogger(__name__)

@dataclass
class CrawledPage:
    """Represents a crawled web page with its content and metadata."""
    url: str
    title: str
    content: str
    word_count: int
    
class WebCrawler:
    """
    Web crawler that extracts meaningful textual content from websites.
    Removes headers, footers, navigation, and advertisements.
    """
    
    def __init__(self, max_pages: int = 50, timeout: int = 30, delay: float = 1.0):
        self.max_pages = max_pages
        self.timeout = timeout
        self.delay = delay
        self.url_validator = URLValidator(timeout=timeout)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Tags to remove (headers, footers, nav, ads, etc.)
        self.unwanted_tags = {
            'header', 'footer', 'nav', 'aside', 'advertisement', 'ad', 'sidebar',
            'menu', 'breadcrumb', 'social', 'share', 'comment', 'related',
            'widget', 'banner', 'popup', 'modal', 'overlay'
        }
        
        # Tags that typically contain unwanted content
        self.unwanted_classes = {
            'header', 'footer', 'nav', 'navigation', 'sidebar', 'aside', 'menu',
            'breadcrumb', 'breadcrumbs', 'social', 'share', 'sharing', 'comment',
            'comments', 'related', 'widget', 'advertisement', 'ad', 'ads',
            'banner', 'popup', 'modal', 'overlay', 'promo', 'promotion'
        }
        
        # Tags that typically contain unwanted content by ID
        self.unwanted_ids = {
            'header', 'footer', 'nav', 'navigation', 'sidebar', 'menu',
            'breadcrumb', 'social', 'share', 'comments', 'related', 'widget',
            'advertisement', 'ad', 'banner', 'popup', 'modal'
        }
    
    def crawl_website(self, start_url: str) -> List[CrawledPage]:
        """
        Crawl a website starting from the given URL.
        
        Args:
            start_url: The starting URL to crawl
            
        Returns:
            List of CrawledPage objects containing extracted content
        """
        logger.info(f"Starting to crawl website: {start_url}")
        
        # Validate the starting URL
        is_valid, normalized_url, error = self.url_validator.validate_url(start_url)
        if not is_valid:
            logger.error(f"Invalid starting URL: {error}")
            return []
        
        crawled_pages = []
        visited_urls = set()
        urls_to_visit = [normalized_url]
        
        while urls_to_visit and len(crawled_pages) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            
            try:
                page = self._crawl_page(current_url)
                if page and page.content.strip():
                    crawled_pages.append(page)
                    logger.info(f"Successfully crawled: {current_url} ({page.word_count} words)")
                    
                    # Find additional URLs to crawl (same domain only)
                    if len(crawled_pages) < self.max_pages:
                        additional_urls = self._find_links(current_url, normalized_url)
                        for url in additional_urls:
                            if url not in visited_urls and url not in urls_to_visit:
                                urls_to_visit.append(url)
                
                # Rate limiting
                time.sleep(self.delay)
                
            except Exception as e:
                logger.warning(f"Failed to crawl {current_url}: {str(e)}")
                continue
        
        logger.info(f"Crawling completed. Total pages: {len(crawled_pages)}")
        return crawled_pages
    
    def _crawl_page(self, url: str) -> Optional[CrawledPage]:
        """
        Crawl a single page and extract its content.
        
        Args:
            url: The URL to crawl
            
        Returns:
            CrawledPage object or None if crawling failed
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, url)
            
            # Extract and clean content
            content = self._extract_content(soup)
            
            if not content.strip():
                logger.warning(f"No meaningful content found on {url}")
                return None
            
            word_count = len(content.split())
            
            return CrawledPage(
                url=url,
                title=title,
                content=content,
                word_count=word_count
            )
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback to h1 tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        # Last resort: use URL
        return urlparse(url).path.split('/')[-1] or url
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract meaningful content from the page, removing unwanted elements.
        """
        # Remove unwanted elements
        self._remove_unwanted_elements(soup)
        
        # Try to find main content area
        main_content = self._find_main_content(soup)
        if main_content:
            content = self._extract_text_from_element(main_content)
        else:
            # Fallback to body content
            body = soup.find('body') or soup
            content = self._extract_text_from_element(body)
        
        # Clean and normalize text
        content = self._clean_text(content)
        
        return content
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements."""
        # Remove script and style tags
        for tag in soup(['script', 'style', 'link', 'meta']):
            tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove elements by tag name
        for tag_name in self.unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove elements by class
        for class_name in self.unwanted_classes:
            for tag in soup.find_all(class_=lambda x: x and class_name in ' '.join(x).lower()):
                tag.decompose()
        
        # Remove elements by ID
        for id_name in self.unwanted_ids:
            for tag in soup.find_all(id=lambda x: x and id_name in x.lower()):
                tag.decompose()
        
        # Remove elements with minimal content
        for tag in soup.find_all():
            text = tag.get_text(strip=True)
            if len(text) < 10 and tag.name not in ['img', 'br', 'hr']:
                if not any(child.name for child in tag.children if hasattr(child, 'name')):
                    tag.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup):
        """Try to identify the main content area of the page."""
        # Look for common main content selectors
        main_selectors = [
            'main',
            '[role=\"main\"]',
            '.main-content',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            'article',
            '.container .content',
            '#main-content',
            '#content'
        ]
        
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                return element
        
        # Look for the largest text block
        text_containers = soup.find_all(['div', 'section', 'article'])
        if text_containers:
            largest_container = max(text_containers, 
                                   key=lambda x: len(x.get_text(strip=True)))
            if len(largest_container.get_text(strip=True)) > 500:
                return largest_container
        
        return None
    
    def _extract_text_from_element(self, element) -> str:
        """Extract and structure text from an HTML element."""
        # Get text with some structure preservation
        texts = []
        
        for tag in element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
            text = tag.get_text(separator=' ', strip=True)
            if text and len(text) > 10:  # Only include substantial text
                texts.append(text)
        
        # If no structured text found, get all text
        if not texts:
            texts = [element.get_text(separator=' ', strip=True)]
        
        return '\n\n'.join(texts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Remove common unwanted patterns
        unwanted_patterns = [
            r'cookie.*policy',
            r'privacy.*policy',
            r'terms.*service',
            r'subscribe.*newsletter',
            r'follow.*us',
            r'share.*this',
            r'related.*articles?',
            r'you.*might.*like',
            r'advertisement',
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _find_links(self, current_url: str, base_url: str) -> List[str]:
        """Find additional links to crawl from the current page."""
        try:
            response = self.session.get(current_url, timeout=self.timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)
                
                # Only include same-domain links
                if self.url_validator.is_same_domain(full_url, base_url):
                    # Skip non-content links
                    if not self._is_content_link(href, link.get_text()):
                        continue
                    links.append(full_url)
            
            return links[:10]  # Limit number of links per page
            
        except Exception:
            return []
    
    def _is_content_link(self, href: str, link_text: str) -> bool:
        """Check if a link points to content worth crawling."""
        href = href.lower()
        link_text = link_text.lower() if link_text else ''
        
        # Skip certain file types and paths
        skip_patterns = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.zip', '.rar', '.tar', '.gz',
            '/search', '/login', '/register', '/contact', '/about/contact',
            '#', 'javascript:', 'mailto:', 'tel:',
            'download', 'subscribe', 'newsletter'
        ]
        
        for pattern in skip_patterns:
            if pattern in href or pattern in link_text:
                return False
        
        return True
