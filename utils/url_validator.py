"""
URL validation utilities for the Website Chatbot.
"""

import validators
import requests
from typing import Tuple, Optional
from urllib.parse import urlparse, urljoin
import logging

logger = logging.getLogger(__name__)

class URLValidator:
    """Validates and normalizes URLs for web crawling."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate a URL and return validation status, normalized URL, and error message.
        
        Args:
            url: The URL to validate
            
        Returns:
            Tuple of (is_valid, normalized_url, error_message)
        """
        if not url or not url.strip():
            return False, None, "URL cannot be empty"
        
        url = url.strip()
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Validate URL format
        if not validators.url(url):
            return False, None, "Invalid URL format"
        
        # Parse URL to check components
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False, None, "Invalid URL: missing domain"
        except Exception as e:
            return False, None, f"Invalid URL: {str(e)}"
        
        # Test if URL is reachable
        try:
            response = requests.head(url, timeout=self.timeout, allow_redirects=True)
            if response.status_code >= 400:
                return False, None, f"URL not reachable: HTTP {response.status_code}"
            
            # Check if it's an HTML page
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return False, None, "URL does not point to an HTML page"
                
        except requests.exceptions.Timeout:
            return False, None, "URL is not reachable: connection timeout"
        except requests.exceptions.ConnectionError:
            return False, None, "URL is not reachable: connection failed"
        except requests.exceptions.RequestException as e:
            return False, None, f"URL is not reachable: {str(e)}"
        
        return True, url, None
    
    def normalize_url(self, url: str, base_url: str = None) -> str:
        """
        Normalize a URL by resolving relative paths and ensuring proper format.
        
        Args:
            url: The URL to normalize
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Normalized URL
        """
        if base_url:
            return urljoin(base_url, url)
        return url
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs belong to the same domain.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            True if same domain, False otherwise
        """
        try:
            domain1 = urlparse(url1).netloc.lower()
            domain2 = urlparse(url2).netloc.lower()
            return domain1 == domain2
        except Exception:
            return False
