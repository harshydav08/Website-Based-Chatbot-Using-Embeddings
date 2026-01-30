"""
Website Chatbot - AI-powered chatbot for website content using embeddings.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Website-based chatbot using embeddings and RAG"

from .core import WebsiteChatbot
from .config import config

__all__ = ["WebsiteChatbot", "config"]
