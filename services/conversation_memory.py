"""
Conversational memory system for maintaining context across questions in a session.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConversationSession:
    """Represents a conversation session with its messages."""
    session_id: str
    messages: List[Message]
    created_at: str
    last_activity: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the session to a dictionary."""
        return {
            'session_id': self.session_id,
            'messages': [asdict(msg) for msg in self.messages],
            'created_at': self.created_at,
            'last_activity': self.last_activity
        }

class ConversationMemory:
    """
    Manages conversational memory for chat sessions.
    Memory is session-based and resets when the session ends.
    """
    
    def __init__(self, max_messages_per_session: int = 20, max_context_messages: int = 6):
        """
        Initialize the conversation memory.
        
        Args:
            max_messages_per_session: Maximum number of messages to keep per session
            max_context_messages: Maximum number of recent messages to use for context
        """
        self.max_messages_per_session = max_messages_per_session
        self.max_context_messages = max_context_messages
        self.sessions: Dict[str, ConversationSession] = {}
        
        logger.info(f"Conversation memory initialized (max_messages: {max_messages_per_session}, context: {max_context_messages})")
    
    def create_session(self) -> str:
        """
        Create a new conversation session.
        
        Returns:
            New session ID
        """
        session_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        session = ConversationSession(
            session_id=session_id,
            messages=[],
            created_at=current_time,
            last_activity=current_time
        )
        
        self.sessions[session_id] = session
        
        logger.info(f"Created new conversation session: {session_id}")
        return session_id
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to a conversation session.
        
        Args:
            session_id: ID of the session
            role: Role of the message sender ("user" or "assistant")
            content: Content of the message
            metadata: Optional metadata for the message
            
        Returns:
            True if message was added successfully, False otherwise
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return False
        
        session = self.sessions[session_id]
        current_time = datetime.now().isoformat()
        
        message = Message(
            role=role,
            content=content,
            timestamp=current_time,
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        session.last_activity = current_time
        
        # Trim messages if we exceed the limit
        if len(session.messages) > self.max_messages_per_session:
            # Keep the most recent messages
            session.messages = session.messages[-self.max_messages_per_session:]
            logger.debug(f"Trimmed session {session_id} to {self.max_messages_per_session} messages")
        
        logger.debug(f"Added {role} message to session {session_id}")
        return True
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of messages in dictionary format
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return []
        
        session = self.sessions[session_id]
        return [asdict(msg) for msg in session.messages]
    
    def get_recent_context(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get recent messages for context in RAG queries.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of recent messages for context
        """
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        
        # Get the most recent messages for context
        recent_messages = session.messages[-self.max_context_messages:]
        
        return [asdict(msg) for msg in recent_messages]
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session.
        
        Args:
            session_id: ID of the session to clear
            
        Returns:
            True if session was cleared, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared conversation session: {session_id}")
            return True
        else:
            logger.warning(f"Session {session_id} not found for clearing")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a conversation session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session information or None if not found
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            'session_id': session.session_id,
            'message_count': len(session.messages),
            'created_at': session.created_at,
            'last_activity': session.last_activity
        }
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about all active sessions.
        
        Returns:
            List of session information dictionaries
        """
        return [
            {
                'session_id': session.session_id,
                'message_count': len(session.messages),
                'created_at': session.created_at,
                'last_activity': session.last_activity
            }
            for session in self.sessions.values()
        ]
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: ID of the session to check
            
        Returns:
            True if session exists, False otherwise
        """
        return session_id in self.sessions
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Clean up old sessions that haven't been active recently.
        
        Args:
            max_age_hours: Maximum age of sessions in hours
        """
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            last_activity = datetime.fromisoformat(session.last_activity)
            age_hours = (current_time - last_activity).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation memory.
        
        Returns:
            Dictionary with memory statistics
        """
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        
        return {
            'total_sessions': len(self.sessions),
            'total_messages': total_messages,
            'max_messages_per_session': self.max_messages_per_session,
            'max_context_messages': self.max_context_messages,
            'average_messages_per_session': total_messages / len(self.sessions) if self.sessions else 0
        }
