"""
memory.py
Conversation memory management.
"""
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.messages: List[Dict] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str, metadata: Dict = None) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata dictionary
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        # Keep only the last max_history messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        logger.info(f"Added {role} message to history (total: {len(self.messages)})")
    
    def get_history(self, num_messages: int = None) -> List[Dict]:
        """
        Get the conversation history.
        
        Args:
            num_messages: Number of recent messages to return (None for all)
            
        Returns:
            List of message dictionaries
        """
        if num_messages is None:
            return self.messages
        
        return self.messages[-num_messages:]
    
    def get_context(self, num_messages: int = 4) -> List[Dict]:
        """
        Get recent messages formatted for LLM context.
        
        Args:
            num_messages: Number of recent messages to include
            
        Returns:
            List of recent messages for context
        """
        recent = self.get_history(num_messages)
        return [{"role": msg["role"], "content": msg["content"]} for msg in recent]
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        logger.info("Cleared conversation history")
    
    def get_summary(self) -> Dict:
        """
        Get summary of current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        user_messages = [m for m in self.messages if m["role"] == "user"]
        assistant_messages = [m for m in self.messages if m["role"] == "assistant"]
        
        return {
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "first_message": self.messages[0]["timestamp"] if self.messages else None,
            "last_message": self.messages[-1]["timestamp"] if self.messages else None
        }
    
    def search(self, keyword: str) -> List[Dict]:
        """
        Search for messages containing a keyword.
        
        Args:
            keyword: Search keyword
            
        Returns:
            List of matching messages
        """
        results = []
        keyword_lower = keyword.lower()
        
        for message in self.messages:
            if keyword_lower in message["content"].lower():
                results.append(message)
        
        logger.info(f"Found {len(results)} messages matching '{keyword}'")
        return results
