import json
from typing import List, Dict
from datetime import datetime, timezone
import redis
from app.config import settings

class RedisMemoryService:
    """Service for managing conversation memory in Redis."""
    
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            username="default",
            decode_responses=True
        )
        self.ttl = settings.REDIS_TTL
    
    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"session:{session_id}"
    
    def _get_booking_state_key(self, session_id: str) -> str:
        """Generate Redis key for booking state."""
        return f"booking_state:{session_id}"
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        key = self._get_session_key(session_id)
        self.client.rpush(key, json.dumps(message))
        self.client.expire(key, self.ttl)
    
    def get_history(
        self, 
        session_id: str, 
        limit: int = 100
    ) -> List[Dict]:
        """
        Retrieve conversation history.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of messages
        """
        key = self._get_session_key(session_id)
        raw_messages = self.client.lrange(key, 0, -1)
        
        messages = [json.loads(msg) for msg in raw_messages]
        
        if limit:
            return messages[-limit:]
        return messages
    
    def get_conversation_context(
        self, 
        session_id: str, 
        max_turns: int = 6
    ) -> List[Dict]:
        """
        Get recent conversation context for LLM.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of conversation turns
            
        Returns:
            List of messages in format suitable for LLM
        """
        history = self.get_history(session_id, limit=max_turns * 2)
        context = []
        
        for msg in history:
            context.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return context
    
    def clear_session(self, session_id: str):
        """
        Clear all data for a session.
        
        Args:
            session_id: Session identifier
        """
        self.client.delete(self._get_session_key(session_id))
        self.client.delete(self._get_booking_state_key(session_id))
    
    def set_booking_state(self, session_id: str, state: Dict):
        """
        Save booking state for a session.
        
        Args:
            session_id: Session identifier
            state: Booking state dictionary
        """
        key = self._get_booking_state_key(session_id)
        self.client.set(key, json.dumps(state), ex=self.ttl)
    
    def get_booking_state(self, session_id: str) -> Dict:
        """
        Retrieve booking state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Booking state dictionary or empty dict
        """
        key = self._get_booking_state_key(session_id)
        data = self.client.get(key)
        
        if data:
            return json.loads(data)
        return {}
    
    def clear_booking_state(self, session_id: str):
        """
        Clear booking state for a session.
        
        Args:
            session_id: Session identifier
        """
        self.client.delete(self._get_booking_state_key(session_id))
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        return self.client.exists(self._get_session_key(session_id)) > 0

# Singleton instance
redis_memory_service = RedisMemoryService()