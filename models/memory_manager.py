import redis
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from config import settings
import hashlib

class ConversationMemory:
    """Redis-based short-term conversation memory"""
    
    def __init__(self, host: str = None, port: int = None, db: int = None):
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db or settings.REDIS_DB
        self.ttl = settings.REDIS_TTL
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,  # We'll handle encoding/decoding
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            print(f"⚠️  Redis not available: {e}. Using in-memory fallback.")
            self.redis_client = None
            self._memory_store = {}
    
    def _get_key(self, user_id: str, key_type: str = "conversation") -> str:
        """Generate Redis key for user"""
        # Create hash for consistent key length
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
        return f"beauty_bot:{user_hash}:{key_type}"
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        try:
            return pickle.dumps(data)
        except:
            # Fallback to JSON for simple types
            return json.dumps(data, ensure_ascii=False).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis"""
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except:
            # Try JSON fallback
            try:
                return json.loads(data.decode('utf-8'))
            except:
                return data.decode('utf-8') if isinstance(data, bytes) else data
    
    def add_turn(self, user_id: str, role: str, message: str, metadata: Dict = None):
        """Add a conversation turn to memory"""
        key = self._get_key(user_id)
        
        turn = {
            "role": role,  # "user" or "bot"
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if self.redis_client:
            try:
                # Push to list
                self.redis_client.rpush(key, self._serialize(turn))
                # Set TTL
                self.redis_client.expire(key, self.ttl)
                # Trim to last 10 turns (5 user + 5 bot)
                if self.redis_client.llen(key) > 10:
                    self.redis_client.ltrim(key, -10, -1)
            except redis.RedisError as e:
                print(f"Redis error: {e}. Using in-memory store.")
                self._memory_fallback(key, turn)
        else:
            self._memory_fallback(key, turn)
    
    def _memory_fallback(self, key: str, turn: Dict):
        """Fallback to in-memory storage"""
        if not hasattr(self, '_memory_store'):
            self._memory_store = {}
        
        if key not in self._memory_store:
            self._memory_store[key] = []
        
        self._memory_store[key].append(turn)
        
        # Trim to last 10 turns
        if len(self._memory_store[key]) > 10:
            self._memory_store[key] = self._memory_store[key][-10:]
    
    def get_context(self, user_id: str, max_turns: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        key = self._get_key(user_id)
        
        if self.redis_client:
            try:
                if not self.redis_client.exists(key):
                    return []
                
                # Get last N turns
                turns_data = self.redis_client.lrange(key, -max_turns, -1)
                turns = [self._deserialize(turn) for turn in turns_data]
                return turns
            except redis.RedisError:
                # Fallback to in-memory
                pass
        
        # In-memory fallback
        if hasattr(self, '_memory_store') and key in self._memory_store:
            return self._memory_store[key][-max_turns:]
        
        return []
    
    def get_last_user_message(self, user_id: str) -> Optional[str]:
        """Get the last message from user"""
        context = self.get_context(user_id, max_turns=10)
        
        # Find last user message
        for turn in reversed(context):
            if turn.get("role") == "user":
                return turn.get("message", "")
        
        return None
    
    def get_last_intent(self, user_id: str) -> Optional[str]:
        """Get the last detected intent"""
        context = self.get_context(user_id, max_turns=10)
        
        # Find last bot message with intent metadata
        for turn in reversed(context):
            if turn.get("role") == "bot":
                metadata = turn.get("metadata", {})
                if "intent" in metadata:
                    return metadata["intent"]
        
        return None
    
    def clear(self, user_id: str):
        """Clear conversation memory for user"""
        key = self._get_key(user_id)
        
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except redis.RedisError:
                pass
        
        # Clear in-memory store
        if hasattr(self, '_memory_store') and key in self._memory_store:
            del self._memory_store[key]
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        context = self.get_context(user_id, max_turns=20)
        
        if not context:
            return {
                "turns_count": 0,
                "last_activity": None,
                "user_messages": 0,
                "bot_messages": 0,
                "last_intent": None
            }
        
        # Count messages by role
        user_messages = len([t for t in context if t.get("role") == "user"])
        bot_messages = len([t for t in context if t.get("role") == "bot"])
        
        # Get last intent
        last_intent = None
        for turn in reversed(context):
            if turn.get("role") == "bot":
                metadata = turn.get("metadata", {})
                if "intent" in metadata:
                    last_intent = metadata["intent"]
                    break
        
        return {
            "turns_count": len(context),
            "last_activity": context[-1]["timestamp"] if context else None,
            "user_messages": user_messages,
            "bot_messages": bot_messages,
            "last_intent": last_intent
        }
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if self.redis_client:
            try:
                self.redis_client.ping()
                return True
            except:
                return False
        return False

# Create singleton instance
memory_manager = ConversationMemory()