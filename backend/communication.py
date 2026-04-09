# backend/communication.py - Agent Communication Utilities
# Provides utilities for inter-agent communication, message routing, and coordination

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# MESSAGE TYPES & ENUMS
# ============================================================================

class MessageType(Enum):
    """Message types for agent communication."""
    QUERY = "query"
    RESPONSE = "response"
    COMMAND = "command"
    ALERT = "alert"
    STATUS = "status"
    REFLECTION = "reflection"
    COORDINATION = "coordination"
    ETHICAL_REVIEW = "ethical_review"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    BROADCAST = "broadcast"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageStatus(Enum):
    """Message delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# MESSAGE DATA STRUCTURES
# ============================================================================

@dataclass
class Message:
    """Represents a message between agents."""
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = None
    status: MessageStatus = MessageStatus.PENDING
    response: Optional['Message'] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['response'] = asdict(self.response) if self.response else None
        return data
    
    def get_delivery_time(self) -> Optional[float]:
        """Get delivery time in milliseconds."""
        if self.status == MessageStatus.DELIVERED and 'delivered_at' in self.metadata:
            return (self.metadata['delivered_at'] - self.timestamp) * 1000
        return None


@dataclass
class MessageBatch:
    """Represents a batch of messages."""
    id: str
    messages: List[Message]
    timestamp: float = None
    status: MessageStatus = MessageStatus.PENDING
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def get_success_count(self) -> int:
        """Get count of successfully delivered messages."""
        return sum(1 for m in self.messages if m.status == MessageStatus.DELIVERED)
    
    def get_failure_count(self) -> int:
        """Get count of failed messages."""
        return sum(1 for m in self.messages if m.status == MessageStatus.FAILED)


# ============================================================================
# AGENT COMMUNICATION MANAGER
# ============================================================================

class AgentCommunicationManager:
    """Manages communication between agents."""
    
    def __init__(self, archive_path: str = "Shadow/manus_archive"):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        self.message_history: Dict[str, List[Message]] = {}
        self.message_queue: Dict[str, asyncio.Queue] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.message_cache: Dict[str, Message] = {}
        self.delivery_log_path = self.archive_path / "message_delivery.json"
        
        self._load_delivery_log()
    
    def _load_delivery_log(self):
        """Load message delivery log from file."""
        if self.delivery_log_path.exists():
            try:
                with open(self.delivery_log_path, 'r') as f:
                    self.delivery_log = json.load(f)
            except:
                self.delivery_log = []
        else:
            self.delivery_log = []
    
    def _save_delivery_log(self):
        """Save message delivery log to file."""
        with open(self.delivery_log_path, 'w') as f:
            json.dump(self.delivery_log, f, indent=2)
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        timeout: float = 5.0
    ) -> Message:
        """
        Send a message from one agent to another.
        
        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            message_type: Type of message
            content: Message content
            priority: Message priority
            timeout: Delivery timeout in seconds
        
        Returns:
            Message object with delivery status
        """
        message_id = f"msg_{from_agent}_{to_agent}_{int(time.time()*1000)}"
        message = Message(
            id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        # Add to history
        if from_agent not in self.message_history:
            self.message_history[from_agent] = []
        self.message_history[from_agent].append(message)
        
        # Simulate delivery
        try:
            message.status = MessageStatus.SENT
            await asyncio.sleep(0.01)  # Simulate network delay
            message.status = MessageStatus.DELIVERED
            message.metadata['delivered_at'] = time.time()
            
            # Log delivery
            self.delivery_log.append({
                'message_id': message_id,
                'from': from_agent,
                'to': to_agent,
                'type': message_type.value,
                'priority': priority.value,
                'status': message.status.value,
                'timestamp': datetime.utcnow().isoformat(),
                'delivery_time_ms': message.get_delivery_time()
            })
            self._save_delivery_log()
            
            logger.info(f"Message {message_id} delivered: {from_agent} → {to_agent}")
            
        except asyncio.TimeoutError:
            message.status = MessageStatus.TIMEOUT
            logger.warning(f"Message {message_id} timed out")
        except Exception as e:
            message.status = MessageStatus.FAILED
            message.metadata['error'] = str(e)
            logger.error(f"Message {message_id} failed: {e}")
        
        return message
    
    async def broadcast_message(
        self,
        from_agent: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        agent_list: Optional[List[str]] = None
    ) -> MessageBatch:
        """
        Send a message to multiple agents.
        
        Args:
            from_agent: Sender agent ID
            message_type: Type of message
            content: Message content
            priority: Message priority
            agent_list: List of recipient agent IDs (None = all agents)
        
        Returns:
            MessageBatch with delivery status for all recipients
        """
        if agent_list is None:
            agent_list = await self._get_all_agents()
        
        batch_id = f"batch_{from_agent}_{int(time.time()*1000)}"
        messages = []
        
        tasks = [
            self.send_message(from_agent, agent, message_type, content, priority)
            for agent in agent_list
        ]
        
        messages = await asyncio.gather(*tasks)
        
        batch = MessageBatch(
            id=batch_id,
            messages=messages,
            status=MessageStatus.DELIVERED
        )
        
        logger.info(f"Broadcast {batch_id}: {batch.get_success_count()}/{len(messages)} delivered")
        
        return batch
    
    async def send_message_sequence(
        self,
        sequence: List[tuple],
        timeout: float = 30.0
    ) -> List[Message]:
        """
        Send messages in sequence, waiting for each to complete.
        
        Args:
            sequence: List of (from_agent, to_agent, message_type, content) tuples
            timeout: Total timeout for sequence
        
        Returns:
            List of Message objects
        """
        results = []
        start_time = time.time()
        
        for from_agent, to_agent, msg_type, content in sequence:
            if time.time() - start_time > timeout:
                logger.warning("Message sequence timeout")
                break
            
            message = await self.send_message(
                from_agent, to_agent, msg_type, content
            )
            results.append(message)
        
        return results
    
    async def send_message_parallel(
        self,
        messages: List[tuple],
        timeout: float = 30.0
    ) -> List[Message]:
        """
        Send multiple messages in parallel.
        
        Args:
            messages: List of (from_agent, to_agent, message_type, content) tuples
            timeout: Total timeout for all messages
        
        Returns:
            List of Message objects
        """
        tasks = [
            self.send_message(from_agent, to_agent, msg_type, content)
            for from_agent, to_agent, msg_type, content in messages
        ]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=timeout
            )
            return results
        except asyncio.TimeoutError:
            logger.error("Parallel message send timeout")
            return []
    
    def register_message_handler(
        self,
        agent_id: str,
        handler: Callable
    ):
        """Register a handler for incoming messages."""
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = []
        self.message_handlers[agent_id].append(handler)
    
    async def get_message_history(
        self,
        agent_id: str,
        limit: int = 50
    ) -> List[Message]:
        """Get message history for an agent."""
        if agent_id not in self.message_history:
            return []
        return self.message_history[agent_id][-limit:]
    
    def get_message_statistics(self) -> Dict[str, Any]:
        """Get message statistics."""
        total_messages = sum(len(msgs) for msgs in self.message_history.values())
        
        status_counts = {
            'delivered': sum(
                1 for msgs in self.message_history.values()
                for msg in msgs if msg.status == MessageStatus.DELIVERED
            ),
            'failed': sum(
                1 for msgs in self.message_history.values()
                for msg in msgs if msg.status == MessageStatus.FAILED
            ),
            'timeout': sum(
                1 for msgs in self.message_history.values()
                for msg in msgs if msg.status == MessageStatus.TIMEOUT
            )
        }
        
        return {
            'total_messages': total_messages,
            'status_counts': status_counts,
            'delivery_rate': status_counts['delivered'] / total_messages if total_messages > 0 else 0,
            'agents_communicating': len(self.message_history)
        }
    
    async def _get_all_agents(self) -> List[str]:
        """Get list of all agents."""
        # This would be implemented to get actual agent list
        return [
            "kael", "lumina", "vega", "gemini", "agni", "kavach",
            "sanghacore", "shadow", "echo", "phoenix", "oracle",
            "claude", "manus", "memoryroot"
        ]


# ============================================================================
# AGENT CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manages multi-turn conversations between agents."""
    
    def __init__(self):
        self.conversations: Dict[str, List[Message]] = {}
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def start_conversation(
        self,
        participants: List[str],
        topic: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Start a new conversation."""
        conv_id = f"conv_{int(time.time()*1000)}"
        self.conversations[conv_id] = []
        self.conversation_metadata[conv_id] = {
            'participants': participants,
            'topic': topic,
            'context': context or {},
            'started_at': datetime.utcnow().isoformat(),
            'message_count': 0
        }
        return conv_id
    
    async def add_to_conversation(
        self,
        conv_id: str,
        message: Message
    ) -> bool:
        """Add a message to a conversation."""
        if conv_id not in self.conversations:
            return False
        
        self.conversations[conv_id].append(message)
        self.conversation_metadata[conv_id]['message_count'] += 1
        return True
    
    async def get_conversation(self, conv_id: str) -> List[Message]:
        """Get all messages in a conversation."""
        return self.conversations.get(conv_id, [])
    
    async def end_conversation(self, conv_id: str) -> Dict[str, Any]:
        """End a conversation and return summary."""
        if conv_id not in self.conversations:
            return {}
        
        metadata = self.conversation_metadata[conv_id]
        metadata['ended_at'] = datetime.utcnow().isoformat()
        
        return {
            'conversation_id': conv_id,
            'metadata': metadata,
            'message_count': len(self.conversations[conv_id])
        }


# ============================================================================
# GLOBAL COMMUNICATION MANAGER INSTANCE
# ============================================================================

_communication_manager: Optional[AgentCommunicationManager] = None

def get_communication_manager() -> AgentCommunicationManager:
    """Get or create the global communication manager."""
    global _communication_manager
    if _communication_manager is None:
        _communication_manager = AgentCommunicationManager()
    return _communication_manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def send_message(
    from_agent: str,
    to_agent: str,
    message_type: str,
    content: Dict[str, Any],
    priority: str = "normal"
) -> Message:
    """Convenience function to send a message."""
    manager = get_communication_manager()
    msg_type = MessageType[message_type.upper()] if isinstance(message_type, str) else message_type
    msg_priority = MessagePriority[priority.upper()] if isinstance(priority, str) else priority
    
    return await manager.send_message(
        from_agent, to_agent, msg_type, content, msg_priority
    )


async def broadcast_message(
    from_agent: str,
    message_type: str,
    content: Dict[str, Any],
    priority: str = "normal"
) -> MessageBatch:
    """Convenience function to broadcast a message."""
    manager = get_communication_manager()
    msg_type = MessageType[message_type.upper()] if isinstance(message_type, str) else message_type
    msg_priority = MessagePriority[priority.upper()] if isinstance(priority, str) else priority
    
    return await manager.broadcast_message(
        from_agent, msg_type, content, msg_priority
    )


async def get_message_history(agent_id: str, limit: int = 50) -> List[Message]:
    """Get message history for an agent."""
    manager = get_communication_manager()
    return await manager.get_message_history(agent_id, limit)


def get_message_statistics() -> Dict[str, Any]:
    """Get communication statistics."""
    manager = get_communication_manager()
    return manager.get_message_statistics()
