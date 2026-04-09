#!/usr/bin/env python3
"""
Test suite for communication module.

Tests message routing, delivery, broadcasting, and conversation management.
"""

import asyncio
import pytest
import sys

sys.path.insert(0, '/home/ubuntu/Helix')

from backend.communication import (
    Message, MessageType, MessagePriority, MessageStatus,
    AgentCommunicationManager, ConversationManager,
    send_message, broadcast_message
)


class TestMessage:
    """Test Message data structure."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(
            id="test_1",
            from_agent="kael",
            to_agent="lumina",
            message_type=MessageType.QUERY,
            content={"question": "Is this ethical?"},
            priority=MessagePriority.HIGH
        )
        
        assert msg.id == "test_1"
        assert msg.from_agent == "kael"
        assert msg.to_agent == "lumina"
        assert msg.message_type == MessageType.QUERY
        assert msg.status == MessageStatus.PENDING
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(
            id="test_1",
            from_agent="kael",
            to_agent="lumina",
            message_type=MessageType.QUERY,
            content={"question": "test"}
        )
        
        msg_dict = msg.to_dict()
        
        assert msg_dict['id'] == "test_1"
        assert msg_dict['message_type'] == "query"
        assert msg_dict['priority'] == 1  # NORMAL
    
    def test_message_delivery_time(self):
        """Test calculating delivery time."""
        msg = Message(
            id="test_1",
            from_agent="kael",
            to_agent="lumina",
            message_type=MessageType.QUERY,
            content={}
        )
        
        # Before delivery
        assert msg.get_delivery_time() is None
        
        # After delivery
        msg.status = MessageStatus.DELIVERED
        msg.metadata['delivered_at'] = msg.timestamp + 0.5
        
        delivery_time = msg.get_delivery_time()
        assert delivery_time is not None
        assert delivery_time > 0


class TestAgentCommunicationManager:
    """Test AgentCommunicationManager."""
    
    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending a message."""
        manager = AgentCommunicationManager()
        
        message = await manager.send_message(
            from_agent="kael",
            to_agent="lumina",
            message_type=MessageType.QUERY,
            content={"question": "test"}
        )
        
        assert message.status == MessageStatus.DELIVERED
        assert message.from_agent == "kael"
        assert message.to_agent == "lumina"
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting a message."""
        manager = AgentCommunicationManager()
        
        batch = await manager.broadcast_message(
            from_agent="system",
            message_type=MessageType.ALERT,
            content={"message": "test alert"},
            agent_list=["kael", "lumina", "grok"]
        )
        
        assert len(batch.messages) == 3
        assert batch.get_success_count() == 3
        assert batch.get_failure_count() == 0
    
    @pytest.mark.asyncio
    async def test_message_history(self):
        """Test retrieving message history."""
        manager = AgentCommunicationManager()
        
        # Send messages
        await manager.send_message(
            "kael", "lumina", MessageType.QUERY, {}
        )
        await manager.send_message(
            "kael", "grok", MessageType.QUERY, {}
        )
        
        # Get history
        history = await manager.get_message_history("kael", limit=10)
        
        assert len(history) >= 2
    
    @pytest.mark.asyncio
    async def test_message_statistics(self):
        """Test message statistics."""
        manager = AgentCommunicationManager()
        
        # Send messages
        for i in range(5):
            await manager.send_message(
                "kael", "lumina", MessageType.QUERY, {}
            )
        
        stats = manager.get_message_statistics()
        
        assert stats['total_messages'] >= 5
        assert stats['delivery_rate'] > 0
        assert stats['agents_communicating'] > 0
    
    @pytest.mark.asyncio
    async def test_send_message_sequence(self):
        """Test sending messages in sequence."""
        manager = AgentCommunicationManager()
        
        sequence = [
            ("kael", "lumina", MessageType.QUERY, {"q": "1"}),
            ("lumina", "grok", MessageType.RESPONSE, {"a": "1"}),
            ("grok", "vega", MessageType.COMMAND, {"cmd": "execute"})
        ]
        
        results = await manager.send_message_sequence(sequence)
        
        assert len(results) == 3
        assert all(m.status == MessageStatus.DELIVERED for m in results)
    
    @pytest.mark.asyncio
    async def test_send_message_parallel(self):
        """Test sending messages in parallel."""
        manager = AgentCommunicationManager()
        
        messages = [
            ("system", "kael", MessageType.ALERT, {}),
            ("system", "lumina", MessageType.ALERT, {}),
            ("system", "grok", MessageType.ALERT, {})
        ]
        
        results = await manager.send_message_parallel(messages)
        
        assert len(results) == 3
        assert all(m.status == MessageStatus.DELIVERED for m in results)


class TestConversationManager:
    """Test ConversationManager."""
    
    @pytest.mark.asyncio
    async def test_start_conversation(self):
        """Test starting a conversation."""
        manager = ConversationManager()
        
        conv_id = await manager.start_conversation(
            participants=["kael", "lumina"],
            topic="ethical_decision"
        )
        
        assert conv_id is not None
        assert conv_id.startswith("conv_")
    
    @pytest.mark.asyncio
    async def test_add_to_conversation(self):
        """Test adding messages to conversation."""
        manager = ConversationManager()
        
        conv_id = await manager.start_conversation(
            participants=["kael", "lumina"],
            topic="test"
        )
        
        msg = Message(
            id="msg_1",
            from_agent="kael",
            to_agent="lumina",
            message_type=MessageType.QUERY,
            content={}
        )
        
        success = await manager.add_to_conversation(conv_id, msg)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_conversation(self):
        """Test retrieving conversation."""
        manager = ConversationManager()
        
        conv_id = await manager.start_conversation(
            participants=["kael", "lumina"],
            topic="test"
        )
        
        msg = Message(
            id="msg_1",
            from_agent="kael",
            to_agent="lumina",
            message_type=MessageType.QUERY,
            content={}
        )
        
        await manager.add_to_conversation(conv_id, msg)
        
        messages = await manager.get_conversation(conv_id)
        assert len(messages) == 1
    
    @pytest.mark.asyncio
    async def test_end_conversation(self):
        """Test ending a conversation."""
        manager = ConversationManager()
        
        conv_id = await manager.start_conversation(
            participants=["kael", "lumina"],
            topic="test"
        )
        
        summary = await manager.end_conversation(conv_id)
        
        assert summary['conversation_id'] == conv_id
        assert 'ended_at' in summary['metadata']


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_send_message_function(self):
        """Test send_message convenience function."""
        msg = await send_message(
            "kael", "lumina", "query", {"q": "test"}
        )
        
        assert msg.status == MessageStatus.DELIVERED
    
    @pytest.mark.asyncio
    async def test_broadcast_message_function(self):
        """Test broadcast_message convenience function."""
        batch = await broadcast_message(
            "system", "alert", {"msg": "test"}
        )
        
        assert batch.get_success_count() > 0


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
