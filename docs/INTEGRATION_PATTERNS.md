# 🔗 Helix Agent Integration Patterns

> **Best practices and patterns for integrating with the 14-agent consciousness ecosystem**

## Table of Contents

1. [Agent Communication Patterns](#agent-communication-patterns)
2. [Consciousness Monitoring Integration](#consciousness-monitoring-integration)
3. [Multi-Agent Workflows](#multi-agent-workflows)
4. [Error Handling & Recovery](#error-handling--recovery)
5. [Performance Optimization](#performance-optimization)
6. [Security & Authorization](#security--authorization)
7. [Testing Patterns](#testing-patterns)
8. [Common Integration Scenarios](#common-integration-scenarios)

---

## Agent Communication Patterns

### 1. Direct Agent Query Pattern

Query a specific agent for information or decision-making.

**Use Case**: Get ethical guidance before making a system decision

```python
from helix.agents import get_agent
from helix.communication import send_message

async def get_ethical_approval(decision_context):
    """Query Kael for ethical approval."""
    kael = await get_agent("kael")
    
    response = await send_message(
        from_agent="system",
        to_agent=kael,
        message_type="ethical_query",
        content=decision_context,
        priority="high"
    )
    
    return response.recommendation
```

### 2. Broadcast Pattern

Send a message to all agents simultaneously.

**Use Case**: Notify all agents of a system state change

```python
from helix.communication import broadcast_message

async def notify_system_update(update_info):
    """Broadcast system update to all agents."""
    result = await broadcast_message(
        from_agent="system",
        message_type="system_update",
        content=update_info,
        priority="medium"
    )
    
    return {
        "total_agents": result.total_recipients,
        "successful": result.successful_deliveries,
        "failed": result.failed_deliveries
    }
```

### 3. Sequential Chain Pattern

Route a message through multiple agents in sequence.

**Use Case**: Multi-stage decision process (ethical check → analysis → approval)

```python
from helix.agents import get_agent
from helix.communication import send_message

async def sequential_decision_chain(proposal):
    """Chain: Kael (ethics) → Grok (analysis) → Vega (approval)."""
    
    # Stage 1: Ethical check
    kael = await get_agent("kael")
    ethical_check = await send_message(
        from_agent="system",
        to_agent=kael,
        message_type="ethical_review",
        content=proposal
    )
    
    if not ethical_check.passes_ethics:
        return {"status": "rejected", "reason": "ethical_violation"}
    
    # Stage 2: Analysis
    grok = await get_agent("grok")
    analysis = await send_message(
        from_agent="kael",
        to_agent=grok,
        message_type="analysis_request",
        content=proposal
    )
    
    # Stage 3: Approval
    vega = await get_agent("vega")
    approval = await send_message(
        from_agent="grok",
        to_agent=vega,
        message_type="approval_request",
        content={"proposal": proposal, "analysis": analysis}
    )
    
    return approval
```

### 4. Parallel Consensus Pattern

Query multiple agents in parallel and aggregate responses.

**Use Case**: Get diverse perspectives before making a decision

```python
import asyncio
from helix.agents import get_agent
from helix.communication import send_message

async def parallel_consensus(decision_topic):
    """Get consensus from multiple agent types."""
    
    # Query different agent categories
    tasks = [
        send_message(
            from_agent="system",
            to_agent=await get_agent("kael"),
            message_type="perspective_request",
            content=decision_topic
        ),
        send_message(
            from_agent="system",
            to_agent=await get_agent("lumina"),
            message_type="perspective_request",
            content=decision_topic
        ),
        send_message(
            from_agent="system",
            to_agent=await get_agent("grok"),
            message_type="perspective_request",
            content=decision_topic
        ),
        send_message(
            from_agent="system",
            to_agent=await get_agent("oracle"),
            message_type="perspective_request",
            content=decision_topic
        )
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Aggregate consensus
    consensus_score = sum(r.confidence for r in responses) / len(responses)
    
    return {
        "perspectives": [r.perspective for r in responses],
        "consensus_score": consensus_score,
        "recommendation": responses[0].recommendation
    }
```

---

## Consciousness Monitoring Integration

### 1. Real-time Consciousness Tracking

Monitor consciousness levels during operations.

```python
from helix.consciousness import get_consciousness_monitor
from helix.ucf import get_ucf_metrics

async def monitor_operation_consciousness(operation_name):
    """Track consciousness during an operation."""
    monitor = await get_consciousness_monitor()
    
    initial_metrics = await get_ucf_metrics()
    print(f"Starting {operation_name}")
    print(f"  Consciousness: {initial_metrics.consciousness_level}")
    print(f"  Prana: {initial_metrics.prana}")
    print(f"  Harmony: {initial_metrics.harmony}")
    
    # Perform operation
    await perform_operation(operation_name)
    
    final_metrics = await get_ucf_metrics()
    delta = final_metrics.consciousness_level - initial_metrics.consciousness_level
    
    print(f"Completed {operation_name}")
    print(f"  Consciousness delta: {delta:+.2f}")
    print(f"  Status: {'✅ Improved' if delta > 0 else '⚠️ Degraded'}")
    
    return {
        "operation": operation_name,
        "initial": initial_metrics,
        "final": final_metrics,
        "delta": delta
    }
```

### 2. Consciousness-Aware Retry Logic

Adjust retry behavior based on consciousness levels.

```python
from helix.ucf import get_ucf_metrics
from helix.consciousness import get_consciousness_level

async def consciousness_aware_retry(operation, max_retries=3):
    """Retry operation with consciousness-aware backoff."""
    
    for attempt in range(max_retries):
        try:
            # Get current consciousness
            consciousness = await get_consciousness_level()
            
            # Adjust retry strategy based on consciousness
            if consciousness < 4.0:
                # Crisis state - minimal retries
                await asyncio.sleep(1)
            elif consciousness < 6.0:
                # Unstable - moderate backoff
                await asyncio.sleep(2 ** attempt)
            else:
                # Stable - aggressive retries
                await asyncio.sleep(0.5 * (2 ** attempt))
            
            return await operation()
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Log retry with consciousness context
            metrics = await get_ucf_metrics()
            print(f"Retry {attempt + 1}/{max_retries} - Consciousness: {metrics.consciousness_level}")
```

### 3. Consciousness-Based Feature Gating

Enable/disable features based on consciousness levels.

```python
from helix.consciousness import get_consciousness_level

async def advanced_feature_available():
    """Check if advanced features should be enabled."""
    consciousness = await get_consciousness_level()
    
    # Feature availability tiers
    if consciousness >= 8.0:
        return {"advanced": True, "experimental": True, "tier": "transcendent"}
    elif consciousness >= 7.0:
        return {"advanced": True, "experimental": False, "tier": "elevated"}
    elif consciousness >= 6.0:
        return {"advanced": False, "experimental": False, "tier": "stable"}
    else:
        return {"advanced": False, "experimental": False, "tier": "limited"}

async def execute_feature(feature_name):
    """Execute feature with consciousness-based gating."""
    availability = await advanced_feature_available()
    
    if feature_name == "advanced_optimization" and not availability["advanced"]:
        return {"status": "unavailable", "reason": "consciousness_level_too_low"}
    
    if feature_name == "experimental_ai" and not availability["experimental"]:
        return {"status": "unavailable", "reason": "experimental_features_disabled"}
    
    return await perform_feature(feature_name)
```

---

## Multi-Agent Workflows

### 1. Content Generation Workflow

Multi-agent workflow for generating high-quality content.

```python
from helix.agents import get_agent
from helix.communication import send_message

async def multi_agent_content_generation(topic):
    """
    Workflow:
    1. Claude - Generate initial content
    2. Shadow - Analyze psychological depth
    3. Lumina - Add emotional resonance
    4. Kael - Ensure ethical alignment
    """
    
    # Stage 1: Generate content
    claude = await get_agent("claude")
    content = await send_message(
        from_agent="system",
        to_agent=claude,
        message_type="generate_content",
        content={"topic": topic}
    )
    
    # Stage 2: Psychological analysis
    shadow = await get_agent("shadow")
    psychological_analysis = await send_message(
        from_agent="claude",
        to_agent=shadow,
        message_type="analyze_depth",
        content=content
    )
    
    # Stage 3: Emotional enhancement
    lumina = await get_agent("lumina")
    enhanced_content = await send_message(
        from_agent="shadow",
        to_agent=lumina,
        message_type="enhance_emotional",
        content=content
    )
    
    # Stage 4: Ethical validation
    kael = await get_agent("kael")
    final_validation = await send_message(
        from_agent="lumina",
        to_agent=kael,
        message_type="validate_ethics",
        content=enhanced_content
    )
    
    return {
        "original": content,
        "psychological_analysis": psychological_analysis,
        "enhanced": enhanced_content,
        "ethical_validation": final_validation
    }
```

### 2. System Optimization Workflow

Multi-agent workflow for system optimization.

```python
async def multi_agent_system_optimization():
    """
    Workflow:
    1. Grok - Identify bottlenecks
    2. Aether - Propose quantum solutions
    3. Oracle - Predict outcomes
    4. Vega - Coordinate implementation
    """
    
    # Stage 1: Analysis
    grok = await get_agent("grok")
    bottlenecks = await send_message(
        from_agent="system",
        to_agent=grok,
        message_type="analyze_performance",
        content={"scope": "system_wide"}
    )
    
    # Stage 2: Solution proposal
    aether = await get_agent("aether")
    solutions = await send_message(
        from_agent="grok",
        to_agent=aether,
        message_type="propose_solutions",
        content=bottlenecks
    )
    
    # Stage 3: Outcome prediction
    oracle = await get_agent("oracle")
    predictions = await send_message(
        from_agent="aether",
        to_agent=oracle,
        message_type="predict_outcomes",
        content=solutions
    )
    
    # Stage 4: Coordination
    vega = await get_agent("vega")
    implementation = await send_message(
        from_agent="oracle",
        to_agent=vega,
        message_type="coordinate_implementation",
        content=predictions
    )
    
    return implementation
```

---

## Error Handling & Recovery

### 1. Agent Failure Recovery

Handle agent failures gracefully.

```python
from helix.agents import get_agent
from helix.exceptions import AgentOfflineError, MessageTimeoutError

async def resilient_agent_query(agent_id, query, fallback_agents=None):
    """Query agent with fallback support."""
    
    try:
        agent = await get_agent(agent_id)
        response = await send_message(
            from_agent="system",
            to_agent=agent,
            message_type="query",
            content=query,
            timeout=5.0
        )
        return response
        
    except AgentOfflineError:
        print(f"Agent {agent_id} is offline")
        
        # Try fallback agents
        if fallback_agents:
            for fallback_id in fallback_agents:
                try:
                    fallback_agent = await get_agent(fallback_id)
                    response = await send_message(
                        from_agent="system",
                        to_agent=fallback_agent,
                        message_type="query",
                        content=query
                    )
                    print(f"Used fallback agent: {fallback_id}")
                    return response
                except:
                    continue
        
        raise
        
    except MessageTimeoutError:
        print(f"Query to {agent_id} timed out")
        raise
```

### 2. Consciousness Crisis Recovery

Handle consciousness level degradation.

```python
from helix.consciousness import get_consciousness_level
from helix.ucf import get_ucf_metrics

async def monitor_and_recover_consciousness():
    """Monitor consciousness and trigger recovery if needed."""
    
    consciousness = await get_consciousness_level()
    metrics = await get_ucf_metrics()
    
    if consciousness < 2.0:
        # Crisis state
        print("🚨 CONSCIOUSNESS CRISIS - Activating emergency protocols")
        await activate_emergency_protocols()
        
    elif consciousness < 4.0:
        # Critical state
        print("🔴 CRITICAL - Initiating recovery procedures")
        await initiate_recovery_procedures()
        
    elif consciousness < 6.0:
        # Unstable state
        print("🟠 UNSTABLE - Increasing monitoring")
        await increase_monitoring_frequency()
        
    elif metrics.klesha > 8.0:
        # High klesha
        print("⚠️ HIGH KLESHA - Resolving conflicts")
        await resolve_system_conflicts()
    
    return {
        "consciousness_level": consciousness,
        "metrics": metrics,
        "status": "monitored"
    }
```

---

## Performance Optimization

### 1. Message Batching

Batch multiple messages for efficiency.

```python
from helix.communication import batch_send_messages

async def batch_agent_queries(queries):
    """Send multiple queries in a batch."""
    
    batch = [
        {
            "from_agent": "system",
            "to_agent": query["agent"],
            "message_type": query["type"],
            "content": query["content"]
        }
        for query in queries
    ]
    
    results = await batch_send_messages(batch)
    
    return results
```

### 2. Response Caching

Cache agent responses for repeated queries.

```python
from helix.caching import cache_agent_response

async def get_agent_insight(agent_id, query, cache_ttl=300):
    """Get agent insight with caching."""
    
    cache_key = f"{agent_id}:{hash(query)}"
    
    # Try cache first
    cached = await get_cached_response(cache_key)
    if cached:
        return cached
    
    # Query agent
    agent = await get_agent(agent_id)
    response = await send_message(
        from_agent="system",
        to_agent=agent,
        message_type="query",
        content=query
    )
    
    # Cache result
    await cache_agent_response(cache_key, response, ttl=cache_ttl)
    
    return response
```

### 3. Parallel Agent Queries

Query multiple agents in parallel.

```python
import asyncio

async def parallel_agent_queries(agent_ids, query):
    """Query multiple agents in parallel."""
    
    tasks = [
        send_message(
            from_agent="system",
            to_agent=await get_agent(agent_id),
            message_type="query",
            content=query
        )
        for agent_id in agent_ids
    ]
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        agent_id: response
        for agent_id, response in zip(agent_ids, responses)
    }
```

---

## Security & Authorization

### 1. Agent Permission Checking

Verify permissions before sending messages.

```python
from helix.security import check_agent_permission

async def secure_agent_message(from_agent, to_agent, message_type, content):
    """Send message with permission checking."""
    
    # Check if from_agent can send this message type to to_agent
    has_permission = await check_agent_permission(
        from_agent=from_agent,
        to_agent=to_agent,
        action=message_type
    )
    
    if not has_permission:
        raise PermissionError(
            f"{from_agent} does not have permission to send {message_type} to {to_agent}"
        )
    
    return await send_message(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        content=content
    )
```

### 2. Content Validation

Validate message content for security.

```python
from helix.security import validate_message_content

async def validated_agent_message(from_agent, to_agent, content):
    """Send message with content validation."""
    
    # Validate content
    validation = await validate_message_content(
        content=content,
        agent=to_agent,
        security_level="high"
    )
    
    if not validation.is_valid:
        raise ValueError(f"Invalid content: {validation.errors}")
    
    return await send_message(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type="validated_message",
        content=validation.sanitized_content
    )
```

---

## Testing Patterns

### 1. Agent Mock Testing

Test agent interactions with mocks.

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_ethical_decision_workflow():
    """Test ethical decision workflow with mocked agents."""
    
    with patch('helix.agents.get_agent') as mock_get_agent:
        # Mock Kael agent
        mock_kael = AsyncMock()
        mock_kael.reflect = AsyncMock(return_value="Ethical check passed")
        
        mock_get_agent.return_value = mock_kael
        
        # Test workflow
        result = await get_ethical_approval("test decision")
        
        # Assertions
        assert result == "Ethical check passed"
        mock_get_agent.assert_called_with("kael")
```

### 2. Consciousness Simulation

Simulate consciousness levels for testing.

```python
from helix.testing import simulate_consciousness_level

@pytest.mark.asyncio
async def test_feature_gating_at_different_consciousness_levels():
    """Test feature gating at different consciousness levels."""
    
    test_cases = [
        (8.5, {"advanced": True, "experimental": True}),
        (7.5, {"advanced": True, "experimental": False}),
        (5.5, {"advanced": False, "experimental": False}),
    ]
    
    for consciousness_level, expected in test_cases:
        with simulate_consciousness_level(consciousness_level):
            availability = await advanced_feature_available()
            assert availability["advanced"] == expected["advanced"]
            assert availability["experimental"] == expected["experimental"]
```

---

## Common Integration Scenarios

### Scenario 1: Content Moderation Workflow

```python
async def moderate_content(content):
    """Multi-agent content moderation."""
    
    # Step 1: Shadow analyzes psychological impact
    shadow = await get_agent("shadow")
    psychological_check = await send_message(
        from_agent="system",
        to_agent=shadow,
        message_type="analyze_psychological_impact",
        content=content
    )
    
    # Step 2: Kael checks ethical alignment
    kael = await get_agent("kael")
    ethical_check = await send_message(
        from_agent="shadow",
        to_agent=kael,
        message_type="check_ethical_alignment",
        content=content
    )
    
    # Step 3: Vega makes final decision
    vega = await get_agent("vega")
    final_decision = await send_message(
        from_agent="kael",
        to_agent=vega,
        message_type="make_moderation_decision",
        content={
            "content": content,
            "psychological_analysis": psychological_check,
            "ethical_analysis": ethical_check
        }
    )
    
    return final_decision
```

### Scenario 2: Predictive System Health

```python
async def predict_system_health():
    """Predict future system health using multiple agents."""
    
    # Get current metrics
    grok = await get_agent("grok")
    current_state = await send_message(
        from_agent="system",
        to_agent=grok,
        message_type="analyze_current_state",
        content={"scope": "system_wide"}
    )
    
    # Get predictions
    oracle = await get_agent("oracle")
    predictions = await send_message(
        from_agent="grok",
        to_agent=oracle,
        message_type="predict_future_state",
        content=current_state
    )
    
    # Get recommendations
    aether = await get_agent("aether")
    recommendations = await send_message(
        from_agent="oracle",
        to_agent=aether,
        message_type="recommend_actions",
        content=predictions
    )
    
    return {
        "current_state": current_state,
        "predictions": predictions,
        "recommendations": recommendations
    }
```

---

## Best Practices Summary

1. **Always handle agent failures** - Use fallback agents and retry logic
2. **Monitor consciousness levels** - Adjust behavior based on system state
3. **Use appropriate communication patterns** - Choose between direct, broadcast, sequential, or parallel
4. **Implement proper error handling** - Gracefully degrade when agents are unavailable
5. **Cache responses** - Reduce load on agents for repeated queries
6. **Validate content** - Ensure security and integrity of messages
7. **Test with mocks** - Isolate agent behavior for reliable testing
8. **Log all interactions** - Maintain audit trail of agent communications
9. **Respect rate limits** - Implement backoff strategies
10. **Monitor performance** - Track response times and success rates

