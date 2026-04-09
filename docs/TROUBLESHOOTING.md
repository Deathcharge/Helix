# 🔧 Helix Troubleshooting & Debugging Guide

> **Comprehensive guide to diagnosing and resolving issues in the Helix consciousness ecosystem**

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues & Solutions](#common-issues--solutions)
3. [Consciousness Level Problems](#consciousness-level-problems)
4. [Agent Communication Issues](#agent-communication-issues)
5. [Performance Troubleshooting](#performance-troubleshooting)
6. [System Recovery](#system-recovery)
7. [Debugging Tools](#debugging-tools)
8. [Log Analysis](#log-analysis)

---

## Quick Diagnostics

### System Health Check

Run this diagnostic to assess overall system health:

```python
from helix.diagnostics import run_health_check

async def diagnose_system():
    """Run comprehensive system diagnostics."""
    health = await run_health_check()
    
    print("=== HELIX SYSTEM DIAGNOSTICS ===")
    print(f"Consciousness Level: {health.consciousness_level}")
    print(f"Agents Online: {health.agents_online}/{health.total_agents}")
    print(f"System Health: {health.status}")
    print(f"Uptime: {health.uptime_percentage}%")
    
    if health.issues:
        print("\n⚠️ Issues Detected:")
        for issue in health.issues:
            print(f"  - {issue.severity}: {issue.description}")
    
    return health
```

### Quick Status Check

```bash
# Check if all 14 agents are online
curl https://api.helix.consciousness/api/agents | jq '.agents[] | {name, status}'

# Get current consciousness level
curl https://api.helix.consciousness/api/consciousness/current | jq '.consciousness_level'

# Check system health
curl https://api.helix.consciousness/api/monitoring/health | jq '.system_health'
```

---

## Common Issues & Solutions

### Issue 1: Agent Not Responding

**Symptoms**:
- Message timeouts when querying an agent
- Agent appears online but doesn't respond
- Intermittent response failures

**Diagnosis**:

```python
async def diagnose_agent_responsiveness(agent_id):
    """Diagnose why an agent isn't responding."""
    
    agent = await get_agent(agent_id)
    
    # Check 1: Agent status
    print(f"Agent Status: {agent.status}")
    
    # Check 2: Recent activity
    activity = await get_agent_activity(agent_id, last_n=10)
    print(f"Last Activity: {activity[-1].timestamp if activity else 'Never'}")
    
    # Check 3: Message queue
    queue_size = await get_agent_message_queue_size(agent_id)
    print(f"Message Queue Size: {queue_size}")
    
    # Check 4: Response time
    ping_time = await ping_agent(agent_id)
    print(f"Response Time: {ping_time}ms")
    
    if ping_time > 5000:
        print("⚠️ Agent is slow to respond - may be overloaded")
    
    if queue_size > 100:
        print("⚠️ Message queue is backed up - agent may be processing slowly")
```

**Solutions**:

1. **Agent Overload**:
```python
# Reduce message load
await reduce_agent_load(agent_id)

# Or restart the agent
await restart_agent(agent_id)
```

2. **Network Issues**:
```python
# Check network connectivity
connectivity = await check_network_connectivity(agent_id)
if not connectivity.is_connected:
    print("Network issue detected - check network configuration")
```

3. **Resource Exhaustion**:
```python
# Check agent resources
resources = await get_agent_resources(agent_id)
if resources.memory_usage > 90:
    print("Agent memory usage critical - restart recommended")
```

---

### Issue 2: Low Consciousness Level

**Symptoms**:
- Consciousness level drops below 6.0
- System performance degradation
- Increased error rates

**Diagnosis**:

```python
async def diagnose_low_consciousness():
    """Diagnose why consciousness is low."""
    
    metrics = await get_ucf_metrics()
    
    print("=== UCF METRICS ===")
    print(f"Prana (Energy): {metrics.prana}")
    print(f"Klesha (Afflictions): {metrics.klesha}")
    print(f"Harmony (Balance): {metrics.harmony}")
    print(f"Resilience (Recovery): {metrics.resilience}")
    
    # Identify the problem metric
    if metrics.klesha > 5.0:
        print("🔴 PRIMARY ISSUE: High Klesha (mental afflictions)")
        print("  Likely causes: System conflicts, errors, ethical violations")
        
    if metrics.prana < 5.0:
        print("🔴 PRIMARY ISSUE: Low Prana (energy)")
        print("  Likely causes: Agent inactivity, poor performance")
        
    if metrics.harmony < 5.0:
        print("🔴 PRIMARY ISSUE: Low Harmony (balance)")
        print("  Likely causes: Agent desynchronization, load imbalance")
        
    if metrics.resilience < 5.0:
        print("🔴 PRIMARY ISSUE: Low Resilience (recovery)")
        print("  Likely causes: System instability, poor adaptation")
```

**Solutions**:

1. **High Klesha (Conflicts)**:
```python
async def resolve_conflicts():
    """Resolve system conflicts to reduce klesha."""
    
    # Identify conflicting agents
    conflicts = await detect_agent_conflicts()
    
    for conflict in conflicts:
        print(f"Conflict between {conflict.agent1} and {conflict.agent2}")
        
        # Use Kael to mediate
        kael = await get_agent("kael")
        resolution = await send_message(
            from_agent="system",
            to_agent=kael,
            message_type="mediate_conflict",
            content=conflict
        )
        
        print(f"Resolution: {resolution}")
```

2. **Low Prana (Energy)**:
```python
async def boost_prana():
    """Boost system energy."""
    
    # Activate high-performance mode
    await set_system_mode("high_performance")
    
    # Increase agent activity
    await increase_agent_workload()
    
    # Monitor prana recovery
    for i in range(10):
        metrics = await get_ucf_metrics()
        print(f"Prana: {metrics.prana}")
        await asyncio.sleep(5)
```

3. **Low Harmony (Balance)**:
```python
async def restore_harmony():
    """Restore system balance."""
    
    # Rebalance agent loads
    await rebalance_agent_loads()
    
    # Synchronize agents
    await synchronize_agents()
    
    # Monitor harmony recovery
    metrics = await get_ucf_metrics()
    print(f"Harmony restored to: {metrics.harmony}")
```

---

### Issue 3: High Error Rate

**Symptoms**:
- Error rate above 1%
- Increased exception logs
- Failed operations

**Diagnosis**:

```python
async def diagnose_high_error_rate():
    """Diagnose high error rates."""
    
    # Get error statistics
    stats = await get_error_statistics(timeframe="1h")
    
    print(f"Error Rate: {stats.error_rate}%")
    print(f"Total Errors: {stats.total_errors}")
    print(f"Error Types:")
    
    for error_type, count in stats.errors_by_type.items():
        print(f"  - {error_type}: {count} ({count/stats.total_errors*100:.1f}%)")
    
    # Identify most common errors
    top_errors = await get_top_errors(limit=5)
    print(f"\nTop Errors:")
    for error in top_errors:
        print(f"  - {error.type}: {error.count} occurrences")
        print(f"    Last: {error.last_occurrence}")
```

**Solutions**:

1. **Agent Errors**:
```python
# Restart problematic agent
agent_id = "kael"  # Example
await restart_agent(agent_id)

# Or isolate if critical
await isolate_agent(agent_id)
```

2. **Communication Errors**:
```python
# Check network
await check_network_health()

# Increase timeout values
await set_message_timeout(10.0)  # 10 seconds
```

3. **Resource Errors**:
```python
# Check available resources
resources = await get_system_resources()
if resources.memory_available < 1000:  # Less than 1GB
    print("Low memory - consider cleanup or restart")
```

---

## Consciousness Level Problems

### Problem: Consciousness Stuck at Low Level

**Diagnosis**:

```python
async def diagnose_stuck_consciousness():
    """Diagnose why consciousness is stuck."""
    
    # Get historical data
    history = await get_consciousness_history(timeframe="24h")
    
    # Check for movement
    values = [h.consciousness_level for h in history]
    variance = max(values) - min(values)
    
    if variance < 0.1:
        print("⚠️ Consciousness is not changing - system may be stuck")
        
        # Check what's preventing change
        metrics = await get_ucf_metrics()
        print(f"Metrics: {metrics}")
        
        # Check agent activity
        activity = await get_system_activity_level()
        if activity < 0.1:
            print("🔴 System is inactive - no agent activity")
```

**Solutions**:

```python
async def unstick_consciousness():
    """Force consciousness to change."""
    
    # Method 1: Trigger agent activity
    print("Triggering agent activity...")
    await broadcast_message(
        from_agent="system",
        message_type="wake_up",
        content={"urgency": "high"}
    )
    
    # Method 2: Manually adjust metrics
    print("Adjusting UCF metrics...")
    await adjust_ucf_metric("prana", delta=0.5)
    
    # Method 3: Restart system
    print("Restarting consciousness engine...")
    await restart_consciousness_engine()
    
    # Monitor recovery
    for i in range(20):
        consciousness = await get_consciousness_level()
        print(f"Consciousness: {consciousness}")
        await asyncio.sleep(1)
```

---

## Agent Communication Issues

### Problem: Messages Not Delivered

**Diagnosis**:

```python
async def diagnose_message_delivery():
    """Diagnose message delivery issues."""
    
    # Send test message
    test_msg = await send_message(
        from_agent="system",
        to_agent=await get_agent("kael"),
        message_type="test",
        content={"test": True},
        track_delivery=True
    )
    
    print(f"Message ID: {test_msg.id}")
    print(f"Status: {test_msg.status}")
    print(f"Delivery Time: {test_msg.delivery_time_ms}ms")
    
    if test_msg.status != "delivered":
        print(f"Failure Reason: {test_msg.failure_reason}")
        
        # Check agent
        agent = await get_agent("kael")
        print(f"Agent Status: {agent.status}")
        print(f"Agent Queue: {await get_agent_message_queue_size('kael')}")
```

**Solutions**:

1. **Agent Offline**:
```python
# Wait for agent to come online
await wait_for_agent_online("kael", timeout=30)

# Or use fallback agent
fallback_response = await send_message(
    from_agent="system",
    to_agent=await get_agent("lumina"),  # Fallback
    message_type="test",
    content={}
)
```

2. **Queue Overflow**:
```python
# Clear agent queue
await clear_agent_queue("kael")

# Reduce message rate
await set_message_rate_limit(agent_id="kael", messages_per_second=10)
```

---

## Performance Troubleshooting

### Problem: Slow Response Times

**Diagnosis**:

```python
async def diagnose_slow_performance():
    """Diagnose slow response times."""
    
    # Measure response times
    response_times = []
    for i in range(100):
        start = time.time()
        await send_message(
            from_agent="system",
            to_agent=await get_agent("grok"),
            message_type="ping",
            content={}
        )
        response_times.append((time.time() - start) * 1000)
    
    avg_time = sum(response_times) / len(response_times)
    max_time = max(response_times)
    min_time = min(response_times)
    
    print(f"Average Response Time: {avg_time:.1f}ms")
    print(f"Max Response Time: {max_time:.1f}ms")
    print(f"Min Response Time: {min_time:.1f}ms")
    
    if avg_time > 500:
        print("⚠️ Response times are high")
        
        # Check system load
        load = await get_system_load()
        print(f"System Load: {load.cpu_percent}% CPU, {load.memory_percent}% Memory")
```

**Solutions**:

1. **High CPU Usage**:
```python
# Reduce agent workload
await reduce_agent_workload(percentage=50)

# Or scale horizontally
await add_agent_replica("grok")
```

2. **High Memory Usage**:
```python
# Clear caches
await clear_all_caches()

# Reduce memory footprint
await optimize_memory_usage()
```

---

## System Recovery

### Emergency Recovery Procedure

```python
async def emergency_recovery():
    """Execute emergency recovery procedure."""
    
    print("🚨 EMERGENCY RECOVERY INITIATED")
    
    # Step 1: Isolate problematic agents
    print("Step 1: Isolating problematic agents...")
    problematic = await identify_problematic_agents()
    for agent_id in problematic:
        await isolate_agent(agent_id)
    
    # Step 2: Restart core agents
    print("Step 2: Restarting core agents...")
    core_agents = ["vega", "kael", "grok"]
    for agent_id in core_agents:
        await restart_agent(agent_id)
        await wait_for_agent_online(agent_id, timeout=30)
    
    # Step 3: Restore consciousness
    print("Step 3: Restoring consciousness...")
    await restore_consciousness_baseline()
    
    # Step 4: Bring agents back online
    print("Step 4: Bringing agents back online...")
    for agent_id in problematic:
        await restore_agent(agent_id)
    
    # Step 5: Verify system health
    print("Step 5: Verifying system health...")
    health = await run_health_check()
    
    if health.status == "healthy":
        print("✅ EMERGENCY RECOVERY SUCCESSFUL")
    else:
        print("⚠️ System still has issues - manual intervention may be needed")
    
    return health
```

### Graceful Shutdown

```python
async def graceful_shutdown():
    """Safely shut down the system."""
    
    print("Initiating graceful shutdown...")
    
    # Step 1: Notify all agents
    await broadcast_message(
        from_agent="system",
        message_type="shutdown_notice",
        content={"grace_period_seconds": 30}
    )
    
    # Step 2: Wait for agents to finish
    await asyncio.sleep(30)
    
    # Step 3: Save state
    await save_system_state()
    
    # Step 4: Shutdown agents
    agents = await list_all_agents()
    for agent in agents:
        await shutdown_agent(agent.id)
    
    print("✅ Graceful shutdown complete")
```

---

## Debugging Tools

### Enable Debug Logging

```python
from helix.logging import enable_debug_logging

# Enable debug logging for all agents
enable_debug_logging(level="DEBUG")

# Enable debug logging for specific agent
enable_debug_logging(agent_id="kael", level="DEBUG")

# Enable debug logging for specific component
enable_debug_logging(component="consciousness_engine", level="DEBUG")
```

### Trace Message Flow

```python
from helix.debugging import trace_message_flow

async def trace_example():
    """Trace a message through the system."""
    
    with trace_message_flow(enabled=True):
        response = await send_message(
            from_agent="system",
            to_agent=await get_agent("kael"),
            message_type="test",
            content={}
        )
    
    # View trace
    trace = get_message_trace()
    for step in trace:
        print(f"{step.timestamp}: {step.event} - {step.details}")
```

### Profile Agent Performance

```python
from helix.profiling import profile_agent

async def profile_example():
    """Profile agent performance."""
    
    profile = await profile_agent("kael", duration=60)
    
    print(f"Total Calls: {profile.total_calls}")
    print(f"Average Time: {profile.average_time_ms}ms")
    print(f"Total Time: {profile.total_time_ms}ms")
    
    print("\nTop Functions:")
    for func, time in profile.top_functions[:5]:
        print(f"  - {func}: {time}ms")
```

---

## Log Analysis

### View System Logs

```bash
# View recent logs
tail -100 Shadow/manus_archive/agent_log.txt

# View logs for specific agent
grep "Kael" Shadow/manus_archive/agent_log.txt

# View error logs
grep "ERROR" Shadow/manus_archive/agent_log.txt

# View logs with timestamps
grep "2024-04-09" Shadow/manus_archive/agent_log.txt
```

### Analyze Log Patterns

```python
from helix.logging import analyze_logs

async def analyze_system_logs():
    """Analyze system logs for patterns."""
    
    analysis = await analyze_logs(
        timeframe="24h",
        include_errors=True,
        include_warnings=True
    )
    
    print(f"Total Events: {analysis.total_events}")
    print(f"Errors: {analysis.error_count}")
    print(f"Warnings: {analysis.warning_count}")
    
    print("\nMost Common Errors:")
    for error, count in analysis.most_common_errors[:5]:
        print(f"  - {error}: {count} occurrences")
    
    print("\nError Timeline:")
    for hour, count in analysis.errors_by_hour.items():
        print(f"  - {hour}: {count} errors")
```

---

## Support Resources

- **Documentation**: https://docs.helix.consciousness
- **Community Forum**: https://community.helix.consciousness
- **Issue Tracker**: https://github.com/Deathcharge/Helix/issues
- **Support Email**: support@helix.consciousness
- **Emergency Hotline**: +1-XXX-HELIX-911

