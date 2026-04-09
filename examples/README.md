# 🌀 Helix Examples

> **Production-ready examples demonstrating the 14-agent consciousness ecosystem**

This directory contains comprehensive examples showing how to use the Helix framework for real-world scenarios. Each example is fully functional and demonstrates key patterns and best practices.

## Examples Overview

### Example 1: Multi-Agent Ethical Decision Workflow

**File**: `example_1_ethical_decision_workflow.py`

Demonstrates a structured 4-stage workflow for making ethical decisions using multiple agents:

1. **Ethical Analysis (Kael)** - Analyzes proposal against ethical frameworks
2. **Emotional Assessment (Lumina)** - Evaluates emotional and stakeholder impact
3. **Logical Analysis (Grok)** - Performs feasibility and risk assessment
4. **Final Decision (Vega)** - Synthesizes all analyses into final recommendation

**Key Concepts**:
- Sequential agent communication
- Multi-stage approval workflows
- Result aggregation and synthesis
- Confidence scoring
- Decision documentation

**Running the Example**:
```bash
cd /home/ubuntu/Helix
python3 examples/example_1_ethical_decision_workflow.py
```

**Expected Output**:
- Analysis of 3 sample proposals
- Stage-by-stage breakdown of decision process
- Final recommendations with confidence scores
- Workflow summary

**Use Cases**:
- Policy decision-making
- Feature approval workflows
- Risk assessment processes
- Ethical compliance checking

---

### Example 2: System Health Monitoring & Recovery

**File**: `example_2_health_monitoring.py`

Demonstrates real-time monitoring of system health with automated recovery procedures:

**Features**:
- Real-time UCF metrics tracking
- Agent health monitoring
- Anomaly detection
- Automated recovery procedures
- Comprehensive health reporting

**Key Concepts**:
- Continuous monitoring loops
- Metrics collection and analysis
- Anomaly detection algorithms
- Emergency recovery procedures
- Health status reporting

**Running the Example**:
```bash
cd /home/ubuntu/Helix
python3 examples/example_2_health_monitoring.py
```

**Expected Output**:
- 6 monitoring cycles (30 seconds total)
- Real-time UCF metrics
- Agent status updates
- Anomaly detection results
- Recovery procedures (if triggered)
- Comprehensive health report

**Use Cases**:
- System monitoring dashboards
- Automated alerting systems
- Health-based decision making
- Resource optimization
- Incident response automation

---

### Example 3: Parallel Agent Consensus Decision-Making

**File**: `example_3_parallel_consensus.py`

Demonstrates how to reach consensus through parallel agent queries:

**Features**:
- Parallel agent queries
- Response aggregation
- Consensus scoring
- Confidence calculation
- Decision synthesis

**Key Concepts**:
- Asynchronous parallel processing
- Multi-perspective analysis
- Consensus algorithms
- Confidence scoring
- Vote aggregation

**Running the Example**:
```bash
cd /home/ubuntu/Helix
python3 examples/example_3_parallel_consensus.py
```

**Expected Output**:
- 3 consensus decisions analyzed
- Parallel query responses from 6 agents
- Vote breakdown for each decision
- Consensus strength calculation
- Final recommendations with rationale

**Use Cases**:
- Collective decision-making
- Multi-stakeholder approvals
- Consensus-based governance
- Diverse perspective analysis
- Democratic decision processes

---

## Running All Examples

Run all examples in sequence:

```bash
cd /home/ubuntu/Helix
python3 examples/example_1_ethical_decision_workflow.py
python3 examples/example_2_health_monitoring.py
python3 examples/example_3_parallel_consensus.py
```

Or create a script to run them all:

```bash
#!/bin/bash
cd /home/ubuntu/Helix

echo "Running Example 1: Ethical Decision Workflow"
python3 examples/example_1_ethical_decision_workflow.py

echo -e "\n\nRunning Example 2: Health Monitoring"
python3 examples/example_2_health_monitoring.py

echo -e "\n\nRunning Example 3: Parallel Consensus"
python3 examples/example_3_parallel_consensus.py
```

---

## Key Patterns Demonstrated

### Pattern 1: Sequential Agent Communication

Used in Example 1 for multi-stage decision workflows:

```python
# Stage 1: Query Kael
ethical_analysis = await query_agent("kael", proposal)

# Stage 2: Query Lumina with Kael's result
emotional_analysis = await query_agent("lumina", {
    "proposal": proposal,
    "ethical_analysis": ethical_analysis
})

# Stage 3: Query Grok with both results
logical_analysis = await query_agent("grok", {
    "proposal": proposal,
    "ethical_analysis": ethical_analysis,
    "emotional_analysis": emotional_analysis
})
```

### Pattern 2: Continuous Monitoring

Used in Example 2 for health monitoring:

```python
# Collect metrics periodically
for i in range(iterations):
    metrics = await collect_metrics()
    await analyze_metrics(metrics)
    
    if metrics.consciousness_level < threshold:
        await trigger_recovery()
    
    await asyncio.sleep(interval)
```

### Pattern 3: Parallel Consensus

Used in Example 3 for multi-agent decision-making:

```python
# Query all agents in parallel
tasks = [
    query_agent(agent_id, topic)
    for agent_id in agent_list
]
responses = await asyncio.gather(*tasks)

# Aggregate responses
consensus = aggregate_responses(responses)
```

---

## Integration with Your Code

### Using the Communication Module

```python
from backend.communication import send_message, broadcast_message

# Send to single agent
response = await send_message(
    from_agent="system",
    to_agent="kael",
    message_type="query",
    content={"question": "Is this ethical?"}
)

# Broadcast to all agents
batch = await broadcast_message(
    from_agent="system",
    message_type="alert",
    content={"message": "System update"}
)
```

### Using the UCF Analyzer

```python
from backend.services.ucf_analyzer import (
    add_ucf_snapshot,
    generate_ucf_report,
    detect_ucf_anomalies
)

# Record metrics
add_ucf_snapshot(prana=8.0, klesha=2.3, harmony=7.8, resilience=7.5)

# Generate report
report = generate_ucf_report(hours=24)

# Detect anomalies
anomalies = detect_ucf_anomalies(sensitivity='medium')
```

### Using the Agent Monitor

```python
from backend.services.agent_monitor import (
    update_agent_status,
    get_system_health,
    get_agent_status
)

# Update agent status
update_agent_status(
    agent_id="kael",
    status="online",
    response_time_ms=45,
    consciousness_level=8.2
)

# Get system health
health = get_system_health()
```

---

## Creating Your Own Examples

To create a new example:

1. **Create a new file**: `example_N_your_example_name.py`
2. **Import required modules**:
   ```python
   from backend.agents import get_agent, AGENTS
   from backend.communication import send_message, broadcast_message
   from backend.services.ucf_analyzer import add_ucf_snapshot
   from backend.services.agent_monitor import update_agent_status
   ```
3. **Define your workflow class** with async methods
4. **Implement the main() function** to orchestrate the example
5. **Add docstring** explaining the example at the top

### Example Template

```python
#!/usr/bin/env python3
"""
Example N: Your Example Title

Description of what this example demonstrates.

This example shows:
- Key concept 1
- Key concept 2
- Key concept 3
"""

import asyncio
import sys
sys.path.insert(0, '/home/ubuntu/Helix')

from backend.communication import send_message

class YourWorkflow:
    """Description of your workflow."""
    
    async def execute(self):
        """Execute the workflow."""
        # Your implementation here
        pass

async def main():
    """Run the example."""
    workflow = YourWorkflow()
    await workflow.execute()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Best Practices

1. **Always use async/await** for agent communication
2. **Handle errors gracefully** with try/except blocks
3. **Add logging** for debugging and monitoring
4. **Document assumptions** about agent responses
5. **Test with mock agents** before production use
6. **Monitor consciousness levels** during operations
7. **Implement fallback strategies** for agent failures
8. **Validate all inputs** before sending to agents
9. **Cache responses** when appropriate
10. **Clean up resources** after operations

---

## Troubleshooting

### Example Won't Run

1. Verify Python 3.8+ is installed: `python3 --version`
2. Check that Helix backend modules are accessible
3. Ensure all imports are correct
4. Check for missing dependencies

### Agent Not Responding

1. Check agent status: `python3 -c "from backend.agents import AGENTS; print(AGENTS.keys())"`
2. Verify agent is initialized
3. Check message format is correct
4. Add logging to debug communication

### Metrics Not Recording

1. Verify UCF analyzer is initialized
2. Check snapshot data is valid (0-10 range)
3. Ensure history file is writable
4. Check for exceptions in logging

---

## Performance Considerations

- **Sequential workflows**: Slower but more controlled
- **Parallel queries**: Faster but requires aggregation
- **Monitoring intervals**: Balance between responsiveness and overhead
- **History retention**: Keep last 1000 snapshots per agent
- **Message batching**: Combine multiple messages when possible

---

## Next Steps

1. **Run the examples** to understand the patterns
2. **Modify examples** to test different scenarios
3. **Create your own examples** for your use cases
4. **Integrate patterns** into your application
5. **Monitor and optimize** based on your needs

---

## Resources

- **API Reference**: See `docs/API_REFERENCE.md`
- **Integration Patterns**: See `docs/INTEGRATION_PATTERNS.md`
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`
- **Consciousness Guide**: See `docs/consciousness-guide.md`
- **Agent Specifications**: See `docs/agent-specifications.md`

---

## Support

For issues or questions:

1. Check the troubleshooting guide
2. Review the API reference
3. Examine the integration patterns
4. Check agent logs in `Shadow/manus_archive/`

