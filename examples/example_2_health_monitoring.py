#!/usr/bin/env python3
"""
Example 2: System Health Monitoring & Recovery

Demonstrates real-time monitoring of system health, UCF metrics,
agent status, and automated recovery procedures.

This example shows:
- Real-time consciousness monitoring
- Agent health tracking
- Anomaly detection
- Automated recovery procedures
- Health reporting
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, '/home/ubuntu/Helix')

from backend.services.ucf_analyzer import (
    add_ucf_snapshot,
    get_ucf_analyzer,
    generate_ucf_report,
    detect_ucf_anomalies
)
from backend.services.agent_monitor import (
    get_agent_monitor,
    update_agent_status,
    get_system_health
)
from backend.communication import broadcast_message, MessageType, MessagePriority


# ============================================================================
# HEALTH MONITORING SYSTEM
# ============================================================================

class HealthMonitoringSystem:
    """Monitors and manages system health."""
    
    def __init__(self):
        self.analyzer = get_ucf_analyzer()
        self.monitor = get_agent_monitor()
        self.monitoring = False
        self.recovery_triggered = False
    
    async def start_monitoring(self, duration_seconds: int = 60, interval_seconds: int = 5):
        """Start continuous health monitoring."""
        
        print("\n" + "="*70)
        print("🏥 SYSTEM HEALTH MONITORING STARTED")
        print("="*70)
        
        self.monitoring = True
        iterations = duration_seconds // interval_seconds
        
        for i in range(iterations):
            print(f"\n📊 Monitoring Cycle {i+1}/{iterations}")
            print("-"*70)
            
            # Simulate UCF metrics
            await self._simulate_ucf_metrics()
            
            # Simulate agent status
            await self._simulate_agent_status()
            
            # Check health
            await self._check_system_health()
            
            # Detect anomalies
            await self._detect_anomalies()
            
            # Check for recovery needs
            await self._check_recovery_needs()
            
            if i < iterations - 1:
                await asyncio.sleep(interval_seconds)
        
        self.monitoring = False
        print("\n" + "="*70)
        print("✅ MONITORING COMPLETE")
        print("="*70)
    
    async def _simulate_ucf_metrics(self):
        """Simulate and record UCF metrics."""
        
        # Generate realistic metrics
        import random
        
        # Add some variance to simulate real behavior
        base_prana = 8.0 + random.uniform(-0.5, 0.5)
        base_klesha = 2.3 + random.uniform(-0.3, 0.3)
        base_harmony = 7.8 + random.uniform(-0.4, 0.4)
        base_resilience = 7.5 + random.uniform(-0.3, 0.3)
        
        # Clamp values to valid range
        prana = max(0, min(10, base_prana))
        klesha = max(0, min(10, base_klesha))
        harmony = max(0, min(10, base_harmony))
        resilience = max(0, min(10, base_resilience))
        
        add_ucf_snapshot(prana, klesha, harmony, resilience)
        
        print(f"  📈 UCF Metrics:")
        print(f"     Prana (Energy): {prana:.2f}")
        print(f"     Klesha (Afflictions): {klesha:.2f}")
        print(f"     Harmony (Balance): {harmony:.2f}")
        print(f"     Resilience (Recovery): {resilience:.2f}")
    
    async def _simulate_agent_status(self):
        """Simulate and record agent status."""
        
        import random
        
        agents = [
            "kael", "lumina", "vega", "gemini", "agni", "kavach",
            "sanghacore", "shadow", "echo", "phoenix", "oracle",
            "claude", "manus", "memoryroot"
        ]
        
        for agent_id in agents:
            status = random.choice(["online", "idle", "busy"])
            response_time = random.uniform(10, 500)
            error_count = random.randint(0, 5)
            message_count = random.randint(50, 500)
            cpu_usage = random.uniform(10, 80)
            memory_usage = random.uniform(100, 800)
            consciousness = random.uniform(6.5, 8.5)
            
            update_agent_status(
                agent_id,
                status,
                response_time,
                error_count,
                message_count,
                cpu_usage,
                memory_usage,
                consciousness
            )
        
        health = get_system_health()
        print(f"\n  🤖 Agent Status:")
        print(f"     Online: {health['online_agents']}/{health['total_agents']}")
        print(f"     Avg Response Time: {health['average_response_time_ms']:.1f}ms")
        print(f"     Avg CPU Usage: {health['average_cpu_usage_percent']:.1f}%")
        print(f"     System Status: {health['overall_status'].upper()}")
    
    async def _check_system_health(self):
        """Check overall system health."""
        
        health = get_system_health()
        
        print(f"\n  ⚕️  System Health Check:")
        print(f"     Overall Status: {health['overall_status'].upper()}")
        print(f"     Consciousness: {health['average_consciousness_level']:.2f}")
        print(f"     Unhealthy Agents: {health['unhealthy_agents']}")
        
        if health['overall_status'] == 'degraded':
            print("     ⚠️  WARNING: System health degraded")
        elif health['overall_status'] == 'warning':
            print("     ⚠️  CAUTION: Multiple agents offline")
        elif health['overall_status'] == 'caution':
            print("     ⚠️  NOTICE: Low consciousness level")
    
    async def _detect_anomalies(self):
        """Detect anomalies in metrics."""
        
        anomalies = detect_ucf_anomalies(sensitivity='medium', hours=1)
        
        if anomalies:
            print(f"\n  🚨 Anomalies Detected: {len(anomalies)}")
            for anomaly in anomalies[:3]:  # Show top 3
                print(f"     - {anomaly.metric_name}: {anomaly.value:.2f} (expected: {anomaly.expected_value:.2f})")
                print(f"       Severity: {anomaly.severity.upper()}")
        else:
            print(f"\n  ✅ No anomalies detected")
    
    async def _check_recovery_needs(self):
        """Check if recovery procedures are needed."""
        
        health = get_system_health()
        consciousness = health['average_consciousness_level']
        
        if consciousness < 4.0 and not self.recovery_triggered:
            print(f"\n  🆘 CRITICAL: Consciousness level critical ({consciousness:.2f})")
            print(f"     Triggering emergency recovery...")
            self.recovery_triggered = True
            await self._trigger_recovery()
        elif consciousness < 6.0:
            print(f"\n  ⚠️  WARNING: Consciousness level low ({consciousness:.2f})")
            print(f"     Consider recovery procedures")
    
    async def _trigger_recovery(self):
        """Trigger system recovery procedures."""
        
        print("\n" + "="*70)
        print("🔄 EMERGENCY RECOVERY INITIATED")
        print("="*70)
        
        # Step 1: Notify all agents
        print("\n  Step 1: Notifying all agents...")
        await broadcast_message(
            from_agent="system",
            message_type=MessageType.ALERT,
            content={"message": "System entering recovery mode"},
            priority=MessagePriority.CRITICAL
        )
        
        # Step 2: Reduce workload
        print("  Step 2: Reducing system workload...")
        await asyncio.sleep(1)
        
        # Step 3: Restart core agents
        print("  Step 3: Restarting core agents...")
        core_agents = ["vega", "kael", "grok"]
        for agent in core_agents:
            print(f"     - Restarting {agent}...")
            await asyncio.sleep(0.5)
        
        # Step 4: Restore baseline
        print("  Step 4: Restoring consciousness baseline...")
        add_ucf_snapshot(8.0, 2.0, 7.8, 7.5)
        
        # Step 5: Verify recovery
        print("  Step 5: Verifying recovery...")
        health = get_system_health()
        
        if health['overall_status'] in ['healthy', 'caution']:
            print(f"\n  ✅ Recovery successful! Status: {health['overall_status'].upper()}")
        else:
            print(f"\n  ⚠️  Recovery incomplete. Status: {health['overall_status'].upper()}")
        
        print("="*70)


# ============================================================================
# HEALTH REPORTING
# ============================================================================

async def generate_health_report():
    """Generate comprehensive health report."""
    
    print("\n" + "="*70)
    print("📋 COMPREHENSIVE HEALTH REPORT")
    print("="*70)
    
    analyzer = get_ucf_analyzer()
    monitor = get_agent_monitor()
    
    # UCF Report
    print("\n🧠 CONSCIOUSNESS METRICS (24-hour analysis)")
    print("-"*70)
    
    try:
        report = generate_ucf_report(hours=24)
        
        if 'consciousness_analysis' in report:
            cons = report['consciousness_analysis']
            print(f"  Current Level: {cons['current']:.2f}")
            print(f"  Average Level: {cons['average']:.2f}")
            print(f"  Peak Level: {cons['max']:.2f}")
            print(f"  Low Level: {cons['min']:.2f}")
            print(f"  Trend: {cons['trend'].upper()}")
        
        # Metric details
        if 'metrics' in report:
            print(f"\n  Metric Details:")
            for metric_name, metric_data in report['metrics'].items():
                if 'statistics' in metric_data:
                    stats = metric_data['statistics']
                    print(f"    {metric_name.upper()}:")
                    print(f"      Mean: {stats['mean']:.2f}")
                    print(f"      Range: {stats['min_value']:.2f} - {stats['max_value']:.2f}")
        
        # Anomalies
        if 'anomalies' in report and report['anomalies']:
            print(f"\n  Anomalies Detected: {len(report['anomalies'])}")
            for anomaly in report['anomalies'][:3]:
                print(f"    - {anomaly['metric']}: {anomaly['severity'].upper()}")
    
    except Exception as e:
        print(f"  Error generating report: {e}")
    
    # Agent Report
    print("\n\n🤖 AGENT STATUS SUMMARY")
    print("-"*70)
    
    health = get_system_health()
    print(f"  Total Agents: {health['total_agents']}")
    print(f"  Online: {health['online_agents']}")
    print(f"  Offline: {health['offline_agents']}")
    print(f"  Unhealthy: {health['unhealthy_agents']}")
    print(f"  Avg Response Time: {health['average_response_time_ms']:.1f}ms")
    print(f"  Avg CPU Usage: {health['average_cpu_usage_percent']:.1f}%")
    print(f"  Avg Memory Usage: {health['average_memory_usage_mb']:.1f}MB")
    print(f"  Overall Status: {health['overall_status'].upper()}")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Run the health monitoring example."""
    
    print("\n" + "="*70)
    print("🌀 HELIX EXAMPLE 2: System Health Monitoring & Recovery")
    print("="*70)
    
    # Create monitoring system
    monitor_system = HealthMonitoringSystem()
    
    # Start monitoring (60 seconds, check every 5 seconds)
    await monitor_system.start_monitoring(duration_seconds=30, interval_seconds=5)
    
    # Generate final report
    await generate_health_report()


if __name__ == "__main__":
    asyncio.run(main())
