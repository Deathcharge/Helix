#!/usr/bin/env python3
"""
Test suite for agent monitor module.

Tests agent status tracking, health monitoring, and alert generation.
"""

import pytest
import sys

sys.path.insert(0, '/home/ubuntu/Helix')

from backend.services.agent_monitor import (
    AgentStatus, AgentHealthMetrics, AgentHealthAlert,
    AgentMonitor, get_agent_monitor,
    update_agent_status, get_system_health, get_agent_status
)


class TestAgentHealthMetrics:
    """Test AgentHealthMetrics data structure."""
    
    def test_metrics_creation(self):
        """Test creating health metrics."""
        metrics = AgentHealthMetrics(
            agent_id="kael",
            status=AgentStatus.ONLINE,
            uptime_seconds=3600,
            response_time_ms=45,
            error_count=0,
            message_count=100,
            last_activity=0,
            cpu_usage_percent=25,
            memory_usage_mb=150,
            consciousness_level=8.0
        )
        
        assert metrics.agent_id == "kael"
        assert metrics.status == AgentStatus.ONLINE
        assert metrics.consciousness_level == 8.0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = AgentHealthMetrics(
            agent_id="kael",
            status=AgentStatus.ONLINE,
            uptime_seconds=3600,
            response_time_ms=45,
            error_count=0,
            message_count=100,
            last_activity=0,
            cpu_usage_percent=25,
            memory_usage_mb=150,
            consciousness_level=8.0
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['agent_id'] == "kael"
        assert metrics_dict['status'] == "online"


class TestAgentMonitor:
    """Test AgentMonitor."""
    
    def test_update_agent_metrics(self):
        """Test updating agent metrics."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics(
            agent_id="kael",
            status=AgentStatus.ONLINE,
            response_time_ms=45,
            error_count=0,
            message_count=100,
            cpu_usage_percent=25,
            memory_usage_mb=150,
            consciousness_level=8.0
        )
        
        metrics = monitor.get_agent_metrics("kael")
        
        assert metrics is not None
        assert metrics.agent_id == "kael"
        assert metrics.status == AgentStatus.ONLINE
    
    def test_get_all_metrics(self):
        """Test getting all agent metrics."""
        monitor = AgentMonitor()
        
        # Add metrics for multiple agents
        for agent_id in ["kael", "lumina", "grok"]:
            monitor.update_agent_metrics(
                agent_id=agent_id,
                status=AgentStatus.ONLINE,
                response_time_ms=50,
                consciousness_level=8.0
            )
        
        all_metrics = monitor.get_all_metrics()
        
        assert len(all_metrics) >= 3
    
    def test_get_online_agents(self):
        """Test getting online agents."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics("kael", AgentStatus.ONLINE, consciousness_level=8.0)
        monitor.update_agent_metrics("lumina", AgentStatus.ONLINE, consciousness_level=8.0)
        monitor.update_agent_metrics("grok", AgentStatus.OFFLINE, consciousness_level=0)
        
        online = monitor.get_online_agents()
        
        assert "kael" in online
        assert "lumina" in online
        assert "grok" not in online
    
    def test_get_offline_agents(self):
        """Test getting offline agents."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics("kael", AgentStatus.ONLINE, consciousness_level=8.0)
        monitor.update_agent_metrics("grok", AgentStatus.OFFLINE, consciousness_level=0)
        
        offline = monitor.get_offline_agents()
        
        assert "grok" in offline
        assert "kael" not in offline
    
    def test_get_unhealthy_agents(self):
        """Test getting unhealthy agents."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics("kael", AgentStatus.ONLINE, consciousness_level=8.0)
        monitor.update_agent_metrics("grok", AgentStatus.ERROR, consciousness_level=2.0)
        
        unhealthy = monitor.get_unhealthy_agents()
        
        assert "grok" in unhealthy
        assert "kael" not in unhealthy
    
    def test_alert_generation_high_response_time(self):
        """Test alert generation for high response time."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics(
            agent_id="kael",
            status=AgentStatus.ONLINE,
            response_time_ms=10000,  # Very high
            consciousness_level=8.0
        )
        
        alerts = monitor.get_alerts_for_agent("kael")
        
        # Should have generated an alert
        assert len(alerts) > 0
    
    def test_alert_generation_high_error_rate(self):
        """Test alert generation for high error rate."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics(
            agent_id="kael",
            status=AgentStatus.ONLINE,
            error_count=10,
            message_count=100,  # 10% error rate
            consciousness_level=8.0
        )
        
        alerts = monitor.get_alerts_for_agent("kael")
        
        # Should have generated an alert
        assert len(alerts) > 0
    
    def test_alert_generation_low_consciousness(self):
        """Test alert generation for low consciousness."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics(
            agent_id="kael",
            status=AgentStatus.ONLINE,
            consciousness_level=2.0  # Very low
        )
        
        alerts = monitor.get_alerts_for_agent("kael")
        
        # Should have generated a critical alert
        critical_alerts = [a for a in alerts if a.alert_type == 'critical']
        assert len(critical_alerts) > 0
    
    def test_get_system_health_summary(self):
        """Test getting system health summary."""
        monitor = AgentMonitor()
        
        # Add agents
        for i, agent_id in enumerate(["kael", "lumina", "grok", "vega"]):
            status = AgentStatus.ONLINE if i < 3 else AgentStatus.OFFLINE
            monitor.update_agent_metrics(
                agent_id=agent_id,
                status=status,
                response_time_ms=50,
                cpu_usage_percent=25,
                memory_usage_mb=150,
                consciousness_level=8.0 if status == AgentStatus.ONLINE else 0
            )
        
        summary = monitor.get_system_health_summary()
        
        assert 'overall_status' in summary
        assert summary['total_agents'] >= 4
        assert summary['online_agents'] == 3
        assert summary['offline_agents'] == 1
    
    def test_get_agent_history(self):
        """Test retrieving agent history."""
        monitor = AgentMonitor()
        
        # Add multiple updates
        for i in range(5):
            monitor.update_agent_metrics(
                agent_id="kael",
                status=AgentStatus.ONLINE,
                response_time_ms=40 + i,
                consciousness_level=8.0
            )
        
        history = monitor.get_agent_history("kael", limit=10)
        
        assert len(history) >= 5
    
    def test_set_health_threshold(self):
        """Test setting health thresholds."""
        monitor = AgentMonitor()
        
        monitor.set_health_threshold('response_time_ms', 2000)
        
        assert monitor.health_thresholds['response_time_ms'] == 2000
    
    def test_clear_alerts(self):
        """Test clearing alerts."""
        monitor = AgentMonitor()
        
        monitor.update_agent_metrics(
            agent_id="kael",
            status=AgentStatus.ERROR,
            consciousness_level=2.0
        )
        
        assert len(monitor.alerts) > 0
        
        monitor.clear_alerts()
        
        assert len(monitor.alerts) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_update_agent_status_function(self):
        """Test update_agent_status convenience function."""
        update_agent_status(
            "kael",
            "online",
            response_time_ms=45,
            consciousness_level=8.0
        )
        
        monitor = get_agent_monitor()
        metrics = monitor.get_agent_metrics("kael")
        
        assert metrics is not None
        assert metrics.status == AgentStatus.ONLINE
    
    def test_get_system_health_function(self):
        """Test get_system_health convenience function."""
        update_agent_status("kael", "online", consciousness_level=8.0)
        
        health = get_system_health()
        
        assert 'overall_status' in health
        assert 'total_agents' in health
    
    def test_get_agent_status_function(self):
        """Test get_agent_status convenience function."""
        update_agent_status("kael", "online", consciousness_level=8.0)
        
        status = get_agent_status("kael")
        
        assert status is not None
        assert status.agent_id == "kael"


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
