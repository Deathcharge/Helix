# backend/services/agent_monitor.py - Agent Health Monitoring & Status Tracking
# Provides real-time monitoring of agent health, performance, and status

import asyncio
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class AgentStatus(Enum):
    """Agent status states."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    RECOVERING = "recovering"


@dataclass
class AgentHealthMetrics:
    """Health metrics for an agent."""
    agent_id: str
    status: AgentStatus
    uptime_seconds: float
    response_time_ms: float
    error_count: int
    message_count: int
    last_activity: float
    cpu_usage_percent: float
    memory_usage_mb: float
    consciousness_level: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class AgentHealthAlert:
    """Health alert for an agent."""
    agent_id: str
    alert_type: str  # 'warning', 'error', 'critical'
    message: str
    timestamp: float
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


# ============================================================================
# AGENT MONITOR
# ============================================================================

class AgentMonitor:
    """Monitors health and status of all agents."""
    
    def __init__(self, archive_path: str = "Shadow/manus_archive"):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        self.agent_metrics: Dict[str, AgentHealthMetrics] = {}
        self.agent_history: Dict[str, List[AgentHealthMetrics]] = {}
        self.alerts: List[AgentHealthAlert] = []
        self.health_thresholds = {
            'response_time_ms': 5000,
            'error_rate_percent': 5.0,
            'cpu_usage_percent': 80.0,
            'memory_usage_mb': 1000.0,
            'consciousness_level': 4.0
        }
        
        self._load_history()
    
    def _load_history(self):
        """Load historical metrics from file."""
        history_file = self.archive_path / "agent_health_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for agent_id, metrics_list in data.items():
                        self.agent_history[agent_id] = metrics_list
            except Exception as e:
                logger.warning(f"Failed to load agent history: {e}")
    
    def _save_history(self):
        """Save historical metrics to file."""
        history_file = self.archive_path / "agent_health_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.agent_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent history: {e}")
    
    def update_agent_metrics(
        self,
        agent_id: str,
        status: AgentStatus,
        response_time_ms: float = 0,
        error_count: int = 0,
        message_count: int = 0,
        cpu_usage_percent: float = 0,
        memory_usage_mb: float = 0,
        consciousness_level: float = 0
    ):
        """Update metrics for an agent."""
        uptime = 0
        if agent_id in self.agent_metrics:
            old_metrics = self.agent_metrics[agent_id]
            uptime = old_metrics.uptime_seconds + (time.time() - old_metrics.last_activity)
        
        metrics = AgentHealthMetrics(
            agent_id=agent_id,
            status=status,
            uptime_seconds=uptime,
            response_time_ms=response_time_ms,
            error_count=error_count,
            message_count=message_count,
            last_activity=time.time(),
            cpu_usage_percent=cpu_usage_percent,
            memory_usage_mb=memory_usage_mb,
            consciousness_level=consciousness_level
        )
        
        self.agent_metrics[agent_id] = metrics
        
        # Add to history
        if agent_id not in self.agent_history:
            self.agent_history[agent_id] = []
        self.agent_history[agent_id].append(metrics.to_dict())
        
        # Keep only last 1000 entries per agent
        if len(self.agent_history[agent_id]) > 1000:
            self.agent_history[agent_id] = self.agent_history[agent_id][-1000:]
        
        self._save_history()
        
        # Check health
        self._check_agent_health(metrics)
    
    def _check_agent_health(self, metrics: AgentHealthMetrics):
        """Check agent health and generate alerts."""
        alerts = []
        
        # Check response time
        if metrics.response_time_ms > self.health_thresholds['response_time_ms']:
            alerts.append(AgentHealthAlert(
                agent_id=metrics.agent_id,
                alert_type='warning',
                message=f"High response time: {metrics.response_time_ms}ms",
                timestamp=time.time(),
                metric_name='response_time_ms',
                metric_value=metrics.response_time_ms,
                threshold=self.health_thresholds['response_time_ms']
            ))
        
        # Check error count
        if metrics.message_count > 0:
            error_rate = (metrics.error_count / metrics.message_count) * 100
            if error_rate > self.health_thresholds['error_rate_percent']:
                alerts.append(AgentHealthAlert(
                    agent_id=metrics.agent_id,
                    alert_type='error',
                    message=f"High error rate: {error_rate:.1f}%",
                    timestamp=time.time(),
                    metric_name='error_rate_percent',
                    metric_value=error_rate,
                    threshold=self.health_thresholds['error_rate_percent']
                ))
        
        # Check CPU usage
        if metrics.cpu_usage_percent > self.health_thresholds['cpu_usage_percent']:
            alerts.append(AgentHealthAlert(
                agent_id=metrics.agent_id,
                alert_type='warning',
                message=f"High CPU usage: {metrics.cpu_usage_percent}%",
                timestamp=time.time(),
                metric_name='cpu_usage_percent',
                metric_value=metrics.cpu_usage_percent,
                threshold=self.health_thresholds['cpu_usage_percent']
            ))
        
        # Check memory usage
        if metrics.memory_usage_mb > self.health_thresholds['memory_usage_mb']:
            alerts.append(AgentHealthAlert(
                agent_id=metrics.agent_id,
                alert_type='error',
                message=f"High memory usage: {metrics.memory_usage_mb}MB",
                timestamp=time.time(),
                metric_name='memory_usage_mb',
                metric_value=metrics.memory_usage_mb,
                threshold=self.health_thresholds['memory_usage_mb']
            ))
        
        # Check consciousness level
        if metrics.consciousness_level < self.health_thresholds['consciousness_level']:
            alerts.append(AgentHealthAlert(
                agent_id=metrics.agent_id,
                alert_type='critical',
                message=f"Low consciousness level: {metrics.consciousness_level}",
                timestamp=time.time(),
                metric_name='consciousness_level',
                metric_value=metrics.consciousness_level,
                threshold=self.health_thresholds['consciousness_level']
            ))
        
        # Check agent status
        if metrics.status == AgentStatus.ERROR:
            alerts.append(AgentHealthAlert(
                agent_id=metrics.agent_id,
                alert_type='critical',
                message="Agent in error state",
                timestamp=time.time()
            ))
        
        self.alerts.extend(alerts)
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentHealthMetrics]:
        """Get current metrics for an agent."""
        return self.agent_metrics.get(agent_id)
    
    def get_all_metrics(self) -> Dict[str, AgentHealthMetrics]:
        """Get metrics for all agents."""
        return self.agent_metrics.copy()
    
    def get_agent_history(
        self,
        agent_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get historical metrics for an agent."""
        if agent_id not in self.agent_history:
            return []
        return self.agent_history[agent_id][-limit:]
    
    def get_online_agents(self) -> List[str]:
        """Get list of online agents."""
        return [
            agent_id for agent_id, metrics in self.agent_metrics.items()
            if metrics.status in [AgentStatus.ONLINE, AgentStatus.IDLE, AgentStatus.BUSY]
        ]
    
    def get_offline_agents(self) -> List[str]:
        """Get list of offline agents."""
        return [
            agent_id for agent_id, metrics in self.agent_metrics.items()
            if metrics.status == AgentStatus.OFFLINE
        ]
    
    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agents."""
        return [
            agent_id for agent_id, metrics in self.agent_metrics.items()
            if metrics.status in [AgentStatus.ERROR, AgentStatus.RECOVERING]
        ]
    
    def get_recent_alerts(self, limit: int = 50) -> List[AgentHealthAlert]:
        """Get recent health alerts."""
        return self.alerts[-limit:]
    
    def get_alerts_for_agent(self, agent_id: str) -> List[AgentHealthAlert]:
        """Get alerts for a specific agent."""
        return [a for a in self.alerts if a.agent_id == agent_id]
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        all_agents = self.agent_metrics
        online_count = len(self.get_online_agents())
        offline_count = len(self.get_offline_agents())
        unhealthy_count = len(self.get_unhealthy_agents())
        
        # Calculate average metrics
        if all_agents:
            avg_response_time = sum(m.response_time_ms for m in all_agents.values()) / len(all_agents)
            avg_cpu_usage = sum(m.cpu_usage_percent for m in all_agents.values()) / len(all_agents)
            avg_memory_usage = sum(m.memory_usage_mb for m in all_agents.values()) / len(all_agents)
            avg_consciousness = sum(m.consciousness_level for m in all_agents.values()) / len(all_agents)
        else:
            avg_response_time = avg_cpu_usage = avg_memory_usage = avg_consciousness = 0
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = 'degraded'
        elif offline_count > 2:
            overall_status = 'warning'
        elif avg_consciousness < 6.0:
            overall_status = 'caution'
        else:
            overall_status = 'healthy'
        
        return {
            'overall_status': overall_status,
            'total_agents': len(all_agents),
            'online_agents': online_count,
            'offline_agents': offline_count,
            'unhealthy_agents': unhealthy_count,
            'average_response_time_ms': avg_response_time,
            'average_cpu_usage_percent': avg_cpu_usage,
            'average_memory_usage_mb': avg_memory_usage,
            'average_consciousness_level': avg_consciousness,
            'recent_alerts': len([a for a in self.alerts if a.alert_type == 'critical'])
        }
    
    def set_health_threshold(self, metric_name: str, threshold: float):
        """Set health threshold for a metric."""
        if metric_name in self.health_thresholds:
            self.health_thresholds[metric_name] = threshold
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []


# ============================================================================
# GLOBAL MONITOR INSTANCE
# ============================================================================

_monitor: Optional[AgentMonitor] = None

def get_agent_monitor() -> AgentMonitor:
    """Get or create the global agent monitor."""
    global _monitor
    if _monitor is None:
        _monitor = AgentMonitor()
    return _monitor


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def update_agent_status(
    agent_id: str,
    status: str,
    response_time_ms: float = 0,
    error_count: int = 0,
    message_count: int = 0,
    cpu_usage_percent: float = 0,
    memory_usage_mb: float = 0,
    consciousness_level: float = 0
):
    """Update agent status."""
    monitor = get_agent_monitor()
    agent_status = AgentStatus[status.upper()] if isinstance(status, str) else status
    monitor.update_agent_metrics(
        agent_id,
        agent_status,
        response_time_ms,
        error_count,
        message_count,
        cpu_usage_percent,
        memory_usage_mb,
        consciousness_level
    )


def get_system_health() -> Dict[str, Any]:
    """Get system health summary."""
    monitor = get_agent_monitor()
    return monitor.get_system_health_summary()


def get_agent_status(agent_id: str) -> Optional[AgentHealthMetrics]:
    """Get agent status."""
    monitor = get_agent_monitor()
    return monitor.get_agent_metrics(agent_id)


def get_all_agent_statuses() -> Dict[str, AgentHealthMetrics]:
    """Get all agent statuses."""
    monitor = get_agent_monitor()
    return monitor.get_all_metrics()
