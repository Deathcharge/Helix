# backend/monitoring/analytics.py - Analytics & Metrics Aggregation
# Provides comprehensive analytics and metrics aggregation for system analysis

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict
import statistics


class AnalyticsEngine:
    """Aggregates and analyzes system metrics."""
    
    def __init__(self, archive_path: str = "Shadow/manus_archive"):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        self.events: List[Dict[str, Any]] = []
        self.event_types: Dict[str, int] = defaultdict(int)
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_event(
        self,
        event_type: str,
        agent_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """Record an event."""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'agent_id': agent_id,
            'data': data or {}
        }
        
        self.events.append(event)
        self.event_types[event_type] += 1
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of recorded events."""
        total_events = len(self.events)
        
        return {
            'total_events': total_events,
            'event_types': dict(self.event_types),
            'unique_agents': len(set(e['agent_id'] for e in self.events if e['agent_id'])),
            'date_range': {
                'start': self.events[0]['timestamp'] if self.events else None,
                'end': self.events[-1]['timestamp'] if self.events else None
            }
        }
    
    def get_agent_analytics(self, agent_id: str) -> Dict[str, Any]:
        """Get analytics for a specific agent."""
        agent_events = [e for e in self.events if e['agent_id'] == agent_id]
        
        if not agent_events:
            return {}
        
        event_types = defaultdict(int)
        for event in agent_events:
            event_types[event['type']] += 1
        
        return {
            'agent_id': agent_id,
            'total_events': len(agent_events),
            'event_types': dict(event_types),
            'first_event': agent_events[0]['timestamp'],
            'last_event': agent_events[-1]['timestamp']
        }
    
    def get_time_series_data(
        self,
        event_type: Optional[str] = None,
        bucket_size_minutes: int = 5
    ) -> Dict[str, int]:
        """Get time series data for events."""
        events = self.events
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        
        if not events:
            return {}
        
        # Create time buckets
        buckets = defaultdict(int)
        
        for event in events:
            timestamp = datetime.fromisoformat(event['timestamp'])
            bucket_time = timestamp.replace(
                minute=(timestamp.minute // bucket_size_minutes) * bucket_size_minutes,
                second=0,
                microsecond=0
            )
            bucket_key = bucket_time.isoformat()
            buckets[bucket_key] += 1
        
        return dict(sorted(buckets.items()))
    
    def get_top_agents(self, limit: int = 10) -> List[tuple]:
        """Get agents with most events."""
        agent_counts = defaultdict(int)
        
        for event in self.events:
            if event['agent_id']:
                agent_counts[event['agent_id']] += 1
        
        sorted_agents = sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_agents[:limit]
    
    def get_event_type_distribution(self) -> Dict[str, float]:
        """Get distribution of event types."""
        total = len(self.events)
        
        if total == 0:
            return {}
        
        distribution = {}
        for event_type, count in self.event_types.items():
            distribution[event_type] = (count / total) * 100
        
        return distribution
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'event_summary': self.get_event_summary(),
            'event_distribution': self.get_event_type_distribution(),
            'top_agents': self.get_top_agents(10),
            'agent_analytics': {
                agent_id: self.get_agent_analytics(agent_id)
                for agent_id in set(e['agent_id'] for e in self.events if e['agent_id'])
            }
        }
        
        return report
    
    def save_analytics_report(self, filename: str = "analytics_report.json") -> str:
        """Save analytics report to file."""
        report = self.generate_analytics_report()
        
        filepath = self.archive_path / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(filepath)


class MetricsAggregator:
    """Aggregates metrics from multiple sources."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timestamps: Dict[str, List[float]] = defaultdict(list)
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value."""
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        
        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(timestamp)
    
    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics.get(metric_name, [])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'sum': sum(values)
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        
        for metric_name in self.metrics.keys():
            summary[metric_name] = self.get_metric_statistics(metric_name)
        
        return summary
    
    def get_metric_trend(self, metric_name: str) -> str:
        """Get trend for a metric (ascending/descending/stable)."""
        values = self.metrics.get(metric_name, [])
        
        if len(values) < 2:
            return 'unknown'
        
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        diff = second_half - first_half
        
        if abs(diff) < 0.1 * first_half:
            return 'stable'
        elif diff > 0:
            return 'ascending'
        else:
            return 'descending'


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_analytics_engine: Optional[AnalyticsEngine] = None
_metrics_aggregator: Optional[MetricsAggregator] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get or create the global analytics engine."""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    return _analytics_engine


def get_metrics_aggregator() -> MetricsAggregator:
    """Get or create the global metrics aggregator."""
    global _metrics_aggregator
    if _metrics_aggregator is None:
        _metrics_aggregator = MetricsAggregator()
    return _metrics_aggregator


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def record_event(
    event_type: str,
    agent_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
):
    """Record an event."""
    engine = get_analytics_engine()
    engine.record_event(event_type, agent_id, data)


def record_metric(metric_name: str, value: float):
    """Record a metric value."""
    aggregator = get_metrics_aggregator()
    aggregator.record_metric(metric_name, value)


def get_analytics_report() -> Dict[str, Any]:
    """Get analytics report."""
    engine = get_analytics_engine()
    return engine.generate_analytics_report()


def get_metrics_summary() -> Dict[str, Dict[str, float]]:
    """Get metrics summary."""
    aggregator = get_metrics_aggregator()
    return aggregator.get_all_metrics_summary()
