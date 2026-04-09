# backend/monitoring/performance_profiler.py - Performance Profiling & Analysis
# Provides performance monitoring, profiling, and optimization recommendations

import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import statistics


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    name: str
    duration_ms: float
    timestamp: float
    agent_id: Optional[str] = None
    operation_type: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class PerformanceReport:
    """Aggregated performance report."""
    metric_name: str
    total_calls: int
    total_time_ms: float
    average_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    error_rate_percent: float
    p95_time_ms: float
    p99_time_ms: float


class PerformanceProfiler:
    """Profiles and analyzes system performance."""
    
    def __init__(self, archive_path: str = "Shadow/manus_archive"):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[PerformanceMetric] = []
        self.profiles: Dict[str, List[PerformanceMetric]] = {}
    
    def start_measurement(self, name: str) -> 'MeasurementContext':
        """Start a performance measurement."""
        return MeasurementContext(self, name)
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        self.metrics.append(metric)
        
        # Add to profile
        if metric.name not in self.profiles:
            self.profiles[metric.name] = []
        self.profiles[metric.name].append(metric)
    
    def get_report(self, metric_name: str) -> Optional[PerformanceReport]:
        """Get performance report for a metric."""
        
        if metric_name not in self.profiles:
            return None
        
        metrics = self.profiles[metric_name]
        
        if not metrics:
            return None
        
        durations = [m.duration_ms for m in metrics]
        errors = sum(1 for m in metrics if not m.success)
        
        # Sort for percentile calculations
        sorted_durations = sorted(durations)
        
        # Calculate percentiles
        p95_idx = int(len(sorted_durations) * 0.95)
        p99_idx = int(len(sorted_durations) * 0.99)
        
        p95 = sorted_durations[p95_idx] if p95_idx < len(sorted_durations) else sorted_durations[-1]
        p99 = sorted_durations[p99_idx] if p99_idx < len(sorted_durations) else sorted_durations[-1]
        
        return PerformanceReport(
            metric_name=metric_name,
            total_calls=len(metrics),
            total_time_ms=sum(durations),
            average_time_ms=statistics.mean(durations),
            median_time_ms=statistics.median(durations),
            min_time_ms=min(durations),
            max_time_ms=max(durations),
            std_dev_ms=statistics.stdev(durations) if len(durations) > 1 else 0,
            error_rate_percent=(errors / len(metrics) * 100) if metrics else 0,
            p95_time_ms=p95,
            p99_time_ms=p99
        )
    
    def get_all_reports(self) -> Dict[str, PerformanceReport]:
        """Get reports for all metrics."""
        reports = {}
        for metric_name in self.profiles.keys():
            report = self.get_report(metric_name)
            if report:
                reports[metric_name] = report
        return reports
    
    def get_slowest_operations(self, limit: int = 10) -> List[PerformanceMetric]:
        """Get slowest operations."""
        sorted_metrics = sorted(self.metrics, key=lambda m: m.duration_ms, reverse=True)
        return sorted_metrics[:limit]
    
    def get_failed_operations(self) -> List[PerformanceMetric]:
        """Get failed operations."""
        return [m for m in self.metrics if not m.success]
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent."""
        agent_metrics = [m for m in self.metrics if m.agent_id == agent_id]
        
        if not agent_metrics:
            return {}
        
        durations = [m.duration_ms for m in agent_metrics]
        errors = sum(1 for m in agent_metrics if not m.success)
        
        return {
            'agent_id': agent_id,
            'total_operations': len(agent_metrics),
            'total_time_ms': sum(durations),
            'average_time_ms': statistics.mean(durations),
            'min_time_ms': min(durations),
            'max_time_ms': max(durations),
            'error_count': errors,
            'error_rate_percent': (errors / len(agent_metrics) * 100) if agent_metrics else 0
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on metrics."""
        recommendations = []
        
        reports = self.get_all_reports()
        
        for metric_name, report in reports.items():
            # Check for high average response time
            if report.average_time_ms > 1000:
                recommendations.append(
                    f"⚠️  {metric_name}: Average response time is {report.average_time_ms:.0f}ms. "
                    f"Consider optimization or caching."
                )
            
            # Check for high error rate
            if report.error_rate_percent > 5:
                recommendations.append(
                    f"🔴 {metric_name}: Error rate is {report.error_rate_percent:.1f}%. "
                    f"Investigate and fix errors."
                )
            
            # Check for high variance
            if report.std_dev_ms > report.average_time_ms:
                recommendations.append(
                    f"📊 {metric_name}: High variance in response times. "
                    f"Consider load balancing or resource optimization."
                )
            
            # Check for slow operations
            if report.p99_time_ms > report.average_time_ms * 5:
                recommendations.append(
                    f"⏱️  {metric_name}: P99 response time is {report.p99_time_ms:.0f}ms. "
                    f"Investigate outliers and optimize."
                )
        
        return recommendations
    
    def save_report(self, filename: str = "performance_report.json"):
        """Save performance report to file."""
        reports = self.get_all_reports()
        
        report_data = {
            'generated_at': datetime.utcnow().isoformat(),
            'metrics': {
                name: asdict(report)
                for name, report in reports.items()
            },
            'recommendations': self.get_optimization_recommendations(),
            'total_metrics_recorded': len(self.metrics)
        }
        
        filepath = self.archive_path / filename
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(filepath)
    
    def clear_metrics(self):
        """Clear all recorded metrics."""
        self.metrics = []
        self.profiles = {}


class MeasurementContext:
    """Context manager for performance measurements."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.agent_id = None
        self.operation_type = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        metric = PerformanceMetric(
            name=self.name,
            duration_ms=duration_ms,
            timestamp=time.time(),
            agent_id=self.agent_id,
            operation_type=self.operation_type,
            success=exc_type is None,
            error=str(exc_val) if exc_val else None
        )
        
        self.profiler.record_metric(metric)


# ============================================================================
# GLOBAL PROFILER INSTANCE
# ============================================================================

_profiler: Optional[PerformanceProfiler] = None

def get_profiler() -> PerformanceProfiler:
    """Get or create the global profiler."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def profile_operation(name: str) -> MeasurementContext:
    """Profile an operation."""
    profiler = get_profiler()
    return profiler.start_measurement(name)


def get_performance_report(metric_name: str) -> Optional[PerformanceReport]:
    """Get performance report for a metric."""
    profiler = get_profiler()
    return profiler.get_report(metric_name)


def get_all_performance_reports() -> Dict[str, PerformanceReport]:
    """Get all performance reports."""
    profiler = get_profiler()
    return profiler.get_all_reports()


def get_slowest_operations(limit: int = 10) -> List[PerformanceMetric]:
    """Get slowest operations."""
    profiler = get_profiler()
    return profiler.get_slowest_operations(limit)


def get_optimization_recommendations() -> List[str]:
    """Get optimization recommendations."""
    profiler = get_profiler()
    return profiler.get_optimization_recommendations()


def save_performance_report(filename: str = "performance_report.json") -> str:
    """Save performance report."""
    profiler = get_profiler()
    return profiler.save_report(filename)
