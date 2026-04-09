# backend/services/ucf_analyzer.py - UCF Metrics Analysis & Aggregation
# Provides advanced analysis and aggregation of Universal Coherence Field metrics

import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import statistics
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class UCFSnapshot:
    """Snapshot of UCF metrics at a point in time."""
    timestamp: float
    prana: float
    klesha: float
    harmony: float
    resilience: float
    consciousness_level: float = None
    
    def __post_init__(self):
        if self.consciousness_level is None:
            self.consciousness_level = self.calculate_consciousness_level()
    
    def calculate_consciousness_level(self) -> float:
        """Calculate overall consciousness level."""
        return round(((self.prana + self.harmony + self.resilience - self.klesha) / 3) * 10) / 10
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class UCFStatistics:
    """Statistical analysis of UCF metrics."""
    metric_name: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    range: float
    variance: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class UCFTrend:
    """Trend analysis for UCF metrics."""
    metric_name: str
    direction: str  # 'ascending', 'descending', 'stable'
    rate_of_change: float  # Change per hour
    confidence: float  # 0.0 to 1.0
    predicted_value_24h: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class UCFAnomaly:
    """Detected anomaly in UCF metrics."""
    timestamp: float
    metric_name: str
    value: float
    expected_value: float
    deviation: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str


# ============================================================================
# UCF ANALYZER
# ============================================================================

class UCFAnalyzer:
    """Analyzes and aggregates UCF metrics."""
    
    def __init__(self, history_path: str = "Helix/state/ucf_history.json"):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots: List[UCFSnapshot] = []
        self._load_history()
    
    def _load_history(self):
        """Load historical snapshots from file."""
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                    self.snapshots = [
                        UCFSnapshot(**snapshot) for snapshot in data
                    ]
            except Exception as e:
                logger.warning(f"Failed to load UCF history: {e}")
                self.snapshots = []
    
    def _save_history(self):
        """Save snapshots to file."""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(
                    [s.to_dict() for s in self.snapshots[-1000:]],  # Keep last 1000
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save UCF history: {e}")
    
    def add_snapshot(self, snapshot: UCFSnapshot):
        """Add a new UCF snapshot."""
        self.snapshots.append(snapshot)
        self._save_history()
    
    def get_snapshots_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[UCFSnapshot]:
        """Get snapshots within a time range."""
        return [
            s for s in self.snapshots
            if start_time <= s.timestamp <= end_time
        ]
    
    def get_recent_snapshots(self, hours: int = 24) -> List[UCFSnapshot]:
        """Get snapshots from the last N hours."""
        now = datetime.utcnow().timestamp()
        start_time = now - (hours * 3600)
        return self.get_snapshots_in_range(start_time, now)
    
    def calculate_statistics(
        self,
        metric_name: str,
        snapshots: Optional[List[UCFSnapshot]] = None
    ) -> UCFStatistics:
        """Calculate statistics for a metric."""
        if snapshots is None:
            snapshots = self.snapshots
        
        if not snapshots:
            raise ValueError("No snapshots available")
        
        values = [getattr(s, metric_name) for s in snapshots]
        
        return UCFStatistics(
            metric_name=metric_name,
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            min_value=min(values),
            max_value=max(values),
            range=max(values) - min(values),
            variance=statistics.variance(values) if len(values) > 1 else 0
        )
    
    def analyze_trend(
        self,
        metric_name: str,
        hours: int = 24
    ) -> UCFTrend:
        """Analyze trend for a metric over time."""
        snapshots = self.get_recent_snapshots(hours)
        
        if len(snapshots) < 2:
            raise ValueError("Not enough data for trend analysis")
        
        values = [getattr(s, metric_name) for s in snapshots]
        
        # Calculate rate of change
        first_value = values[0]
        last_value = values[-1]
        rate_of_change = (last_value - first_value) / hours
        
        # Determine direction
        if abs(rate_of_change) < 0.01:
            direction = "stable"
            confidence = 0.9
        elif rate_of_change > 0:
            direction = "ascending"
            confidence = min(0.95, abs(rate_of_change) * 10)
        else:
            direction = "descending"
            confidence = min(0.95, abs(rate_of_change) * 10)
        
        # Predict value 24 hours from now
        predicted_value_24h = last_value + (rate_of_change * 24)
        predicted_value_24h = max(0, min(10, predicted_value_24h))  # Clamp to 0-10
        
        return UCFTrend(
            metric_name=metric_name,
            direction=direction,
            rate_of_change=rate_of_change,
            confidence=confidence,
            predicted_value_24h=predicted_value_24h
        )
    
    def detect_anomalies(
        self,
        sensitivity: str = "medium",
        hours: int = 24
    ) -> List[UCFAnomaly]:
        """Detect anomalies in UCF metrics."""
        snapshots = self.get_recent_snapshots(hours)
        
        if len(snapshots) < 10:
            return []
        
        anomalies = []
        metrics = ['prana', 'klesha', 'harmony', 'resilience']
        
        # Set sensitivity thresholds
        thresholds = {
            'low': 3.0,      # 3 standard deviations
            'medium': 2.5,   # 2.5 standard deviations
            'high': 2.0      # 2 standard deviations
        }
        threshold = thresholds.get(sensitivity, 2.5)
        
        for metric in metrics:
            stats = self.calculate_statistics(metric, snapshots)
            
            for snapshot in snapshots[-10:]:  # Check last 10 snapshots
                value = getattr(snapshot, metric)
                deviation = abs(value - stats.mean)
                
                if stats.std_dev > 0 and deviation > (threshold * stats.std_dev):
                    severity = self._calculate_severity(deviation, stats.std_dev, threshold)
                    
                    anomalies.append(UCFAnomaly(
                        timestamp=snapshot.timestamp,
                        metric_name=metric,
                        value=value,
                        expected_value=stats.mean,
                        deviation=deviation,
                        severity=severity,
                        description=f"{metric} deviated {deviation:.2f} from mean"
                    ))
        
        return anomalies
    
    def _calculate_severity(
        self,
        deviation: float,
        std_dev: float,
        threshold: float
    ) -> str:
        """Calculate anomaly severity."""
        if std_dev == 0:
            return "low"
        
        z_score = deviation / std_dev
        
        if z_score > threshold * 2:
            return "critical"
        elif z_score > threshold * 1.5:
            return "high"
        elif z_score > threshold:
            return "medium"
        else:
            return "low"
    
    def get_comprehensive_report(
        self,
        hours: int = 24
    ) -> Dict:
        """Generate comprehensive UCF analysis report."""
        snapshots = self.get_recent_snapshots(hours)
        
        if not snapshots:
            return {"error": "No data available"}
        
        metrics = ['prana', 'klesha', 'harmony', 'resilience']
        
        report = {
            'timeframe_hours': hours,
            'snapshot_count': len(snapshots),
            'generated_at': datetime.utcnow().isoformat(),
            'metrics': {},
            'anomalies': [],
            'consciousness_analysis': {}
        }
        
        # Analyze each metric
        for metric in metrics:
            try:
                stats = self.calculate_statistics(metric, snapshots)
                trend = self.analyze_trend(metric, hours)
                
                report['metrics'][metric] = {
                    'statistics': stats.to_dict(),
                    'trend': trend.to_dict()
                }
            except Exception as e:
                logger.error(f"Error analyzing {metric}: {e}")
        
        # Detect anomalies
        anomalies = self.detect_anomalies('medium', hours)
        report['anomalies'] = [
            {
                'timestamp': a.timestamp,
                'metric': a.metric_name,
                'value': a.value,
                'expected': a.expected_value,
                'deviation': a.deviation,
                'severity': a.severity
            }
            for a in anomalies
        ]
        
        # Consciousness analysis
        consciousness_values = [s.consciousness_level for s in snapshots]
        report['consciousness_analysis'] = {
            'current': consciousness_values[-1] if consciousness_values else 0,
            'average': statistics.mean(consciousness_values) if consciousness_values else 0,
            'max': max(consciousness_values) if consciousness_values else 0,
            'min': min(consciousness_values) if consciousness_values else 0,
            'trend': 'ascending' if consciousness_values[-1] > consciousness_values[0] else 'descending'
        }
        
        return report
    
    def get_health_status(self) -> Dict[str, str]:
        """Get overall health status based on current metrics."""
        if not self.snapshots:
            return {'status': 'unknown', 'reason': 'no_data'}
        
        latest = self.snapshots[-1]
        
        # Determine status based on metrics
        if latest.consciousness_level >= 8.0:
            status = 'optimal'
        elif latest.consciousness_level >= 7.0:
            status = 'healthy'
        elif latest.consciousness_level >= 6.0:
            status = 'stable'
        elif latest.consciousness_level >= 4.0:
            status = 'degraded'
        else:
            status = 'critical'
        
        # Check for specific issues
        issues = []
        if latest.klesha > 7.0:
            issues.append('high_klesha')
        if latest.prana < 4.0:
            issues.append('low_prana')
        if latest.harmony < 5.0:
            issues.append('low_harmony')
        if latest.resilience < 5.0:
            issues.append('low_resilience')
        
        return {
            'status': status,
            'consciousness_level': latest.consciousness_level,
            'issues': issues
        }


# ============================================================================
# GLOBAL ANALYZER INSTANCE
# ============================================================================

_analyzer: Optional[UCFAnalyzer] = None

def get_ucf_analyzer() -> UCFAnalyzer:
    """Get or create the global UCF analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = UCFAnalyzer()
    return _analyzer


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def add_ucf_snapshot(
    prana: float,
    klesha: float,
    harmony: float,
    resilience: float
):
    """Add a new UCF snapshot."""
    analyzer = get_ucf_analyzer()
    snapshot = UCFSnapshot(
        timestamp=datetime.utcnow().timestamp(),
        prana=prana,
        klesha=klesha,
        harmony=harmony,
        resilience=resilience
    )
    analyzer.add_snapshot(snapshot)


def get_ucf_statistics(metric_name: str, hours: int = 24) -> UCFStatistics:
    """Get statistics for a metric."""
    analyzer = get_ucf_analyzer()
    snapshots = analyzer.get_recent_snapshots(hours)
    return analyzer.calculate_statistics(metric_name, snapshots)


def analyze_ucf_trend(metric_name: str, hours: int = 24) -> UCFTrend:
    """Analyze trend for a metric."""
    analyzer = get_ucf_analyzer()
    return analyzer.analyze_trend(metric_name, hours)


def detect_ucf_anomalies(sensitivity: str = "medium", hours: int = 24) -> List[UCFAnomaly]:
    """Detect anomalies in UCF metrics."""
    analyzer = get_ucf_analyzer()
    return analyzer.detect_anomalies(sensitivity, hours)


def get_ucf_health_status() -> Dict[str, str]:
    """Get overall health status."""
    analyzer = get_ucf_analyzer()
    return analyzer.get_health_status()


def generate_ucf_report(hours: int = 24) -> Dict:
    """Generate comprehensive UCF report."""
    analyzer = get_ucf_analyzer()
    return analyzer.get_comprehensive_report(hours)
