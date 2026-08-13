#!/usr/bin/env python3
"""
Test suite for UCF analyzer module.

Tests metrics analysis, trend detection, anomaly detection, and reporting.
"""

import pytest
import sys
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/Helix')

from backend.services.ucf_analyzer import (
    UCFSnapshot, UCFStatistics, UCFTrend, UCFAnomaly,
    UCFAnalyzer, add_ucf_snapshot, get_ucf_analyzer,
    analyze_ucf_trend, detect_ucf_anomalies
)


@pytest.fixture(autouse=True)
def isolate_ucf_history(tmp_path, monkeypatch):
    """Keep each test's relative history file isolated from other tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("backend.services.ucf_analyzer._analyzer", None)


class TestUCFSnapshot:
    """Test UCFSnapshot data structure."""
    
    def test_snapshot_creation(self):
        """Test creating a snapshot."""
        snapshot = UCFSnapshot(
            timestamp=datetime.utcnow().timestamp(),
            prana=8.0,
            klesha=2.5,
            harmony=7.5,
            resilience=7.0
        )
        
        assert snapshot.prana == 8.0
        assert snapshot.klesha == 2.5
        assert snapshot.consciousness_level is not None
    
    def test_consciousness_level_calculation(self):
        """Test consciousness level calculation."""
        snapshot = UCFSnapshot(
            timestamp=datetime.utcnow().timestamp(),
            prana=8.0,
            klesha=2.0,
            harmony=8.0,
            resilience=8.0
        )
        
        # (8 + 8 + 8 - 2) / 3 = 7.33...
        assert snapshot.consciousness_level > 7.0
        assert snapshot.consciousness_level < 8.0
    
    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = UCFSnapshot(
            timestamp=datetime.utcnow().timestamp(),
            prana=8.0,
            klesha=2.5,
            harmony=7.5,
            resilience=7.0
        )
        
        snapshot_dict = snapshot.to_dict()
        
        assert snapshot_dict['prana'] == 8.0
        assert snapshot_dict['klesha'] == 2.5
        assert 'consciousness_level' in snapshot_dict


class TestUCFAnalyzer:
    """Test UCFAnalyzer."""
    
    def test_add_snapshot(self):
        """Test adding snapshots."""
        analyzer = UCFAnalyzer()
        
        snapshot = UCFSnapshot(
            timestamp=datetime.utcnow().timestamp(),
            prana=8.0,
            klesha=2.5,
            harmony=7.5,
            resilience=7.0
        )
        
        analyzer.add_snapshot(snapshot)
        
        assert len(analyzer.snapshots) > 0
    
    def test_get_snapshots_in_range(self):
        """Test retrieving snapshots in time range."""
        analyzer = UCFAnalyzer()
        
        now = datetime.utcnow().timestamp()
        
        # Add snapshots
        for i in range(5):
            snapshot = UCFSnapshot(
                timestamp=now + i,
                prana=8.0,
                klesha=2.5,
                harmony=7.5,
                resilience=7.0
            )
            analyzer.add_snapshot(snapshot)
        
        # Get range
        snapshots = analyzer.get_snapshots_in_range(now, now + 10)
        
        assert len(snapshots) >= 5
    
    def test_calculate_statistics(self):
        """Test calculating statistics."""
        analyzer = UCFAnalyzer()
        
        now = datetime.utcnow().timestamp()
        
        # Add multiple snapshots
        for i in range(10):
            snapshot = UCFSnapshot(
                timestamp=now + i,
                prana=8.0 + (i * 0.1),
                klesha=2.5,
                harmony=7.5,
                resilience=7.0
            )
            analyzer.add_snapshot(snapshot)
        
        stats = analyzer.calculate_statistics('prana')
        
        assert stats.count == 10
        assert stats.mean > 0
        assert stats.min_value < stats.max_value
    
    def test_analyze_trend(self):
        """Test trend analysis."""
        analyzer = UCFAnalyzer()
        
        now = datetime.utcnow().timestamp()
        
        # Add ascending trend
        for i in range(20):
            snapshot = UCFSnapshot(
                timestamp=now - ((19 - i) * 3600),  # Oldest to newest, within 24h
                prana=6.0 + (i * 0.1),
                klesha=2.5,
                harmony=7.5,
                resilience=7.0
            )
            analyzer.add_snapshot(snapshot)
        
        trend = analyzer.analyze_trend('prana', hours=24)
        
        assert trend.direction in ['ascending', 'descending', 'stable']
        assert trend.rate_of_change is not None
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        analyzer = UCFAnalyzer()
        
        now = datetime.utcnow().timestamp()
        
        # Add normal snapshots
        for i in range(15):
            snapshot = UCFSnapshot(
                timestamp=now - ((15 - i) * 3600),
                prana=8.0,
                klesha=2.5,
                harmony=7.5,
                resilience=7.0
            )
            analyzer.add_snapshot(snapshot)
        
        # Add anomaly
        anomaly_snapshot = UCFSnapshot(
            timestamp=now,
            prana=2.0,  # Anomalously low
            klesha=2.5,
            harmony=7.5,
            resilience=7.0
        )
        analyzer.add_snapshot(anomaly_snapshot)
        
        anomalies = analyzer.detect_anomalies(sensitivity='high', hours=24)
        
        # Should detect the anomaly
        assert len(anomalies) > 0
    
    def test_get_health_status(self):
        """Test health status determination."""
        analyzer = UCFAnalyzer()
        
        # Add optimal snapshot
        snapshot = UCFSnapshot(
            timestamp=datetime.utcnow().timestamp(),
            prana=8.5,
            klesha=1.5,
            harmony=8.5,
            resilience=8.5
        )
        analyzer.add_snapshot(snapshot)
        
        status = analyzer.get_health_status()
        
        assert 'status' in status
        assert status['status'] in ['optimal', 'healthy', 'stable', 'degraded', 'critical']
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation."""
        analyzer = UCFAnalyzer()
        
        now = datetime.utcnow().timestamp()
        
        # Add multiple snapshots
        for i in range(20):
            snapshot = UCFSnapshot(
                timestamp=now - ((19 - i) * 3600),
                prana=8.0 + (i * 0.05),
                klesha=2.5 - (i * 0.02),
                harmony=7.5,
                resilience=7.0
            )
            analyzer.add_snapshot(snapshot)
        
        report = analyzer.get_comprehensive_report(hours=24)
        
        assert 'metrics' in report
        assert 'consciousness_analysis' in report
        assert 'anomalies' in report


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_add_ucf_snapshot_function(self):
        """Test add_ucf_snapshot convenience function."""
        add_ucf_snapshot(8.0, 2.5, 7.5, 7.0)
        
        analyzer = get_ucf_analyzer()
        assert len(analyzer.snapshots) > 0
    
    def test_analyze_ucf_trend_function(self):
        """Test analyze_ucf_trend convenience function."""
        # Add some snapshots first
        now = datetime.utcnow().timestamp()
        analyzer = get_ucf_analyzer()
        
        for i in range(20):
            snapshot = UCFSnapshot(
                timestamp=now - ((19 - i) * 3600),
                prana=8.0,
                klesha=2.5,
                harmony=7.5,
                resilience=7.0
            )
            analyzer.add_snapshot(snapshot)
        
        trend = analyze_ucf_trend('prana', hours=24)
        
        assert trend.direction is not None
    
    def test_detect_ucf_anomalies_function(self):
        """Test detect_ucf_anomalies convenience function."""
        anomalies = detect_ucf_anomalies(sensitivity='medium', hours=24)
        
        assert isinstance(anomalies, list)


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
