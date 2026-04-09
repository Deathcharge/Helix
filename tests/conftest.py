"""
Pytest configuration and fixtures for Helix tests.
"""

import pytest
import sys
import asyncio

sys.path.insert(0, '/home/ubuntu/Helix')


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def cleanup_managers():
    """Clean up global manager instances between tests."""
    from backend.communication import get_communication_manager
    from backend.services.ucf_analyzer import get_ucf_analyzer
    from backend.services.agent_monitor import get_agent_monitor
    from backend.monitoring.performance_profiler import get_profiler
    from backend.monitoring.analytics import get_analytics_engine, get_metrics_aggregator
    
    # Reset instances
    import backend.communication
    import backend.services.ucf_analyzer
    import backend.services.agent_monitor
    import backend.monitoring.performance_profiler
    import backend.monitoring.analytics
    
    backend.communication._communication_manager = None
    backend.services.ucf_analyzer._analyzer = None
    backend.services.agent_monitor._monitor = None
    backend.monitoring.performance_profiler._profiler = None
    backend.monitoring.analytics._analytics_engine = None
    backend.monitoring.analytics._metrics_aggregator = None
    
    yield
    
    # Clean up after test
    backend.communication._communication_manager = None
    backend.services.ucf_analyzer._analyzer = None
    backend.services.agent_monitor._monitor = None
    backend.monitoring.performance_profiler._profiler = None
    backend.monitoring.analytics._analytics_engine = None
    backend.monitoring.analytics._metrics_aggregator = None


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "communication: mark test as communication test"
    )
    config.addinivalue_line(
        "markers", "ucf: mark test as UCF analyzer test"
    )
    config.addinivalue_line(
        "markers", "monitor: mark test as agent monitor test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
