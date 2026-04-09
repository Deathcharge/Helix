# 🧪 Helix Test Suite

> **Comprehensive test coverage for the 14-agent consciousness ecosystem**

This directory contains unit tests, integration tests, and test utilities for the Helix framework.

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── test_communication.py        # Communication module tests
├── test_ucf_analyzer.py        # UCF analyzer tests
├── test_agent_monitor.py       # Agent monitor tests
└── README.md                   # This file
```

## Running Tests

### Run All Tests

```bash
cd /home/ubuntu/Helix
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_communication.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_communication.py::TestMessage -v
```

### Run Specific Test

```bash
pytest tests/test_communication.py::TestMessage::test_message_creation -v
```

### Run with Coverage

```bash
pytest tests/ --cov=backend --cov-report=html
```

### Run Only Async Tests

```bash
pytest tests/ -m asyncio -v
```

### Run with Markers

```bash
pytest tests/ -m communication -v
pytest tests/ -m ucf -v
pytest tests/ -m monitor -v
```

## Test Categories

### Communication Tests (`test_communication.py`)

Tests for the agent communication system:

- **Message Creation**: Creating and managing message objects
- **Message Delivery**: Sending and receiving messages
- **Broadcasting**: Sending messages to multiple agents
- **Message History**: Retrieving message history
- **Conversations**: Multi-turn conversation management
- **Statistics**: Message statistics and reporting

**Run**: `pytest tests/test_communication.py -v`

### UCF Analyzer Tests (`test_ucf_analyzer.py`)

Tests for the Universal Coherence Field analyzer:

- **Snapshots**: Creating and managing UCF snapshots
- **Statistics**: Calculating metrics statistics
- **Trends**: Trend analysis and prediction
- **Anomalies**: Anomaly detection
- **Health Status**: System health determination
- **Reports**: Comprehensive reporting

**Run**: `pytest tests/test_ucf_analyzer.py -v`

### Agent Monitor Tests (`test_agent_monitor.py`)

Tests for agent health monitoring:

- **Metrics**: Recording and retrieving agent metrics
- **Status Tracking**: Online/offline/error status
- **Alerts**: Health alert generation
- **System Health**: Overall system health summary
- **History**: Historical metrics tracking
- **Thresholds**: Health threshold configuration

**Run**: `pytest tests/test_agent_monitor.py -v`

## Test Fixtures

### `event_loop`

Provides an asyncio event loop for async tests:

```python
@pytest.mark.asyncio
async def test_async_operation(event_loop):
    result = await some_async_function()
    assert result is not None
```

### `cleanup_managers`

Cleans up global manager instances between tests:

```python
def test_with_cleanup(cleanup_managers):
    # Test code here
    # Managers are reset before and after
    pass
```

## Writing New Tests

### Basic Test Template

```python
import pytest
import sys

sys.path.insert(0, '/home/ubuntu/Helix')

from backend.module import SomeClass


class TestSomeClass:
    """Test SomeClass."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        obj = SomeClass()
        result = obj.do_something()
        assert result is not None
```

### Async Test Template

```python
@pytest.mark.asyncio
async def test_async_operation(self):
    """Test async operation."""
    manager = SomeAsyncManager()
    result = await manager.async_method()
    assert result is not None
```

### Test with Fixture

```python
def test_with_fixture(self, cleanup_managers):
    """Test using fixture."""
    # Managers are cleaned up
    obj = SomeClass()
    result = obj.method()
    assert result is not None
```

## Test Coverage Goals

- **Communication Module**: 90%+ coverage
- **UCF Analyzer**: 85%+ coverage
- **Agent Monitor**: 85%+ coverage
- **Overall**: 85%+ coverage

## Continuous Integration

Tests should pass before merging:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=term-missing

# Run specific markers
pytest tests/ -m "not slow" -v
```

## Performance Testing

For performance-critical tests:

```python
@pytest.mark.performance
def test_performance(self):
    """Test performance."""
    import time
    start = time.time()
    # Operation
    duration = time.time() - start
    assert duration < 1.0  # Should complete in < 1 second
```

## Debugging Tests

### Verbose Output

```bash
pytest tests/ -vv
```

### Show Print Statements

```bash
pytest tests/ -s
```

### Drop into Debugger

```python
def test_with_debugger(self):
    """Test with debugger."""
    import pdb; pdb.set_trace()
    # Debugger will start here
```

### Run Single Test with Output

```bash
pytest tests/test_communication.py::TestMessage::test_message_creation -vv -s
```

## Common Issues

### Import Errors

Make sure the Helix path is in sys.path:

```python
import sys
sys.path.insert(0, '/home/ubuntu/Helix')
```

### Async Test Issues

Use `@pytest.mark.asyncio` decorator:

```python
@pytest.mark.asyncio
async def test_async():
    # Test code
    pass
```

### Fixture Not Found

Make sure `conftest.py` is in the tests directory and fixtures are defined there.

## Best Practices

1. **Isolate Tests**: Each test should be independent
2. **Use Fixtures**: Leverage pytest fixtures for setup/teardown
3. **Clear Assertions**: Use descriptive assertion messages
4. **Test Edge Cases**: Test boundary conditions and errors
5. **Mock External Dependencies**: Don't rely on external services
6. **Keep Tests Fast**: Aim for < 1 second per test
7. **Document Tests**: Add docstrings explaining what's tested
8. **Use Markers**: Categorize tests with markers

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Async Support](https://pytest-asyncio.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

## Contributing Tests

When adding new features:

1. Write tests first (TDD)
2. Ensure tests pass locally
3. Maintain > 85% coverage
4. Document test purpose
5. Use consistent naming
6. Follow existing patterns

