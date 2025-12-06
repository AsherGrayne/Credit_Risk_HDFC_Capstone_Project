# Testing and NFR Implementation Guide

## Overview

This document provides a comprehensive guide for implementing testing and Non-Functional Requirements (NFRs) for the Credit Card Delinquency Prediction System.

## Quick Start

### 1. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install production dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only

# Run tests in parallel (faster)
pytest -n auto
```

### 3. View Coverage Report

After running tests with coverage, open `htmlcov/index.html` in your browser to see the coverage report.

## Test Structure

```
tests/
├── __init__.py
├── test_feature_engineering.py    # Unit tests for feature engineering
├── test_risk_flags.py              # Unit tests for risk flag generation
├── test_model_training.py          # Unit tests for model training
├── test_integration.py             # Integration tests
└── test_performance.py             # Performance tests
```

## Key Features Implemented

### 1. Testing Framework
- ✅ Unit tests for core functionality
- ✅ Integration tests for end-to-end workflow
- ✅ Performance tests for scalability
- ✅ Test coverage reporting

### 2. Logging
- ✅ Structured logging with levels
- ✅ File and console logging
- ✅ Configurable log levels

### 3. Security
- ✅ Input validation
- ✅ Rate limiting
- ✅ API key authentication (framework)
- ✅ Data sanitization

### 4. Monitoring
- ✅ Metrics collection
- ✅ Performance tracking
- ✅ Health check endpoints

### 5. Configuration Management
- ✅ Environment-based configuration
- ✅ Centralized config management

### 6. Batch Processing
- ✅ Large dataset processing
- ✅ Progress tracking
- ✅ Error handling

## Usage Examples

### Using Logging

```python
from src.logging_config import get_logger

logger = get_logger(__name__)

logger.info("Processing started")
logger.error("An error occurred", exc_info=True)
```

### Using Security Validation

```python
from src.security import SecurityValidator

# Validate customer ID
if not SecurityValidator.validate_customer_id(customer_id):
    raise ValueError("Invalid customer ID")

# Validate feature inputs
is_valid, error = SecurityValidator.validate_feature_inputs(features)
if not is_valid:
    raise ValueError(error)
```

### Using Monitoring

```python
from src.monitoring import track_performance, metrics_collector

@track_performance("model_prediction")
def predict_risk(features):
    # Your prediction code
    pass

# Get metrics
metrics = metrics_collector.get_all_metrics()
print(f"Total predictions: {metrics['counters']['model_prediction_success']}")
```

### Using Batch Processing

```python
from src.batch_processor import BatchProcessor

processor = BatchProcessor(batch_size=1000)

def process_batch(batch_df):
    # Your processing logic
    return processed_df

result = processor.process_in_batches(large_df, process_batch)
```

## Health Checks

Start the health check server:

```bash
python src/health_check.py
```

Endpoints:
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /health/detailed` - Detailed health check
- `GET /metrics` - System metrics

## Configuration

Set environment variables for configuration:

```bash
# Model configuration
export MODEL_N_ESTIMATORS=100
export MODEL_MAX_DEPTH=5

# Security
export API_KEY=your-secret-api-key
export RATE_LIMIT_REQUESTS=100

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=logs/system.log

# Performance
export BATCH_SIZE=1000
export MAX_WORKERS=4
```

## Continuous Integration

Example GitHub Actions workflow (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Next Steps

1. **Review Test Coverage**: Run `pytest --cov=src` and review coverage report
2. **Add More Tests**: Increase coverage for critical paths
3. **Set Up CI/CD**: Automate test execution
4. **Monitor Production**: Set up monitoring in production environment
5. **Performance Tuning**: Use performance tests to identify bottlenecks

## Resources

- [Testing Guide](docs/TESTING_AND_NFR_GUIDE.md) - Comprehensive testing guide
- [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) - Step-by-step implementation
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Logging Guide](https://docs.python.org/3/library/logging.html)

## Support

For questions or issues, please refer to the main project documentation or create an issue in the repository.

