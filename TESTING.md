# Testing Guide for KLRfome

## Quick Start

To run tests, use the provided test runner script:

```bash
./run_tests.sh
```

Or manually:

```bash
# Activate virtual environment
source venv/bin/activate

# Install package in editable mode (if not already installed)
pip install -e .

# Install test dependencies (if not already installed)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=klrfome --cov-report=html --cov-report=term
```

## Common Issues

### "ERROR: file or directory not found: python"

This error typically occurs when:
1. pytest is run from outside the virtual environment
2. The package is not installed in the virtual environment
3. pytest-cov is having issues finding the Python executable

**Solution:**
- Always activate the virtual environment first: `source venv/bin/activate`
- Install the package: `pip install -e .`
- Use `python -m pytest` instead of just `pytest`

### "ModuleNotFoundError: No module named 'jax'"

The virtual environment doesn't have dependencies installed.

**Solution:**
```bash
source venv/bin/activate
pip install -e .
```

### "ModuleNotFoundError: No module named 'klrfome'"

The package is not installed in editable mode.

**Solution:**
```bash
source venv/bin/activate
pip install -e .
```

## Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_kernels.py -v

# Run a specific test function
pytest tests/test_kernels.py::test_rbf_kernel_properties -v

# Run tests matching a pattern
pytest tests/ -k "kernel" -v
```

## Coverage Reports

After running tests with coverage, view the HTML report:

```bash
# Generate coverage report
pytest tests/ --cov=klrfome --cov-report=html

# Open the report (macOS)
open htmlcov/index.html
```

## Test Structure

- `tests/conftest.py` - Shared fixtures
- `tests/test_kernels.py` - Kernel implementation tests
- `tests/test_klr.py` - Kernel Logistic Regression tests
- `tests/test_prediction.py` - Focal prediction tests
- `tests/test_data.py` - Data structure tests
- `tests/test_integration.py` - End-to-end workflow tests

