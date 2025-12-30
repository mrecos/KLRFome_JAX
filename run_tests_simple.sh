#!/bin/bash
# Simple test runner - use this if run_tests.sh has issues
# This script explicitly uses the venv Python

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: venv Python not found at $VENV_PYTHON"
    echo "Please ensure your virtual environment is set up correctly."
    exit 1
fi

echo "Using Python: $VENV_PYTHON"
echo "Python version: $($VENV_PYTHON --version)"
echo ""

# Check if klrfome is importable
echo "Checking klrfome installation..."
$VENV_PYTHON -c "import klrfome; print('✓ klrfome imported successfully')" || {
    echo "✗ klrfome not found. Installing..."
    $VENV_PYTHON -m pip install -e . --quiet
}

# Check if pytest is available
echo "Checking pytest..."
$VENV_PYTHON -c "import pytest; print('✓ pytest available')" || {
    echo "✗ pytest not found. Installing dev dependencies..."
    $VENV_PYTHON -m pip install -e ".[dev]" --quiet
}

# Run tests using the venv Python explicitly
echo ""
echo "Running tests..."
echo ""

$VENV_PYTHON -m pytest tests/ -v

echo ""
echo "✓ Tests completed!"

