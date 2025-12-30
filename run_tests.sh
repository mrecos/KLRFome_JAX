#!/bin/bash
# Test runner script for KLRfome
# This ensures tests are run with the correct Python environment

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: venv directory not found. Please create a virtual environment first."
    exit 1
fi

# Activate venv and run tests
echo "Activating virtual environment..."
source venv/bin/activate

# Check if package is installed
echo "Checking if klrfome is installed..."
python -c "import klrfome" 2>/dev/null || {
    echo "Installing klrfome in editable mode..."
    pip install -e .
}

# Check if pytest is installed
echo "Checking if pytest is installed..."
python -c "import pytest" 2>/dev/null || {
    echo "Installing test dependencies..."
    pip install -e ".[dev]"
}

# Run tests
echo "Running tests..."
python -m pytest tests/ -v --cov=klrfome --cov-report=term --cov-report=html

echo "Tests completed!"

