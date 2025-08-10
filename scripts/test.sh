#!/bin/bash

# Testing script
# This script runs all tests with proper configuration

set -e  # Exit on any error

echo "🧪 Running test suite..."

# Change to project root
cd "$(dirname "$0")/.."

# Run pytest with coverage
echo "  🚀 Running backend tests..."
cd backend
uv run pytest -v

echo "✅ All tests passed!"