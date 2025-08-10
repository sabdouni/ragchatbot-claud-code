#!/bin/bash

# Code linting script
# This script runs all linting tools on the backend codebase

set -e  # Exit on any error

echo "🔍 Running code quality checks..."

# Change to project root
cd "$(dirname "$0")/.."

# Run flake8 (style and error checking)
echo "  📏 Checking code style with flake8..."
uv run flake8 backend/

echo "✅ All linting checks passed!"