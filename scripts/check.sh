#!/bin/bash

# Code quality check script (no modifications)
# This script checks code quality without making any changes

set -e  # Exit on any error

echo "🔍 Running code quality checks (no modifications)..."

# Change to project root
cd "$(dirname "$0")/.."

# Check import sorting
echo "  📦 Checking import order with isort..."
uv run isort backend/ --check-only --diff

# Check code formatting
echo "  🖤 Checking code formatting with black..."
uv run black backend/ --check --diff

# Run flake8 (style and error checking)
echo "  📏 Checking code style with flake8..."
uv run flake8 backend/

# Note: mypy type checking available but disabled for now
# Uncomment to enable: uv run mypy backend/ --ignore-missing-imports

echo "✅ All quality checks passed!"