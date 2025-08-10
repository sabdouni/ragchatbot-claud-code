#!/bin/bash

# Code formatting script
# This script runs all code formatting tools on the backend codebase

set -e  # Exit on any error

echo "🎨 Running code formatters..."

# Change to project root
cd "$(dirname "$0")/.."

# Run isort (import sorting)
echo "  📦 Sorting imports with isort..."
uv run isort backend/

# Run black (code formatting)
echo "  🖤 Formatting code with black..."
uv run black backend/

echo "✅ Code formatting complete!"