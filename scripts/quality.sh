#!/bin/bash

# Complete code quality script
# This script runs formatting, linting, and testing in sequence

set -e  # Exit on any error

echo "🚀 Running complete code quality pipeline..."

# Get script directory
SCRIPT_DIR="$(dirname "$0")"

# Run formatting
"$SCRIPT_DIR/format.sh"

# Run linting
"$SCRIPT_DIR/lint.sh"

# Run tests
"$SCRIPT_DIR/test.sh"

echo ""
echo "🎉 Code quality pipeline completed successfully!"
echo "   ✅ Formatting applied"
echo "   ✅ Linting passed"  
echo "   ✅ Tests passed"