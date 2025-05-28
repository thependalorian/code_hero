#!/bin/bash

# Code formatting script for Code Hero project
echo "🎨 Formatting Python code with Black and isort..."

# Format with Black
echo "📝 Running Black formatter..."
black src/ tests/ --line-length 88 --target-version py311

# Organize imports with isort
echo "📚 Organizing imports with isort..."
isort src/ tests/ --profile black --line-length 88

# Run syntax check
echo "🔍 Checking syntax..."
python -m py_compile src/code_hero/*.py

# Run tests to ensure everything works
echo "🧪 Running tests to verify formatting didn't break anything..."
python -m pytest tests/ -q

echo "✅ Code formatting complete!" 