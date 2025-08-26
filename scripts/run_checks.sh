#!/usr/bin/env bash
# Run formatting, linting, and tests before pushing.
set -euo pipefail

echo "Formatting code with Black"
black .

echo "Linting with flake8"
flake8 .

echo "Running tests"
pytest

echo "Validating installed packages"
python -m pip check
