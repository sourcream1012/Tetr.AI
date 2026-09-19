#!/usr/bin/env bash
set -e

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Installing dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Tetr.AI setup complete!"
echo "Activate the environment with:"
echo "source .venv/bin/activate"