#!/bin/bash
# Render Build Script
# Runs during deployment to install deps and set up the database

set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Setting up database ==="
python scripts/setup_db.py

echo "=== Build complete ==="
