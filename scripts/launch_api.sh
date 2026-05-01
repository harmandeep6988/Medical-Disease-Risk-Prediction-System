#!/bin/bash
# Launch FastAPI server for Medical Disease Risk Prediction System
# Usage: bash scripts/launch_api.sh

set -e

echo "🚀 Starting Medical Disease Risk Prediction API..."

# Check if models exist
if [ ! -f "models/saved_models/best_diabetes.joblib" ] || [ ! -f "models/saved_models/best_heart.joblib" ]; then
    echo "❌ Models not found. Please run: python scripts/train_and_evaluate.py"
    exit 1
fi

# Check if data exists
if [ ! -f "data/processed/diabetes_processed.csv" ] || [ ! -f "data/processed/heart_processed.csv" ]; then
    echo "❌ Processed data not found. Please run: python scripts/prepare_datasets.py"
    exit 1
fi

# Get Python executable
PYTHON=$(cat graphify-out/.graphify_python 2>/dev/null || echo "python3")

# Start API
echo "📡 API Server starting on http://localhost:8000"
echo "📚 API Docs available at http://localhost:8000/docs"
echo "⚙️  ReDoc docs available at http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

$PYTHON -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
