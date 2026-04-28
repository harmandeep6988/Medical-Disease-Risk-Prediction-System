#!/bin/bash

# Docker entrypoint script for multiple entry points
# Allows flexible command execution

set -e

# Function to display usage
show_usage() {
    echo "Usage: docker-compose run [service] [command]"
    echo ""
    echo "Available commands:"
    echo "  dashboard              - Start Streamlit dashboard (default)"
    echo "  train                  - Run model training pipeline"
    echo "  prepare-data           - Prepare and fetch Kaggle datasets"
    echo "  full-pipeline          - Run complete workflow (data prep + train)"
    echo "  jupyter                - Start Jupyter Lab"
    echo "  bash                   - Open interactive bash shell"
    echo "  python <script>        - Run a Python script"
    echo ""
    echo "Examples:"
    echo "  docker-compose run training bash"
    echo "  docker-compose run training python -c 'import pandas; print(pandas.__version__)'"
    echo "  docker-compose run --rm training python medical_risk_predictor/scripts/train_and_evaluate.py"
    echo ""
}

# If no arguments, default to dashboard
if [ $# -eq 0 ]; then
    exec streamlit run medical_risk_predictor/dashboard/app.py
fi

# Handle different commands
case "$1" in
    dashboard)
        echo "Starting Streamlit dashboard..."
        exec streamlit run medical_risk_predictor/dashboard/app.py
        ;;
    train)
        echo "Running model training pipeline..."
        exec python medical_risk_predictor/scripts/train_and_evaluate.py
        ;;
    prepare-data)
        echo "Preparing datasets with Kaggle augmentation..."
        exec python medical_risk_predictor/scripts/prepare_datasets.py --include-kaggle
        ;;
    full-pipeline)
        echo "Running full pipeline (data prep + training)..."
        exec python medical_risk_predictor/scripts/run_all.py --include-kaggle --no-dashboard
        ;;
    jupyter)
        echo "Starting Jupyter Lab..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    bash|sh)
        echo "Opening interactive shell..."
        exec /bin/bash
        ;;
    python)
        # Pass through to python with remaining arguments
        shift
        exec python "$@"
        ;;
    help|-h|--help)
        show_usage
        ;;
    *)
        # Allow any command to pass through
        echo "Running command: $@"
        exec "$@"
        ;;
esac
