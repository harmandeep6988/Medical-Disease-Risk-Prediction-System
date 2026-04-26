#!/bin/bash

# Docker Setup & Deployment Script
# Handles Docker build, permissions, and container management

set -e

PROJECT_DIR="/home/blaze/important/project.ml/Medical-Disease-Risk-Prediction-System"
IMAGE_NAME="medical-risk-predictor"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 1. Check Docker installation
print_header "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
fi
DOCKER_VERSION=$(docker --version 2>/dev/null || echo "Docker CLI unavailable")
print_success "Docker found: $DOCKER_VERSION"

# 2. Check Docker daemon
print_header "Checking Docker daemon..."
if ! docker ps &> /dev/null; then
    print_warning "Docker daemon may not be accessible. Attempting to fix permissions..."
    if id -nG "$USER" | grep -qw docker; then
        print_success "User is in docker group. Please start a new shell session:"
        echo "  exec newgrp docker"
        exit 1
    else
        print_error "User not in docker group. Run: sudo usermod -aG docker \$USER"
    fi
fi
print_success "Docker daemon is accessible"

# 3. Navigate to project
print_header "Setting up project directory..."
if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory not found: $PROJECT_DIR"
fi
cd "$PROJECT_DIR"
print_success "Working in: $(pwd)"

# 4. Show build options
print_header "Docker Build Options"
echo ""
echo "1. Build Docker image (standard)"
echo "2. Build with no cache (force rebuild)"
echo "3. Build with verbose output"
echo "4. Skip build and use existing image"
echo ""
read -p "Choose option (1-4) [1]: " BUILD_OPTION
BUILD_OPTION=${BUILD_OPTION:-1}

case $BUILD_OPTION in
    2)
        print_header "Building Docker image (--no-cache)..."
        docker build --no-cache -t "$FULL_IMAGE_NAME" .
        ;;
    3)
        print_header "Building Docker image (verbose)..."
        docker build -t "$FULL_IMAGE_NAME" . --progress=plain
        ;;
    4)
        print_header "Skipping build, checking for existing image..."
        if ! docker images | grep -q "$IMAGE_NAME"; then
            print_error "Image $FULL_IMAGE_NAME not found. Please build first."
        fi
        print_success "Using existing image: $FULL_IMAGE_NAME"
        ;;
    *)
        print_header "Building Docker image..."
        docker build -t "$FULL_IMAGE_NAME" .
        ;;
esac

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed"
fi

# 5. Show image info
echo ""
print_header "Docker Image Information"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# 6. Deploy options
echo ""
print_header "Deployment Options"
echo ""
echo "1. Start Dashboard (Streamlit UI) - http://localhost:8501"
echo "2. Run Full Training Pipeline (data prep + train)"
echo "3. Run Data Preparation Only (Kaggle fetch)"
echo "4. Run Training Only"
echo "5. Start Jupyter Lab - http://localhost:8888"
echo "6. Interactive Shell (bash)"
echo "7. Exit"
echo ""
read -p "Choose deployment option (1-7) [1]: " DEPLOY_OPTION
DEPLOY_OPTION=${DEPLOY_OPTION:-1}

case $DEPLOY_OPTION in
    1)
        print_header "Launching Dashboard..."
        docker-compose up dashboard
        ;;
    2)
        print_header "Running Full Training Pipeline..."
        docker-compose --profile pipeline up full-pipeline
        ;;
    3)
        print_header "Preparing Data with Kaggle Augmentation..."
        docker-compose --profile data-prep up prepare-data
        ;;
    4)
        print_header "Running Training Only..."
        docker-compose --profile training up training
        ;;
    5)
        print_header "Launching Jupyter Lab..."
        print_warning "Access: http://localhost:8888"
        print_warning "Check container logs for token: docker-compose logs notebook"
        docker-compose --profile notebook up notebook
        ;;
    6)
        print_header "Opening Interactive Shell..."
        docker-compose run --rm dashboard bash
        ;;
    7)
        print_success "Exiting setup script"
        exit 0
        ;;
    *)
        print_error "Invalid option"
        ;;
esac

print_success "Docker operation completed"
