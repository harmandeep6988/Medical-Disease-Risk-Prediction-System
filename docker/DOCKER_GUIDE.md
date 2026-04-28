# Docker Integration Guide

## Overview

This project is fully containerized with Docker for reproducible deployment across environments. The setup includes:

- **Dockerfile:** Multi-stage build for optimized image size (~1.2GB with all ML libraries)
- **docker-compose.yml:** Orchestration with multiple service profiles
- **.dockerignore:** Excludes unnecessary files to keep build context small
- **docker-entrypoint.sh:** Flexible entry point for multiple commands

## Prerequisites

```bash
# Install Docker
# macOS / Windows: Download Docker Desktop from https://www.docker.com/products/docker-desktop
# Linux: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Verify installation
docker --version
docker-compose --version
```

## Quick Start

### 1. Build the Docker Image

```bash
cd /home/blaze/important/project.ml/Medical-Disease-Risk-Prediction-System

# Build the image
docker build -t medical-risk-predictor:latest .

# (Optional) Build with custom tag
docker build -t medical-risk-predictor:v1.0 .
```

**Build Time:** ~5-10 minutes (first time, depends on internet speed)  
**Image Size:** ~1.2GB (multi-stage optimization)

### 2. Launch Dashboard (Interactive)

#### Option A: Using docker-compose (Recommended)

```bash
# Start the dashboard service
docker-compose up dashboard

# Output:
#   You can now view your Streamlit app in your browser at:
#   http://localhost:8501

# Access in browser: http://localhost:8501
```

#### Option B: Using docker run (Direct)

```bash
docker run -it --rm \
  -p 8501:8501 \
  -v $(pwd)/medical_risk_predictor/data/raw:/app/medical_risk_predictor/data/raw \
  -v $(pwd)/medical_risk_predictor/models/saved_models:/app/medical_risk_predictor/models/saved_models \
  medical-risk-predictor:latest \
  streamlit run medical_risk_predictor/dashboard/app.py
```

### 3. Run Full Pipeline (Data Prep + Training)

```bash
# Option A: Using docker-compose with profile
docker-compose --profile pipeline up full-pipeline

# Option B: Using docker run
docker run -it --rm \
  -v $(pwd)/medical_risk_predictor/data/raw:/app/medical_risk_predictor/data/raw \
  -v $(pwd)/medical_risk_predictor/data/processed:/app/medical_risk_predictor/data/processed \
  -v $(pwd)/medical_risk_predictor/models/saved_models:/app/medical_risk_predictor/models/saved_models \
  -v ~/.kaggle:/home/mluser/.kaggle:ro \
  -e KAGGLE_CONFIG_DIR=/home/mluser/.kaggle \
  medical-risk-predictor:latest \
  python medical_risk_predictor/scripts/run_all.py --include-kaggle --no-dashboard
```

---

## Docker Services

### Service 1: Dashboard (Default)

**Purpose:** Interactive Streamlit UI for risk prediction  
**Port:** 8501  
**Restart Policy:** unless-stopped (persistent)

```bash
# Start
docker-compose up dashboard

# Run in background
docker-compose up -d dashboard

# View logs
docker-compose logs -f dashboard

# Stop
docker-compose down
```

### Service 2: Training

**Purpose:** Run model training on augmented datasets  
**Profile:** training  
**Exit Behavior:** One-shot (completes then exits)

```bash
# Run training
docker-compose --profile training up training

# View output
docker-compose logs training

# Access container shell (for debugging)
docker-compose --profile training run training bash
```

### Service 3: Data Preparation

**Purpose:** Fetch and prepare Kaggle datasets  
**Profile:** data-prep  
**Exit Behavior:** One-shot

```bash
# Prepare data with Kaggle augmentation
docker-compose --profile data-prep up prepare-data

# View output
docker-compose logs prepare-data
```

### Service 4: Full Pipeline

**Purpose:** Complete end-to-end workflow  
**Profile:** pipeline  
**Exit Behavior:** One-shot

```bash
# Run full pipeline (data prep → training → evaluation)
docker-compose --profile pipeline up full-pipeline

# Monitor progress
docker-compose logs -f full-pipeline
```

### Service 5: Jupyter Notebook

**Purpose:** Interactive exploration and experimentation  
**Port:** 8888  
**Profile:** notebook

```bash
# Start Jupyter Lab
docker-compose --profile notebook up notebook

# Access: http://localhost:8888
# Copy the token from logs: docker-compose logs notebook
```

---

## Common Docker Tasks

### Build Image with Custom Name/Tag

```bash
docker build -t my-medical-predictor:v1 .
```

### Run Interactive Shell

```bash
# Using docker-compose
docker-compose run --rm dashboard bash

# Using docker run
docker run -it --rm medical-risk-predictor:latest bash
```

### Check Image Size

```bash
docker images medical-risk-predictor
```

### Remove Image

```bash
docker rmi medical-risk-predictor:latest
```

### View Container Logs

```bash
# Real-time logs
docker-compose logs -f dashboard

# Last 100 lines
docker-compose logs --tail 100 dashboard

# Logs with timestamps
docker-compose logs --timestamps dashboard
```

### Execute Command in Running Container

```bash
# Using container name (if running)
docker exec -it medical-risk-dashboard bash

# Using docker-compose
docker-compose exec dashboard bash
```

### Stop All Containers

```bash
docker-compose down
```

### Remove All Data (Clean Slate)

```bash
# Stop and remove containers + networks
docker-compose down --volumes
```

---

## Volume Mounts (Data Persistence)

The docker-compose.yml mounts these directories for data persistence:

| Container Path | Host Path | Purpose |
|---|---|---|
| `/app/medical_risk_predictor/data/raw` | `./medical_risk_predictor/data/raw` | Raw datasets |
| `/app/medical_risk_predictor/data/processed` | `./medical_risk_predictor/data/processed` | Processed datasets |
| `/app/medical_risk_predictor/models/saved_models` | `./medical_risk_predictor/models/saved_models` | Trained model artifacts |
| `/app/medical_risk_predictor/reports/generated` | `./medical_risk_predictor/reports/generated` | Generated reports & plots |
| `/home/mluser/.kaggle` | `~/.kaggle` (read-only) | Kaggle credentials |

**Key:** Changes made in the container are persisted on the host system.

---

## Environment Variables

### Streamlit Settings

```bash
STREAMLIT_SERVER_PORT=8501          # Dashboard port
STREAMLIT_SERVER_ADDRESS=0.0.0.0    # Bind to all interfaces
STREAMLIT_SERVER_HEADLESS=true      # Run without browser
```

### Kaggle Settings

```bash
KAGGLE_CONFIG_DIR=/home/mluser/.kaggle   # Path to kaggle.json
```

### Python Settings

```bash
PYTHONUNBUFFERED=1                  # Unbuffered output
PYTHONDONTWRITEBYTECODE=1           # Don't create __pycache__
```

---

## Production Deployment

### Deploy to Cloud (AWS ECS, Google Cloud Run, etc.)

```bash
# Example: Push to Docker Hub
docker login
docker tag medical-risk-predictor:latest yourname/medical-risk-predictor:latest
docker push yourname/medical-risk-predictor:latest

# Example: AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag medical-risk-predictor:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/medical-risk:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/medical-risk:latest
```

### Deploy with Kubernetes

```bash
# Save docker-compose as Kubernetes manifest
kompose convert -f docker-compose.yml -o k8s-manifests/

# Deploy
kubectl apply -f k8s-manifests/
```

### Docker Production Checklist

- [ ] Remove debug print statements
- [ ] Set `STREAMLIT_SERVER_HEADLESS=true`
- [ ] Configure resource limits (memory, CPU)
- [ ] Set up log aggregation (ELK, Datadog, etc.)
- [ ] Enable container health checks (✓ configured)
- [ ] Use non-root user (✓ mluser)
- [ ] Version tag all images
- [ ] Document environment secrets (Kaggle creds, API keys)

---

## Troubleshooting

### Issue: Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>

# Or use a different port
docker run -p 8502:8501 medical-risk-predictor:latest
```

### Issue: Kaggle Credentials Not Found

```bash
# Ensure kaggle.json exists
ls ~/.kaggle/kaggle.json

# Check file permissions
chmod 600 ~/.kaggle/kaggle.json

# Verify in container
docker-compose run --rm dashboard cat ~/.kaggle/kaggle.json
```

### Issue: Insufficient Disk Space

```bash
# Check Docker disk usage
docker system df

# Clean up unused images/containers
docker system prune -a --volumes
```

### Issue: Container Exits Immediately

```bash
# View error logs
docker-compose logs training

# Run interactively to see errors
docker-compose run --rm training bash
```

### Issue: Out of Memory (OOM)

```bash
# Increase Docker memory limit
# Docker Desktop: Preferences → Resources → Memory (increase to 8GB+)

# Or limit container memory
docker run -m 4g medical-risk-predictor:latest
```

---

## Performance Optimization

### Multi-stage Build Benefits

The Dockerfile uses a 2-stage build:
1. **Builder stage:** Installs all dependencies
2. **Runtime stage:** Copies only the venv, discards build tools

**Result:** ~40% smaller image (~1.2GB vs ~2GB)

### Layer Caching

Docker caches layers. To optimize rebuilds:
```bash
# Less frequently changed layers first (base, dependencies)
# More frequently changed layers last (code, scripts)
# Avoid: RUN pip install ... && RUN python script.py

# Good: Install dependencies once, then run scripts
```

### Parallel Builds (Docker Buildx)

```bash
# For multi-platform builds (amd64, arm64, etc.)
docker buildx build --platform linux/amd64,linux/arm64 -t medical-risk:latest .
```

---

## Example Workflows

### Workflow 1: Development

```bash
# 1. Build image
docker build -t medical-risk:dev .

# 2. Run with data directly accessible
docker-compose -f docker-compose.override.yml up dashboard

# 3. Edit code on host (changes reflect in container via volumes)
# 4. Browser auto-refreshes (Streamlit)
```

### Workflow 2: Full CI/CD Pipeline

```bash
# 1. Build
docker build -t medical-risk:latest .

# 2. Test
docker run --rm medical-risk:latest python -m pytest medical_risk_predictor/tests/

# 3. Prepare data
docker-compose --profile data-prep up prepare-data

# 4. Train
docker-compose --profile training up training

# 5. Deploy
docker-compose up -d dashboard
```

### Workflow 3: Production Training Cluster

```bash
# Run training on powerful machine, push results
docker-compose --profile pipeline up full-pipeline

# Pull trained models on inference machine
docker-compose pull
docker-compose up dashboard
```

---

## Version Control & GitOps

### Docker Image Versioning

```bash
# Semantic versioning
docker build -t medical-risk:1.0.0 .

# Git commit hash (recommended for CI/CD)
HASH=$(git rev-parse --short HEAD)
docker build -t medical-risk:${HASH} .
```

### Docker Compose Override for Local Development

Create `docker-compose.override.yml` (automatically loaded):

```yaml
version: '3.8'

services:
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
      cache_from:
        - medical-risk:latest
    command: streamlit run --logger.level=debug medical_risk_predictor/dashboard/app.py
    environment:
      - STREAMLIT_LOGGER_LEVEL=debug
```

---

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Best Practices for Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/dockerfile_best-practices/)
- [Streamlit in Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
