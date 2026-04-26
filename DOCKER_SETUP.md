# 🐳 Docker Integration - Complete Setup Guide

**Project:** Medical Disease Risk Prediction System  
**Date:** April 27, 2026  
**Status:** ✅ Ready for Docker Deployment

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What's Been Created](#whats-been-created)
3. [Prerequisites](#prerequisites)
4. [Build Instructions](#build-instructions)
5. [Running Services](#running-services)
6. [Use Cases & Examples](#use-cases--examples)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### For Impatient Users (TL;DR)

```bash
cd /home/blaze/important/project.ml/Medical-Disease-Risk-Prediction-System

# NEW: One-command setup (interactive menu)
bash docker-setup.sh

# OR manual commands:
docker build -t medical-risk-predictor:latest .
docker-compose up dashboard
# Access: http://localhost:8501
```

**That's it!** The dashboard will be running in ~30 seconds.

---

## What's Been Created

### Files Added to Project

```
Medical-Disease-Risk-Prediction-System/
├── Dockerfile                          # Multi-stage build for optimized image
├── docker-compose.yml                  # Orchestration with 5 service profiles
├── .dockerignore                       # Excludes unnecessary files
├── docker-setup.sh                     # Interactive setup script ⭐ (NEW)
├── DOCKER_GUIDE.md                     # Comprehensive Docker documentation
└── DOCKER_QUICK_REFERENCE.md           # Quick reference card

Project Structure (existing, now Dockerized):
├── medical_risk_predictor/
│   ├── dashboard/app.py               # Streamlit UI
│   ├── src/                           # ML modules
│   ├── scripts/                       # Automation scripts
│   ├── models/saved_models/           # Trained models (mounted volume)
│   ├── data/raw/ & processed/         # Datasets (mounted volumes)
│   └── requirements.txt               # Python dependencies
└── ...
```

---

## Prerequisites

### System Requirements

```bash
# Docker & Docker Compose
✓ Docker 20.10+ (installed: 29.4.1)
✓ Docker Compose 2.0+ (included with Docker Desktop)
✓ 4GB RAM minimum (8GB+ recommended for training)
✓ 2GB disk space for image + data

# User Permissions
✓ User in 'docker' group (for non-sudo access)
✓ Terminal with bash/sh
```

### Check Installation

```bash
docker --version
docker-compose --version
docker ps  # Should work if daemon is running
```

### Fix Permissions (if needed)

```bash
# Add current user to docker group
sudo usermod -aG docker $USER

# Activate in current session
newgrp docker

# Verify
groups $USER | grep docker  # Should show 'docker'
```

---

## Build Instructions

### Method 1: Interactive Setup Script (RECOMMENDED)

```bash
cd /home/blaze/important/project.ml/Medical-Disease-Risk-Prediction-System
bash docker-setup.sh

# Prompts you to:
# 1. Build image (standard / no-cache / verbose)
# 2. Choose deployment (Dashboard / Training / Jupyter / etc.)
```

**Why:** Handles permissions, shows errors clearly, provides menu-driven options.

### Method 2: Manual Docker Build

```bash
cd /home/blaze/important/project.ml/Medical-Disease-Risk-Prediction-System

# Standard build (uses layer cache, faster)
docker build -t medical-risk-predictor:latest .

# Force rebuild (rebuilds all layers)
docker build --no-cache -t medical-risk-predictor:latest .

# Verbose output
docker build -t medical-risk-predictor:latest . --progress=plain
```

**Build Details:**
- **Time:** ~5-10 min (first), <30 sec (cached)
- **Size:** ~1.2 GB (after multi-stage optimization)
- **Base:** python:3.12-slim
- **Dependencies:** All from requirements.txt

### Verify Build

```bash
docker images medical-risk-predictor
# Output: Shows image name, tag, size, creation time
```

---

## Running Services

### Service 1: Dashboard (Interactive UI)

**What:** Streamlit web interface for predictions  
**Default Port:** 8501  
**Browser:** http://localhost:8501

#### Start Dashboard

```bash
# Option A: Using docker-compose (easiest, persistent)
docker-compose up dashboard

# Option B: Using docker run
docker run -it --rm \
  -p 8501:8501 \
  -v $(pwd)/medical_risk_predictor/data:/app/medical_risk_predictor/data \
  -v $(pwd)/medical_risk_predictor/models:/app/medical_risk_predictor/models \
  medical-risk-predictor:latest

# Option C: Run in background
docker-compose up -d dashboard
docker-compose logs -f dashboard  # View logs
docker-compose down              # Stop
```

#### Dashboard Features
- Risk prediction (Diabetes & Heart Disease)
- SHAP explainability charts
- Model comparison
- Anomaly detection
- PDF report generation

### Service 2: Full Training Pipeline

**What:** Complete ML workflow (data prep → train → eval)  
**Time:** ~5-10 minutes  
**Exit:** One-shot (completes then stops)

```bash
# Using docker-compose (recommended)
docker-compose --profile pipeline up full-pipeline

# View progress
docker-compose logs -f full-pipeline

# What it does:
# 1. Fetches Kaggle datasets (1517 heart disease records, 768 diabetes)
# 2. Runs feature engineering & preprocessing
# 3. Trains 4 ML models with GridSearchCV
# 4. Generates evaluation plots & SHAP charts
# 5. Saves models to models/saved_models/
```

### Service 3: Data Preparation Only

**What:** Fetch & prepare Kaggle datasets  
**Time:** ~2-3 minutes

```bash
docker-compose --profile data-prep up prepare-data

# Requires: ~/.kaggle/kaggle.json (your Kaggle credentials)
# Output: CSV files in data/raw/ and data/processed/
```

### Service 4: Training Only

**What:** Train models on already-prepared data  
**Time:** ~2-3 minutes

```bash
docker-compose --profile training up training

# Prerequisites: data/processed/*.csv must exist
# Output: Trained models in models/saved_models/
```

### Service 5: Jupyter Lab

**What:** Interactive notebook environment  
**Port:** 8888  
**Browser:** http://localhost:8888

```bash
# Start Jupyter
docker-compose --profile notebook up notebook

# Get access token
docker-compose logs notebook | grep token

# Access: http://localhost:8888?token=<TOKEN>
```

---

## Use Cases & Examples

### Use Case 1: Development & Experimentation

```bash
# 1. Build once
docker build -t medical-risk-predictor:latest .

# 2. Launch Jupyter for interactive coding
docker-compose --profile notebook up notebook

# 3. Make changes on host system (volumes auto-sync)

# 4. Save changes to Docker image (when ready)
docker build -t medical-risk-predictor:v1.1 .
```

### Use Case 2: End-to-End Model Training

```bash
# 1. Prepare & fetch Kaggle data
docker-compose --profile data-prep up prepare-data

# 2. Train models
docker-compose --profile training up training

# 3. Launch dashboard with new models
docker-compose up dashboard
```

### Use Case 3: Production Deployment

```bash
# 1. Build & tag
docker build -t medical-risk-predictor:v1.0 .

# 2. Test locally
docker-compose up dashboard

# 3. Push to Docker Hub (or Cloud Registry)
docker login
docker tag medical-risk-predictor:v1.0 yourname/medical-risk:v1.0
docker push yourname/medical-risk:v1.0

# 4. Deploy to cloud (AWS, GCP, Azure, etc.)
# See DOCKER_GUIDE.md for cloud deployment steps
```

### Use Case 4: CI/CD Pipeline

```bash
# Dockerfile + docker-compose ready for:
# - GitHub Actions (automated tests + builds)
# - GitLab CI
# - Jenkins
# - CircleCI
# See DOCKER_GUIDE.md for CI/CD examples
```

---

## Volume Mounts (Data Persistence)

### How Volumes Work

Volumes connect container directories to host directories, ensuring:
- **Data persists** after container stops
- **Changes sync** between host & container
- **Model artifacts** are accessible on host

### Configured Volumes

```yaml
Volumes in docker-compose.yml:

Container Path                          Host Path                                  Purpose
───────────────────────────────────── ────────────────────────────────────────── ───────────────────
/app/medical_risk_predictor/data/raw   → ./medical_risk_predictor/data/raw       Raw datasets
/app/medical_risk_predictor/data/proc  → ./medical_risk_predictor/data/processed Processed data
/app/medical_risk_predictor/models     → ./medical_risk_predictor/models/...     Trained models
/app/medical_risk_predictor/reports    → ./medical_risk_predictor/reports/...    Evaluation plots
/home/mluser/.kaggle                   → ~/.kaggle (read-only)                   Kaggle credentials
```

### Access Data from Host

```bash
# After training in Docker, models are immediately available:
ls medical_risk_predictor/models/saved_models/
# Output: best_diabetes.joblib, best_heart.joblib, etc.

# Generate reports directly:
cat medical_risk_predictor/reports/generated/model_comparison.png
```

---

## Environment Variables

### Streamlit Configuration

```bash
STREAMLIT_SERVER_PORT=8501           # Dashboard port
STREAMLIT_SERVER_ADDRESS=0.0.0.0     # Bind to all interfaces
STREAMLIT_SERVER_HEADLESS=true       # No browser auto-open
```

### Kaggle Configuration

```bash
KAGGLE_CONFIG_DIR=/home/mluser/.kaggle   # Credentials location
```

### Python Runtime

```bash
PYTHONUNBUFFERED=1                   # Real-time output
PYTHONDONTWRITEBYTECODE=1            # No .pyc files
```

### Custom Environment (docker-compose)

```yaml
# In docker-compose.yml services section:
environment:
  - STREAMLIT_SERVER_PORT=8501
  - KAGGLE_CONFIG_DIR=/home/mluser/.kaggle
```

---

## Container Management

### Useful Docker Commands

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# View container logs
docker logs <container_id>
docker logs -f <container_id>  # Follow (real-time)

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>

# Inspect container details
docker inspect <container_id>

# Execute command in running container
docker exec -it <container_id> bash

# Copy files from container to host
docker cp <container_id>:/app/path ./local/path
```

### Using docker-compose Commands

```bash
# Start services
docker-compose up [service]

# Start in background
docker-compose up -d [service]

# View logs
docker-compose logs -f [service]

# Stop services
docker-compose stop

# Stop & remove
docker-compose down

# Remove volumes too (careful!)
docker-compose down --volumes

# Execute command
docker-compose run --rm [service] bash

# List running services
docker-compose ps
```

---

## Troubleshooting

### Issue 1: Permission Denied

**Error:** `permission denied while trying to connect to the docker API`

**Solutions:**
```bash
# Check if user is in docker group
groups $USER | grep docker

# If not, add user
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

### Issue 2: Port Already in Use

**Error:** `bind: address already in use`

**Solutions:**
```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>

# OR use different port
docker run -p 8502:8501 medical-risk-predictor:latest
```

### Issue 3: Out of Memory (OOM)

**Error:** Build fails or container crashes

**Solutions:**
```bash
# Check Docker memory limit
docker info | grep "Memory Limit"

# Increase Docker memory:
# Docker Desktop: Preferences → Resources → Memory (increase to 8GB+)

# Or limit container memory
docker run -m 4g medical-risk-predictor:latest
```

### Issue 4: Kaggle Credentials Missing

**Error:** `Kaggle API credentials not found`

**Solutions:**
```bash
# Ensure kaggle.json exists
ls ~/.kaggle/kaggle.json

# Correct permissions
chmod 600 ~/.kaggle/kaggle.json

# Test in container
docker-compose run --rm dashboard cat ~/.kaggle/kaggle.json
```

### Issue 5: Build Takes Too Long

**Error:** Build seems stuck

**Solutions:**
```bash
# Cancel and rebuild with cache
docker build --no-cache -t medical-risk-predictor:latest .

# Or build with progress display
docker build -t medical-risk-predictor:latest . --progress=plain

# Check internet connection (dependencies downloading)
```

### Issue 6: Container Won't Start

**Error:** `docker: Error response from daemon`

**Solutions:**
```bash
# Check Docker daemon status
docker ps

# View container logs
docker logs <container_id>

# Check docker-compose.yml syntax
docker-compose config

# Rebuild image
docker build --no-cache -t medical-risk-predictor:latest .
```

---

## Performance & Optimization

### Build Time Optimization

- **First build:** 5-10 minutes (dependencies download)
- **Cached builds:** <30 seconds (leverages layer cache)
- **Tip:** Don't modify base layers to keep cache valid

### Image Size

- **Current:** ~1.2 GB (multi-stage optimized)
- **Without optimization:** ~2+ GB
- **Layer breakdown:**
  - Python 3.12 slim base: ~150 MB
  - Python packages: ~900 MB
  - Project code: ~20 MB

### Runtime Performance

| Operation | Time | Memory |
|-----------|------|--------|
| Dashboard startup | ~30s | 500 MB |
| Single prediction | <1s | 200 MB |
| Full training | 5-10 min | 2-4 GB |
| Data prep (Kaggle) | 2-3 min | 1-2 GB |

---

## Next Steps

1. ✅ **Run interactive setup:** `bash docker-setup.sh`
2. ✅ **Build image:** `docker build -t medical-risk-predictor:latest .`
3. ✅ **Start dashboard:** `docker-compose up dashboard`
4. ✅ **Test predictions:** Open http://localhost:8501
5. ✅ **Generate reports:** Use dashboard to download PDFs
6. ✅ **Deploy:** Push to Docker Hub or cloud registry

---

## Additional Resources

- **Full Docker Guide:** [DOCKER_GUIDE.md](DOCKER_GUIDE.md)
- **Quick Reference:** [DOCKER_QUICK_REFERENCE.md](DOCKER_QUICK_REFERENCE.md)
- **Official Docs:** https://docs.docker.com
- **Docker Compose:** https://docs.docker.com/compose
- **Streamlit Deployment:** https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker

---

**Status:** ✅ Docker integration complete  
**Ready for:** Local development, cloud deployment, CI/CD pipelines  
**Support:** Check troubleshooting section or see DOCKER_GUIDE.md for advanced topics
