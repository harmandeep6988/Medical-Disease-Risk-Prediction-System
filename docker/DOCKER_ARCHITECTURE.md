# Docker Architecture Overview

## 🏗️ Complete Docker Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LOCAL DEVELOPMENT / PRODUCTION                          │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌──────────────────┐
                            │   docker build   │
                            │      script      │
                            └────────┬─────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
            ┌────────┐          ┌────────┐          ┌──────────┐
            │Dockerfile│       │.dockerignore│      │requirements.txt│
            │          │       │             │      │                │
            │Multi-    │       │Exclude:     │      │Python            │
            │Stage     │       │- .git       │      │Dependencies      │
            │Build     │       │- __pycache__│      │scikit-learn      │
            │          │       │- venv       │      │XGBoost           │
            │Base:     │       │- *.log      │      │SHAP              │
            │python:   │       │- docs       │      │Streamlit         │
            │3.12-slim │       │             │      │Kaggle API        │
            └────────┬─────────┴──────────────┴──────┼──────────────────┘
                     │                               │
                     └───────────┬───────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  Docker Image (~1.2 GB)    │
                    │ - python:3.12-slim         │
                    │ - All dependencies         │
                    │ - Project code             │
                    │ - mluser (non-root)        │
                    │ - Health checks            │
                    └─────────┬──────────────────┘
                              │
                ┌─────────────┼──────────────────┐
                │             │                  │
                ▼             ▼                  ▼
    ┌────────────────┐  ┌───────────┐  ┌──────────────────┐
    │docker-compose  │  │  docker   │  │ docker-setup.sh  │
    │    .yml        │  │   run     │  │   (interactive)  │
    │                │  │           │  │                  │
    │5 Profiles:     │  │ Direct    │  │ Menu-driven      │
    │1. dashboard    │  │ container │  │ setup & deploy   │
    │2. training     │  │ execution │  │                  │
    │3. data-prep    │  ▼           │  └──────────────────┘
    │4. full-pipe    │              │
    │5. notebook     │              │
    └─────┬──────────┘              │
          │                         │
          └────────────┬────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌──────────┐    ┌─────────────┐   ┌────────────────┐
│Container │    │  Container  │   │  Container     │
│Dashboard │    │  Training   │   │  Jupyter       │
│          │    │             │   │                │
│Port 8501 │    │Port: NA     │   │Port 8888       │
│          │    │(one-shot)   │   │                │
│:         │    │:            │   │:               │
│Streamlit │    │Data Prep    │   │Notebook        │
│UI        │    │+ Training   │   │Lab             │
│Inference │    │+ Evaluation │   │Interactive     │
│SHAP      │    │             │   │Exploration     │
│Reports   │    │27min exec   │   │                │
└─────┬────┘    └──────┬──────┘   └────────────────┘
      │                │
      └────────┬───────┘
               │
    ┌──────────▼──────────────────────────┐
    │     VOLUME MOUNTS (Persistence)     │
    ├─────────────────────────────────────┤
    │/app/data/raw/         →  ./data/raw │
    │/app/data/processed/   →  ./data/... │
    │/app/models/           →  ./models/  │
    │/app/reports/          →  ./reports/ │
    │/home/mluser/.kaggle   →  ~/.kaggle/ │
    └─────────────────────────────────────┘
              │
    ┌─────────▼──────────┐
    │  FILE PERSISTENCE  │
    │                    │
    │✓ Datasets sync     │
    │✓ Models saved      │
    │✓ Reports generated │
    │✓ Kaggle creds      │
    │✓ Changes persist   │
    └────────────────────┘
```

---

## 📊 Service Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                   SERVICE EXECUTION FLOW                     │
└─────────────────────────────────────────────────────────────┘

DASHBOARD SERVICE (Persistent)
────────────────────────────────
Start:           docker-compose up dashboard
                        ↓
                 Container starts
                        ↓
              Streamlit app initializes
                        ↓
              Listen on 0.0.0.0:8501
                        ↓
         ✅ Access: http://localhost:8501
                        ↓
              User makes predictions
                        ↓
           SHAP explains predictions
                        ↓
              Generate PDF reports
                        ↓
    Stop: docker-compose down (or Ctrl+C)
    
    Restart Policy: unless-stopped
    ↳ Auto-restarts if crashed


TRAINING SERVICE (One-shot)
────────────────────────────
Start:           docker-compose --profile training up training
                        ↓
                 Container starts
                        ↓
              Load processed datasets
                        ↓
              Train 4 ML models
          (LogisticRegression, RandomForest,
              GradientBoosting, XGBoost)
                        ↓
           Perform cross-validation
                        ↓
            Evaluate & visualize
       (ROC curves, confusion matrix, etc)
                        ↓
           Generate SHAP explanations
                        ↓
        Save models to models/saved_models/
                        ↓
    Exit automatically (success)
    
    Output: Container completes, then stops


DATA PREPARATION SERVICE (One-shot)
────────────────────────────────────
Start:           docker-compose --profile data-prep up prepare-data
                        ↓
                 Container starts
                        ↓
          Load Kaggle API credentials
                        ↓
        Fetch 5 Kaggle datasets
     (3 diabetes + 2 heart disease)
                        ↓
          Standardize & merge schemas
                        ↓
           Deduplicate records
                        ↓
    Save to data/raw/ & data/processed/
                        ↓
    Exit automatically (success)


FULL PIPELINE SERVICE (One-shot)
─────────────────────────────────
Start:           docker-compose --profile pipeline up full-pipeline
                        ↓
        ┌─────────────────────────────────┐
        │  Run Data Prep + Training       │
        │  (Same as both above combined)  │
        └─────────────────────────────────┘
                        ↓
              ~5-10 minute execution
                        ↓
    Exit automatically with all artifacts
    
    Result: data/processed/*.csv
            models/saved_models/*.joblib
            reports/generated/*.png


JUPYTER SERVICE (Persistent, Interactive)
──────────────────────────────────────────
Start:           docker-compose --profile notebook up notebook
                        ↓
                 Container starts
                        ↓
         Jupyter Lab initializes
                        ↓
    ✅ Access: http://localhost:8888
    ↳ Get token from: docker-compose logs notebook
                        ↓
         Create/edit notebooks
         Run Python code interactively
         Explore data & visualizations
                        ↓
    Stop: docker-compose down (or Ctrl+C)
    
    Restart Policy: unless-stopped
    Notebooks persisted on host
```

---

## 🔄 Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      DATA FLOW ARCHITECTURE                     │
└────────────────────────────────────────────────────────────────┘

DEVELOPMENT → CONTAINER → PERSISTENCE → INFERENCE

┌─────────────────┐
│  Raw Datasets   │◄────────────────────────────┐
│                 │                             │
│ Pima Diabetes   │         ┌──────────────┐   │
│ 768 records     │         │   Training   │   │
│                 │         │  Container   │   │
│ Heart Disease   │         │              │   │
│ 1517 records    │         │ - GridSearch │   │
└────────┬────────┘         │ - CV (5-fold)│   │
         │                  │ - ModelComp  │   │
         ├────────────────►  │ - SHAP       │   │
         │                  │ - Eval plots │   │
         │ VOLUME           └─────┬────────┘   │
         │ MOUNT                  │            │
         │ (./data/raw/)          │            │
         │                        ▼            │
         │                  ┌────────────┐    │
         │                  │   Trained  │    │
         │                  │ ML Models  │    │
         │                  │            │    │
         │ ◄──────────────── │ - GradBoost│    │
         │                  │ - XGBoost  │    │
         │   VOLUME         │ - RF       │    │
         │   MOUNT          │ - LR       │    │
         │  (./models/)     │            │    │
         │                  │ + metadata │────┘
         │                  └────────────┘
         │
         │ VOLUME
         │ MOUNT
         │ (./data/processed/)
         │
         ▼
    ┌──────────────┐
    │  Streamlit   │
    │  Dashboard   │
    │ Container    │
    │              │
    │ - User inputs│
    │ - Inference  │
    │ - SHAP viz   │
    │ - PDF report │
    └──────────────┘

    Load models + scalers (joblib)
                │
                ▼
    Transform user inputs
    (raw → normalized → engineered)
                │
                ▼
    Model prediction + probability
                │
                ▼
    SHAP values + explanation
                │
                ▼
    PDF report generation
                │
                ▼
    User downloads report
```

---

## 🐳 Container Networking

```
┌─────────────────────────────────────────────────────┐
│           Docker Compose Network: ml-network       │
│                  (Bridge driver)                     │
└─────────────────────────────────────────────────────┘

                        ml-network
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    ┌─────────┐        ┌──────────┐      ┌──────────┐
    │Dashboard│        │Training  │      │Jupyter   │
    │:8501    │        │(one-shot)│      │:8888     │
    │         │        │          │      │          │
    │         │        │          │      │          │
    └─────────┘        └──────────┘      └──────────┘
        │                                      │
        │◄─────────────────────────────────────┤
        │     Container-to-container           │
        │     communication (if needed)        │
        │
        │
        └─► Each container can access
            volumes independently
            (no cross-container dependency)

EXTERNAL ACCESS
    │
    ├─► localhost:8501 ──► Dashboard
    │
    └─► localhost:8888 ──► Jupyter
```

---

## 📦 Image Layers (Multi-stage Optimization)

```
┌──────────────────────────────────────────────────┐
│           DOCKER IMAGE LAYER STRUCTURE           │
└──────────────────────────────────────────────────┘

BUILDER STAGE (Discarded in final image)
════════════════════════════════════════════════════════
Layer 1: python:3.12-slim base                (~150 MB)
         └─ apt-get update && install build tools

Layer 2: requirements.txt                     (~900 MB)
         └─ pip install in /opt/venv/
            ├─ scikit-learn
            ├─ xgboost
            ├─ shap
            ├─ streamlit
            ├─ pandas
            ├─ numpy
            ├─ plus 10+ dependencies

═══════════════════════════════════════════════════════
❌ DISCARDED (only venv used)
═══════════════════════════════════════════════════════

RUNTIME STAGE (Final image: ~1.2 GB)
════════════════════════════════════════════════════════
Layer 1: python:3.12-slim base                (~150 MB)
         └─ Clean installation (no build tools)

Layer 2: Copy /opt/venv from builder         (~900 MB)
         └─ Python packages only
            (build tools excluded)

Layer 3: Project code                         (~20 MB)
         ├─ medical_risk_predictor/
         ├─ scripts/
         ├─ src/
         └─ dashboard/

Layer 4: Configuration                        (~1 MB)
         ├─ mluser (non-root)
         ├─ Health check
         ├─ Working directory
         └─ ENV variables

═══════════════════════════════════════════════════════
✅ FINAL OPTIMIZED IMAGE: ~1.2 GB (40% smaller than single-stage)
═══════════════════════════════════════════════════════
```

---

## 🔐 Security Layers

```
┌──────────────────────────────────────────────┐
│         SECURITY ARCHITECTURE                 │
└──────────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Host Machine (Linux)                   │
│                                         │
│  User: blaze (may have sudo)           │
│  Groups: docker, adm, sudo, etc.       │
└────────────────────┬────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │ Docker Daemon  │
            │ (PID: 14289)   │
            │ Runs as: root  │
            └────────┬───────┘
                     │
                     ▼ (spawns container process)
            ┌────────────────────────────┐
            │  Container Process         │
            │  (isolated from host)      │
            │                            │
            │  User: mluser (uid 1000)  │◄─ Non-root
            │  ├─ No sudo access        │   ✓ Security
            │  ├─ No host root access   │
            │  └─ Confined processes    │
            │                            │
            │  Mount Points (volumes):   │
            │  ├─ data/raw (read)       │
            │  ├─ models (read/write)   │
            │  ├─ .kaggle (read-only)   │◄─ Creds protected
            │  └─ reports (write)       │
            │                            │
            │  Network Access:           │
            │  └─ Only :8501 exposed    │◄─ Port isolation
            └────────────────────────────┘
                     │
                     ▼ (back to host when done)
            Container stopped/removed
                No trace on host
```

---

## 🚀 Deployment Architectures

```
┌─────────────────────────────────────────────────────┐
│  DEPLOYMENT SCENARIOS (using this Docker setup)    │
└─────────────────────────────────────────────────────┘

SCENARIO 1: Single Machine (Laptop/Desktop)
═══════════════════════════════════════════
┌──────────────┐
│   Laptop     │
│   (Docker)   │
│              │
│ ┌──────────┐ │
│ │Dashboard ├─┼─→ http://localhost:8501
│ │Container │ │
│ └──────────┘ │
│              │
│ ┌──────────┐ │
│ │Jupyter   ├─┼─→ http://localhost:8888
│ │Container │ │
│ └──────────┘ │
└──────────────┘


SCENARIO 2: Linux Server
═════════════════════════
┌────────────────────────────┐
│  Linux Server              │
│  (Docker + docker-compose) │
│                            │
│  ┌────────────────────────┐│
│  │ Dashboard Container    ││
│  │ (persistent)           ││
│  │ Port: 8501             ││
│  └─────────^──────────────┘│
│            │               │
│            ▼               │
│  Reverse Proxy (nginx)     │──→ http://server.example.com:443 (HTTPS)
│                            │
│  ┌─────────────────────────│─────────┐
│  │ CI/CD Integration:      │         │
│  │ - GitHub Actions        │         │
│  │ - Push → Build → Deploy │         │
│  └─────────────────────────┤─────────┘
└────────────────────────────┘


SCENARIO 3: Kubernetes Cluster
═══════════════════════════════
┌─────────────────────────────────┐
│  Kubernetes Cluster             │
│  (3+ nodes with Docker)         │
│                                 │
│  ┌──────────────────────────┐   │
│  │ Node 1                   │   │
│  │ ┌──────────────────────┐ │   │
│  │ │ Dashboard POD        │ │   │
│  │ │ (2 replicas)         │ │   │
│  │ │ Container            │ │   │
│  │ └──────────────────────┘ │   │
│  └──────────────────────────┘   │
│                                 │
│  ┌──────────────────────────┐   │
│  │ Node 2                   │   │
│  │ ┌──────────────────────┐ │   │
│  │ │ Training JOB         │ │   │
│  │ │ (CronJob every night)│ │   │
│  │ │ Container            │ │   │
│  │ └──────────────────────┘ │   │
│  └──────────────────────────┘   │
│                                 │
│  Ingress: kubernetes.io/ingress  │
│  Service: LoadBalancer          │
│  PVC: PersistentVolume (models) │
└─────────────────────────────────┘


SCENARIO 4: Cloud Platform (AWS ECS)
════════════════════════════════════
┌────────────────────────────────────┐
│  AWS ECR                           │
│  (Docker image registry)           │
│                                    │
│ medical-risk:v1.0 ◄──┐            │
│                      │            │
│  ┌──────────────────┴──────────┐  │
│  │  AWS ECS Task Definition   │  │
│  │  (docker-compose.yml spec)│  │
│  └─────────┬──────────────────┘  │
│            │                      │
│            ▼                      │
│  ┌──────────────────────────────┐ │
│  │  ECS Cluster (EC2s)          │ │
│  │  ┌──────────────────────────┐6 │
│  │  │ Dashboard Service        │ │
│  │  │ (LoadBalancer + ALB)     │─┼─→ https://api.example.com
│  │  │ Port 8501                │ │
│  │  └──────────────────────────┘ │
│  └──────────────────────────────┘ │
└────────────────────────────────────┘
```

---

**Docker Integration Status:** ✅ Complete  
**Architecture:** Production-ready  
**Services:** 5 (dashboard, training, data-prep, full-pipeline, notebook)  
**Documentation:** 3 comprehensive guides + quick reference
