# Intelligent Multi-Model ML Orchestration Platform

**RL-Powered Adaptive Model Selection for Autonomous Systems Perception**

[![MLOps](https://img.shields.io/badge/MLOps-IE7374-blue)](https://www.mlwithramin.com)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.11-orange)](https://airflow.apache.org/)
[![DVC](https://img.shields.io/badge/DVC-Enabled-purple)](https://dvc.org/)

---

## Project Overview

### The Problem

Autonomous robots and drones need real-time perception for safe navigation. Current systems use a single heavy model (e.g., YOLOv8-large) for every frame, introducing 40-50ms latency regardless of scene complexity. At speeds of 2-10 m/s, this creates 8-50cm of "blind" travel per inference — a safety and efficiency bottleneck.

**Additionally:** 60-70% of compute is wasted on simple scenes that don't need heavy models.

### Our Solution

We build an **Intelligent Multi-Model ML Orchestration Platform** that:

1. **Deploys three YOLOv8 variants simultaneously** (nano, small, large)
2. **Uses Reinforcement Learning** to route each frame to the optimal model
3. **Achieves 58% latency reduction and 42% cost savings**
4. **Maintains 95%+ accuracy** for safety-critical scenarios

**Core Innovation:** The RL-powered orchestration layer generalizes beyond computer vision to any ML domain (NLP, time series, audio, etc.). We demonstrate with autonomous systems because real-time requirements make the value proposition clear.

### Performance Targets

**Latency:** 48ms → 20ms average (58% reduction)  
**Throughput:** 21 FPS → 55 FPS (2.6× improvement)  
**Cost:** $200/1M → $115/1M inferences (42% savings)  
**Accuracy:** 52.8% → 50.2% mAP (95% retention)  
**GPU Utilization:** 95% → 30% (3× headroom)

**At Scale:** 1,000 robots at 30 FPS = **$72K monthly savings**

---

## System Architecture

### High-Level Design

```
Camera / Video Feed
        │
        ▼
Feature Extraction (32×32 grayscale + Canny edge density)
        │
        ▼
PPO RL Agent  ──────────────────────────────────────────────
        │                                                    │
        ▼ action (0/1/2)                       baseline path (YOLOv8-Small)
   ┌────┴────────────────────┐                              │
   ▼           ▼             ▼                              ▼
YOLOv8-nano  YOLOv8-small  YOLOv8-large          YOLOv8-small (fixed)
  ~70%          ~20%          ~10%
   └────────────┴─────────────┘
                │
                ▼
        Structured detections  ──── WebSocket (JSON) ────► Streamlit UI
                │
                ▼
        MLflow session log
```

### Serving Architecture

| Component | Technology | Role |
|---|---|---|
| `serving/engine.py` | PyTorch / ONNX Runtime | Dual-path inference engine |
| `serving/app.py` | FastAPI + WebSocket | Streaming inference API |
| `serving/ui.py` | Streamlit | Live dashboard (camera + video upload) |
| `serving/tracking.py` | MLflow | Per-session metrics logging |

**Data flow:** Streamlit encodes frames as base64 JPEG → sends over WebSocket to FastAPI → `engine.infer()` runs both RL-adaptive and baseline paths → JSON result back → Streamlit renders annotated video + metrics charts.

### Infrastructure

**Compute:** Google Kubernetes Engine (GKE), `europe-west1-b`  
**Models:** YOLOv8 nano/small/large — ONNX format (deployed), PyTorch .pt (local)  
**RL Agent:** PPO via Stable Baselines3  
**Model Registry:** MLflow  
**Monitoring:** Prometheus + Grafana  
**CI/CD:** GitHub Actions  
**Data Versioning:** DVC

---

## Repository Structure

```
IE7374-MLOps-Adaptive-ML-Inference/
│
├── Data-Pipeline/                     # Checkpoint 2 — Complete
│   ├── dags/                          # Airflow DAGs
│   ├── scripts/                       # Modular data processing (8 stages)
│   ├── tests/                         # 5 test modules
│   ├── dvc.yaml / dvc.lock            # DVC pipeline
│   └── README.md
│
├── model_pipeline/                    # Checkpoints 3 & 4 — Complete
│   ├── configs/                       # YAML configs (train / eval / data)
│   ├── artifacts/                     # dataset.yaml
│   ├── reports/                       # Benchmarks, bias, figures, metrics
│   ├── src/
│   │   ├── evaluation/                # YOLO benchmarking & comparison scripts
│   │   ├── bias/                      # Slice-based bias analysis
│   │   └── RL/                        # RL agent + serving stack
│   │       ├── core/                  # environment.py, features.py, agent.py
│   │       ├── serving/               # FastAPI app, Streamlit UI, engine, tracking
│   │       ├── training/              # train_rl.py, pretrain_bc.py, profile_models.py
│   │       ├── monitoring/            # drift_detector.py, retrain_trigger.py
│   │       ├── tests/                 # RL policy & environment tests
│   │       ├── models/                # Trained PPO weights (PPO_v6/)
│   │       ├── Dockerfile             # Production container (CUDA 12.1)
│   │       ├── requirements_deploy.txt
│   │       └── README.md              # Full RL + serving documentation
│   └── README.md
│
├── infra/                             # Checkpoint 4 — Complete
│   ├── k8s/                           # 14 Kubernetes manifests (incl. drift CronJob)
│   ├── docker/                        # docker-compose.prod.yaml
│   ├── monitoring/                    # Prometheus + Grafana + alert rules
│   └── README.md
│
├── shared/                            # Cross-service contracts
│   ├── contracts/model_contract.md
│   └── README.md
│
├── docs/                              # PDFs, screenshots, pipeline diagrams
├── .github/workflows/ci.yml           # GitHub Actions CI
├── docker-compose.airflow.yml         # Airflow + DVC stack
├── DEPLOYMENT_INTERNAL.md             # Full GKE production deployment guide
├── CLAUDE.md                          # Developer guide for AI-assisted work
└── README.md                          # This file
```

---

## Checkpoint Progress

### Checkpoint 1 — Project Scoping
- Comprehensive scoping document (16 pages)
- Google People+AI worksheet
- System architecture designed
- **Status: Completed**

### Checkpoint 2 — Data Pipeline
- Airflow + DVC pipeline (8 stages)
- COCO 2017 dataset (123K images) → YOLO format
- Schema validation, anomaly detection, bias analysis
- **Status: Completed** — see [Data-Pipeline/README.md](Data-Pipeline/README.md)

### Checkpoint 3 — Model Development
- PPO RL routing agent trained (Behavioral Cloning warm-start)
- YOLOv8 nano/small/large benchmarked and profiled
- FastAPI WebSocket inference API with dual-path inference
- Streamlit dashboard (live camera + video upload)
- MLflow experiment tracking + session logging
- **Status: Completed** — see [model_pipeline/src/RL/README.md](model_pipeline/src/RL/README.md)

### Checkpoint 4 — Deployment
- Kubernetes manifests for all services (backend, UI, MLflow, monitoring)
- Production Docker image with CUDA 12.1
- ONNX-optimised YOLO models for CPU inference
- Prometheus metrics + Grafana dashboards with alerting rules
- HTTPS ingress with TLS, HPA autoscaling
- GitHub Actions CI/CD pipeline (6 workflows)
- Evidently AI drift detection — K8s CronJob runs every 6 hours
- Automated retraining pipeline triggered on drift or performance decay
- Slack notifications for retraining events
- **Status: Completed & Live on GKE** — see [infra/README.md](infra/README.md)

---

## Quick Start

### Option 1 — Local dev (GPU recommended)

```bash
git clone https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference.git
cd IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL

pip install -r requirements_deploy.txt

# Terminal 1 — FastAPI backend
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run serving/ui.py
# Open http://localhost:8501
```

### Option 2 — Docker (single GPU node)

```bash
cd model_pipeline/src/RL
docker build -t adaptive-inference .
docker run --gpus all -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  adaptive-inference
```

### Option 3 — Docker Compose (full stack)

```bash
cd infra/docker
docker compose -f docker-compose.prod.yaml up -d
# Backend:  http://localhost:8000
# UI:       http://localhost:8501
# MLflow:   http://localhost:5000
# Grafana:  http://localhost:3000
```

### Option 4 — Data Pipeline only

```bash
docker-compose -f docker-compose.airflow.yml up -d
# Airflow UI: http://localhost:8080  (login: airflow / airflow)
# Trigger: dvc_coco_pipeline
```

See [Data-Pipeline/README.md](Data-Pipeline/README.md) for detailed pipeline instructions.

---

## RL Reward Function

```
R = 0.5 × (accuracy / baseline_acc)
  − 0.3 × (latency / latency_budget)
  − 0.2 × (cost / cost_budget)
```

The agent learns to route simple scenes to YOLOv8-nano (~70%), moderate scenes to small (~20%), and reserves large for high-complexity frames (~10%).

---

## Model Performance

| Model | Latency | mAP@0.5 | Cost/1K inf | Use case |
|---|---|---|---|---|
| YOLOv8-nano | 8ms | 40% | $0.04 | Simple scenes |
| YOLOv8-small | 15ms | 46% | $0.075 | Moderate scenes |
| YOLOv8-large | 48ms | 53% | $0.20 | Complex scenes |
| **RL Adaptive** | **20ms avg** | **50.2%** | **$0.115** | All scenes |

---

## Technology Stack

**ML:** Ultralytics YOLOv8, Stable Baselines3 (PPO), PyTorch, ONNX Runtime  
**MLOps:** Airflow, DVC, MLflow, Prometheus, Grafana  
**Backend:** FastAPI, Uvicorn, WebSockets  
**Frontend:** Streamlit  
**Infrastructure:** Docker, Kubernetes (GKE), Google Cloud  
**CI/CD:** GitHub Actions  
**Testing:** Pytest

---

## Testing

```bash
# Data pipeline tests
cd Data-Pipeline && pytest tests/ -v --cov=scripts

# RL environment and policy tests
cd model_pipeline/src/RL && python -m pytest tests/ -v
```

GitHub Actions runs data pipeline tests on every push.

### CI/CD Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | Push / PR | Data pipeline lint + pytest |
| `rl-ci.yml` | Push / PR | RL environment unit tests + DVC dry-run |
| `model-pipeline-ci.yml` | Push / PR | Model evaluation + bias reports |
| `rl-docker.yml` | Push to `model_pipeline/src/RL/**` | Build & push Docker image to GHCR |
| `deploy-gke.yml` | After Docker build succeeds | Rolling deploy to GKE (all K8s manifests) |
| `retrain.yml` | `repository_dispatch` or manual | DVC retrain → quality gate → build → GKE deploy |

---

## Team

**Hemanth Sai Madadapu** — madadapu.h@northeastern.edu  
**Sujith Peddireddy** — peddireddy.su@northeastern.edu  
**Jan Mollet** — mollet.j@northeastern.edu  
**Sayee Ashish Aher** — aher.sa@northeastern.edu  
**Sowmyashree Jayaram** — jayaram.so@northeastern.edu

**Course:** IE7374 MLOps — Spring 2026, Northeastern University  
**Instructor:** Prof. Ramin Mohammadi

---

## Documentation

| Document | Description |
|---|---|
| [Data-Pipeline/README.md](Data-Pipeline/README.md) | Airflow + DVC pipeline guide |
| [model_pipeline/src/RL/README.md](model_pipeline/src/RL/README.md) | RL agent training + serving stack |
| [infra/README.md](infra/README.md) | Docker + Kubernetes + drift detection + retraining |
| [DEPLOYMENT_INTERNAL.md](DEPLOYMENT_INTERNAL.md) | Full GKE production deployment guide |
| [docs/project_scoping_document.pdf](docs/project_scoping_document.pdf) | Checkpoint 1 scoping |

---

## Acknowledgments

- **COCO Dataset:** Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
- **Ultralytics** — YOLOv8
- **Prof. Ramin Mohammadi** — Course instruction and guidance
- **Google Cloud Platform** — Infrastructure credits
