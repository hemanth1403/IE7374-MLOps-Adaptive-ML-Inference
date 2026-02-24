# Intelligent Multi-Model ML Orchestration Platform

**RL-Powered Adaptive Model Selection for Autonomous Systems Perception**

[![MLOps](https://img.shields.io/badge/MLOps-IE7374-blue)](https://www.mlwithramin.com)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![Airflow](https://img.shields.io/badge/Airflow-2.11-orange)](https://airflow.apache.org/)
[![DVC](https://img.shields.io/badge/DVC-Enabled-purple)](https://dvc.org/)

---

## Project Overview

### The Problem

Autonomous robots and drones need real-time perception for safe navigation. Current systems use a single heavy model (e.g., YOLOv8-large) for every frame, introducing 40-50ms latency regardless of scene complexity. At speeds of 2-10 m/s, this creates 8-50cm of "blind" travel per inference—a safety and efficiency bottleneck.

**Additionally:** 60-70% of compute is wasted on simple scenes that don't need heavy models.

### Our Solution

We build an **Intelligent Multi-Model ML Orchestration Platform** that:

1. **Deploys three YOLOv8 variants simultaneously** (nano, small, large)
2. **Uses Reinforcement Learning** to route each frame to the optimal model
3. **Achieves 58% latency reduction and 42% cost savings**
4. **Maintains 95%+ accuracy** for safety-critical scenarios

**Core Innovation:** The RL-powered orchestration layer generalizes beyond computer vision to any ML domain (NLP, time series, audio, etc.). We demonstrate with autonomous systems because real-time requirements make the value proposition clear.

### Performance Targets

**Latency:** 48ms -> 20ms average (58% reduction)  
**Throughput:** 21 FPS -> 55 FPS (2.6x improvement)  
**Cost:** $200/1M -> $115/1M inferences (42% savings)  
**Accuracy:** 52.8% -> 50.2% mAP (95% retention)  
**GPU Utilization:** 95% -> 30% (3x headroom)

**At Scale:** 1,000 robots at 30 FPS = **$72K monthly savings**

---

## System Architecture

### High-Level Design

```
Camera Feed -> Feature Extraction (5ms) -> RL Agent (2ms) -> Model Selection
                                                              ↓
                               ┌──────────────────────────────┼────────────────┐
                               ↓                              ↓                ↓
                       YOLOv8-nano (8ms)            YOLOv8-small (15ms)  YOLOv8-large (48ms)
                       Simple scenes (70%)          Moderate (20%)       Complex (10%)
                               ↓                              ↓                ↓
                               └──────────────────────────────┴────────────────┘
                                                    ↓
                                          Navigation System
```

### Deployment Stack

**Infrastructure:** Google Kubernetes Engine (GKE)  
**Models:** YOLO nano/small/large (Ultralytics pretrained)  
**RL Agent:** DQN/PPO (Stable Baselines3)  
**Model Registry:** MLflow  
**Monitoring:** Prometheus + Grafana  
**CI/CD:** GitHub Actions  
**Data Versioning:** DVC

---

## Project Timeline

### Checkpoint Progress

Checkpoint 1: Project Scoping

- Comprehensive scoping document (16 pages)
- Google People+AI worksheet completed
- System architecture designed
- Team formed and roles assigned
- **Status:** Completed & Approved

Checkpoint 2: Data Pipeline

- Complete Airflow + DVC pipeline
- COCO 2017 dataset processed (123K images)
- Comprehensive testing suite
- Docker deployment
- Schema validation, anomaly detection, bias analysis
- **Status:** Completed & Submitted
- **See:** [Data-Pipeline/README.md](Data-Pipeline/README.md)

Checkpoint 3: Model Development

- Train RL routing agent
- Benchmark YOLOv8 variants
- Build orchestration API
- Integration testing
- **Status:** In Progress

Checkpoint 4: Deployment

- Deploy on GKE
- CI/CD pipeline operational
- Monitoring dashboards
- Performance validation
- **Status:** Planned

Google Expo

- Live demo
- Final presentation

---

## Repository Structure

```
IE7374-MLOps-Adaptive-ML-Inference/
│
├── docs/                              # Documentation & screenshots
│   ├── mlops_project_pitch.pdf
│   ├── project_scoping_document.pdf
│   ├── airflow_graph_success.png
│   ├── airflow_gantt_success.png
│   └── dvcDAG.png
│
├── Data-Pipeline/                     #  Checkpoint 2
│   ├── dags/                         # Airflow DAGs
│   ├── scripts/                      # Modular data processing
│   ├── tests/                        # Test suite
│   ├── dvc.yaml                      # DVC pipeline
│   └── README.md                     # Pipeline documentation
│
├── models/                            #  Checkpoint 3 (In Progress)
│   ├── yolov8_benchmarks/           # Model performance metrics
│   ├── rl_agent/                    # RL training code
│   └── (to be added)
│
├── orchestration/                     #  Checkpoint 3 (In Progress)
│   ├── api/                         # FastAPI service
│   ├── feature_extraction/          # Complexity analysis
│   └── (to be added)
│
├── deployment/                        #  Checkpoint 4 (Planned)
│   ├── kubernetes/                  # GKE manifests
│   ├── docker/                      # Dockerfiles
│   └── (to be added)
│
├── monitoring/                        #  Checkpoint 4 (Planned)
│   ├── dashboards/                  # Grafana configs
│   └── (to be added)
│
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI/CD
│
├── Dockerfile.airflow               # Airflow container
├── docker-compose.airflow.yml       # Airflow stack
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## Getting Started

### Prerequisites

- **Docker** (recommended for Airflow)
- **Python 3.10+** (for local development)
- **50GB+ disk space** (for COCO dataset)
- **Git** & **DVC**

### Quick Setup

**For Checkpoint 2 (Data Pipeline):**

```bash
# Clone repository
git clone https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference.git
cd IE7374-MLOps-Adaptive-ML-Inference

# Run data pipeline with Docker
docker-compose -f docker-compose.airflow.yml up -d

# Access Airflow UI
open http://localhost:8080
# Login: airflow / airflow
# Trigger: dvc_coco_pipeline
```

**See [Data-Pipeline/README.md](Data-Pipeline/README.md) for complete instructions.**

---

## Testing

### Run All Tests

```bash
# Data pipeline tests
cd Data-Pipeline
pytest tests/ -v --cov=scripts

# (Model tests coming in Checkpoint 3)
# (Integration tests coming in Checkpoint 4)
```

### Continuous Integration

**GitHub Actions** runs on every push:

- Linting (flake8)
- Unit tests (pytest)
- Coverage reporting
- Build validation

**Status:** Check Actions tab on GitHub

---

## Key Features

### What Makes This Project Unique

**1. Novel Architecture**

- Multi-model orchestration with RL routing
- Not typical single-model deployment
- Generalizable platform (not just CV)

**2. Production MLOps**

- Complete CI/CD pipeline
- Containerized deployment (Docker, GKE)
- Monitoring and observability (Prometheus, Grafana)
- Data versioning (DVC)

**3. Real Business Value**

- 42% cost reduction at scale
- 58% latency improvement
- Maintains safety-critical accuracy
- Addresses actual production problem

**4. Differentiation**

- 80% of MLOps projects: GenAI/LLM chatbots
- We: Deep Learning + RL + Systems Optimization
- Demonstrates depth across multiple ML paradigms

---

## 🎓 Course Information

**Course:** IE7374 - MLOps  
**Semester:** Spring 2026  
**Institution:** Northeastern University  
**Instructor:** Prof. Ramin Mohammadi

### Learning Objectives Demonstrated

**Data Pipeline Engineering** - Airflow, DVC, testing  
 **Model Training & Evaluation** - Multi-model benchmarking  
 **Deployment & Orchestration** - Kubernetes, containers  
 **Monitoring & Observability** - Metrics, dashboards  
 **MLOps Best Practices** - Versioning, testing, CI/CD  
 **Production Thinking** - Cost optimization, SLAs, failure handling

---

## Documentation

### Checkpoint Deliverables

**Checkpoint 1 - Scoping:**

- [Project Scoping Document](docs/project_scoping_document.pdf)
- [Project Pitch](docs/mlops_project_pitch.pdf)
- Google People+AI Worksheet

**Checkpoint 2 - Data Pipeline:**

- [Data Pipeline README](Data-Pipeline/README.md)
- [Airflow DAG](Data-Pipeline/dags/dvc_coco_pipeline.py)
- [DVC Pipeline](Data-Pipeline/dvc.yaml)
- [Pipeline Visualizations](docs/)

**Checkpoint 3 - Model Development:** (In Progress)

- Coming soon...

**Checkpoint 4 - Deployment:** (Planned)

- Coming soon...

---

## Development

### Current Sprint Focus

**Checkpoint 3 Objectives :**

- [ ] Train RL routing agent
- [ ] Benchmark YOLOv8 nano/small/large
- [ ] Build orchestration API (FastAPI)
- [ ] Implement feature extraction
- [ ] Integration testing
- [ ] Basic monitoring

### Contributing

**Team Workflow:**

1. Create feature branch
2. Implement changes
3. Write tests
4. Run `pytest` locally
5. Push and create PR
6. CI/CD validates
7. Team review
8. Merge to main

**Branch Strategy:**

- `main` - Stable, checkpoint submissions
- `develop` - Integration branch
- `feature/*` - Individual features
- `checkpoint-3` - Current sprint work

---

## Expected Outcomes

### Technical Deliverables

**Guaranteed (Phase 1):**

- Working multi-model deployment
- RL agent routing decisions
- Complete MLOps pipeline
- Monitoring dashboards
- Performance benchmarks

**Stretch Goals (Phase 2):**

- Active learning integration
- Multi-domain generalization demo
- Advanced monitoring features

### Performance Metrics

**Success Criteria:**

- 40%+ cost reduction vs baseline
- 2.5x+ throughput improvement
- 95%+ accuracy retention
- <100ms p95 latency
- 90%+ RL routing accuracy

---

## Technology Stack

### Core Technologies

**Machine Learning:**

- YOLO (Ultralytics) - Object detection models
- Stable Baselines3 - Reinforcement learning
- PyTorch - Deep learning framework

**MLOps Tools:**

- Apache Airflow - Workflow orchestration
- DVC - Data version control
- MLflow - Model registry & tracking
- Prometheus - Metrics collection
- Grafana - Visualization & dashboards

**Infrastructure:**

- Docker - Containerization
- Kubernetes (GKE) - Container orchestration
- Google Cloud Storage - Data storage
- FastAPI - Inference API
- PostgreSQL - Metadata storage

**Development:**

- Python 3.11
- Pytest - Testing framework
- GitHub Actions - CI/CD
- VS Code - Development environment

---

## Team

### Team Members

**Hemanth Sai Madadapu** - madadapu.h@northeastern.edu \
**Sujith Peddireddy** - peddireddy.su@northeastern.edu \
**Jan Mollet** - mollet.j@northeastern.edu \
**Sayee Ashish Aher** - aher.sa@northeastern.edu \
**Sowmyashree Jayaram** - jayaram.so@northeastern.edu

### Course Information

**Course:** IE7374 - MLOps  
**Instructor:** Prof. Ramin Mohammadi  
**Semester:** Spring 2026  
**Institution:** Northeastern University

---

## Documentation

### Quick Links

**Project Documentation:**

- [Project Scoping](docs/project_scoping_document.pdf)
- [Data Pipeline README](Data-Pipeline/README.md)
- [Airflow Setup Guide](Data-Pipeline/README.md#running-the-pipeline)
- [DVC Workflow](Data-Pipeline/README.md#data-versioning-with-dvc)

**Visualizations:**

- [Airflow Graph](docs/airflow_graph_success.png)
- [Gantt Chart](docs/airflow_gantt_success.png)
- [DVC Pipeline](docs/dvcDAG.png)

**Resources:**

- [Course Website](https://www.mlwithramin.com)
- [COCO Dataset](https://cocodataset.org/)
- [YOLO Docs](https://docs.ultralytics.com/)

---

## Key Differentiators

### Why This Project Stands Out

**1. Novel Technical Approach**

- RL for system optimization (not just supervised learning)
- Multi-model orchestration (vs single model)
- Production efficiency focus (vs pure accuracy)

**2. Complete MLOps Lifecycle**

- Data pipeline -> Model training -> Deployment -> Monitoring
- Not just "train a model and deploy"
- Full production thinking

**3. Real Business Impact**

- Solves actual problem (inference waste)
- Quantified savings ($72K/month at scale)
- Generalizable architecture

**4. Production Quality**

- Docker containerization
- Kubernetes deployment
- Comprehensive testing
- CI/CD automation
- Monitoring & observability

**5. Google Relevance**

- Waymo (autonomous vehicles)
- Infrastructure cost optimization
- Efficiency at scale

---

## Technical Innovation

### Multi-Objective Optimization

**RL Reward Function:**

```
R = 0.5 × (accuracy/baseline) - 0.3 × (latency/budget) - 0.2 × (cost/budget)
```

**Balances three competing objectives:**

- Maximize accuracy (safety)
- Minimize latency (real-time)
- Minimize cost (efficiency)

**Result:** Pareto-optimal routing decisions

### Adaptive Routing Policy

**RL Agent learns:**

- When nano model is sufficient (70% of cases)
- When small model balances speed/accuracy (20%)
- When large model is necessary for safety (10%)

**Better than rules because:**

- Discovers patterns humans miss
- Adapts to traffic patterns
- Improves over time
- Handles uncertainty

---

## Benchmarking & Metrics

### Model Performance (Expected)

**YOLOv8-nano:**

- Latency: 8ms
- mAP@0.5: 40%
- Cost: $0.04/1K
- Use case: Simple scenes

**YOLOv8-small:**

- Latency: 15ms
- mAP@0.5: 46%
- Cost: $0.075/1K
- Use case: Moderate scenes

**YOLOv8-large:**

- Latency: 48ms
- mAP@0.5: 53%
- Cost: $0.20/1K
- Use case: Complex scenes

**Adaptive System (Ours):**

- Avg Latency: 20ms (58% faster)
- mAP@0.5: 50.2% (95% retention)
- Cost: $115/1K (42% cheaper)
- Throughput: 55 FPS (2.6x more)

---

## Setup & Installation

### For Checkpoint 2 (Data Pipeline)

**Using Docker (Recommended):**

```bash
docker-compose -f docker-compose.airflow.yml up -d
# Access http://localhost:8080
```

**Using Local Setup:**

```bash
cd Data-Pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
dvc repro
```

**See [Data-Pipeline/README.md](Data-Pipeline/README.md) for detailed instructions.**

### For Future Checkpoints

**Instructions will be added as development progresses.**

---

## Testing

### Current Test Coverage

**Data Pipeline:**

- 5 test modules
- Unit + integration tests
- CI/CD automated
- See: `pytest Data-Pipeline/tests/ -v`

**Model Training:** (Coming in Checkpoint 3)

**Deployment:** (Coming in Checkpoint 4)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

This project is for academic purposes as part of IE7374 MLOps coursework.

---

## Acknowledgments

### Datasets & Tools

- **COCO Dataset:** Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
- **YOLO:** Ultralytics team
- **Apache Airflow:** Apache Software Foundation
- **DVC:** Iterative.ai

### Course & Support

- **Prof. Ramin Mohammadi** - Course instruction and guidance
- **Northeastern University** - MLOps program
- **Google Cloud Platform** - Infrastructure credits

---

## Contact

**Project Repository:** https://github.com/hemanth1403/IE7374-MLOps-Adaptive-ML-Inference

**For Questions:**

- Create GitHub issue
- Contact team members via email
- Check documentation in respective folders

---

_README will be updated with each checkpoint completion._

**Current Focus:** Checkpoint 3 - Model Development & RL Agent Training
