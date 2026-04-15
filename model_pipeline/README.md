# Model Pipeline

This module contains the complete **model evaluation, benchmarking, RL agent training, and production serving stack** for the adaptive ML inference project.

The pipeline evaluates three pre-trained YOLOv8 variants (nano, small, large), trains a PPO reinforcement learning agent to route frames between them, and serves the result via a FastAPI + Streamlit stack deployed on GKE.

---

## 1. Scope and Purpose

The goal of this pipeline is to answer:

- Which pre-trained YOLO model gives the best detection quality?
- Which model gives the best latency and throughput?
- How do these tradeoffs change across simple, moderate, and complex scenes?
- Can an RL agent learn to route frames better than a fixed model?

This project does **not** train a new detector architecture from scratch. It uses **pre-trained YOLOv8 nano, small, and large** models and focuses on:

- validation and benchmarking,
- model comparison and slice-based analysis,
- bias/disparity reporting,
- RL environment design and PPO agent training,
- production serving via FastAPI WebSocket + Streamlit dashboard,
- MLflow experiment tracking,
- Kubernetes deployment.

---

## 2. Models Evaluated

| Model | Latency | mAP@0.5 | Use case |
|---|---|---|---|
| `yolov8n.pt` — YOLO Nano | ~8ms | 40% | Simple scenes |
| `yolov8s.pt` — YOLO Small | ~15ms | 46% | Balanced baseline |
| `yolov8l.pt` — YOLO Large | ~48ms | 53% | Complex scenes |

---

## 3. Repository Structure

```
model_pipeline/
├── configs/
│   ├── data/dataset_config.yaml       # Dataset paths for evaluation
│   ├── eval/eval_config.yaml          # Evaluation metrics and benchmark settings
│   ├── train/train_config.yaml        # Model variants and checkpoint config
│   └── tracking/mlflow_config.yaml    # MLflow experiment settings
│
├── artifacts/
│   └── dataset.yaml                   # Dataset metadata for YOLO evaluation
│
├── reports/
│   ├── benchmarks/                    # Per-model runtime benchmark outputs
│   ├── bias/                          # Slice-based bias/disparity reports
│   ├── figures/                       # Comparison plots
│   └── metrics/                       # Per-model evaluation outputs
│
├── src/
│   ├── evaluation/
│   │   ├── evaluate.py                # YOLO validation (mAP, precision, recall)
│   │   ├── benchmark.py               # Latency and throughput profiling
│   │   ├── compare_models.py          # Cross-model comparison report
│   │   └── generate_slice_comparison.py
│   │
│   ├── bias/
│   │   └── generate_bias_report.py    # Slice-based disparity analysis
│   │
│   └── RL/                            # RL agent + production serving stack
│       ├── core/
│       │   ├── environment.py         # Gymnasium env (1028-dim obs, 3 actions)
│       │   ├── features.py            # FeatureExtractor (32×32 grayscale + edge)
│       │   ├── agent.py               # PPO/DQN policy setup
│       │   ├── reward_functions.py    # Multi-objective reward computation
│       │   └── buffer_manager.py      # Experience replay buffer
│       │
│       ├── serving/
│       │   ├── engine.py              # AdaptiveInferenceSystem (dual-path infer)
│       │   ├── app.py                 # FastAPI + WebSocket endpoint
│       │   ├── ui.py                  # Streamlit dashboard
│       │   └── tracking.py            # MLflow session tracker
│       │
│       ├── training/
│       │   ├── train_rl.py            # Main PPO training loop
│       │   ├── pretrain_bc.py         # Behavioral Cloning warm-start
│       │   └── profile_models.py      # Generates model_performance_profile.csv
│       │
│       ├── tests/
│       │   ├── test_rl.py             # Environment smoke tests
│       │   └── test_policy.py         # Policy rollout tests
│       │
│       ├── models/                    # Trained PPO weights (PPO_v6/final_adaptive_model)
│       ├── Dockerfile                 # Production image (nvidia/cuda:12.1.0)
│       ├── requirements_deploy.txt    # Serving dependencies
│       ├── export_onnx.py             # One-time ONNX export for all three YOLOs
│       └── README.md                  # Full RL + serving documentation
│
└── requirements.txt
```

---

## 4. Running the Evaluation Pipeline

```bash
# Install dependencies
pip install -r model_pipeline/requirements.txt

# Profile models (generates model_performance_profile.csv for RL training)
python model_pipeline/src/RL/training/profile_models.py

# Evaluate models
python model_pipeline/src/evaluation/evaluate.py

# Benchmark latency and throughput
python model_pipeline/src/evaluation/benchmark.py

# Generate comparison report
python model_pipeline/src/evaluation/compare_models.py

# Generate bias report
python model_pipeline/src/bias/generate_bias_report.py
```

---

## 5. RL Agent Training

The RL agent learns to route frames to the optimal YOLO variant. Training uses Behavioral Cloning as a warm-start before PPO fine-tuning (prevents policy collapse).

```bash
cd model_pipeline/src/RL

# Step 1 — profile models to get latency/accuracy data
python training/profile_models.py

# Step 2 — warm-start with Behavioral Cloning
python training/pretrain_bc.py

# Step 3 — PPO fine-tuning
python training/train_rl.py

# Test trained policy
python tests/test_policy.py
```

See [src/RL/README.md](src/RL/README.md) for full RL documentation including observation space design, reward function details, and training history.

---

## 6. Serving Stack

The trained agent is served via FastAPI + Streamlit:

```bash
cd model_pipeline/src/RL

# Backend (port 8000)
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000

# Dashboard (port 8501)
streamlit run serving/ui.py
```

Dashboard features:
- **Live camera** mode (HTTPS required for browser camera access)
- **Video upload** mode with frame-skip slider (1/2/4/8/16× speedup)
- **Dual video feeds** — RL-adaptive (Path A) vs YOLOv8-Small baseline (Path B)
- **Live metrics** — latency, confidence, model selection distribution

For production deployment on Kubernetes, see [infra/README.md](../infra/README.md).

---

## 7. Configuration Files

| File | Controls |
|---|---|
| `configs/eval/eval_config.yaml` | Evaluation split, benchmark device, output directories |
| `configs/train/train_config.yaml` | Model variants, pretrained weights, checkpoint paths |
| `configs/data/dataset_config.yaml` | Dataset paths and split files |
| `configs/tracking/mlflow_config.yaml` | MLflow experiment names and registry settings |

---

## 8. MLflow Tracking

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

**Experiments logged:**
- `pretrained_yolo_evaluation` — mAP50, mAP50-95, precision, recall
- `pretrained_yolo_benchmark` — latency, throughput per workload bucket
- `pretrained_yolo_summary` — cross-model comparison

**RL sessions** are logged automatically when a WebSocket client disconnects (see `serving/tracking.py`).

---

## 9. ONNX Export (deployed inference)

The engine automatically uses ONNX models when available — they're 2-3× faster on CPU than PyTorch `.pt` files. To export:

```bash
cd model_pipeline/src/RL
python export_onnx.py
# Produces yolov8n.onnx, yolov8s.onnx, yolov8l.onnx
```

In deployment, ONNX files are stored on a PVC at `/app/models/`. The engine checks that path automatically — no code change needed.

---

## 10. Model Selection Conclusion

Based on benchmarking results:

- **YOLO Nano** — fastest, best for simple low-clutter scenes
- **YOLO Small** — best balance of speed and accuracy, strongest fixed baseline
- **YOLO Large** — highest accuracy, but 3× slower and more expensive

These results confirm that no single model is optimal across all scene types, which motivates the adaptive RL routing strategy.
