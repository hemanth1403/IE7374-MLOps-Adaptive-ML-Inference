# RL-Powered Adaptive Multi-Model Inference System

A reinforcement learning agent that dynamically routes each video frame to the most
appropriate YOLOv8 variant (Nano / Small / Large), trading off detection quality,
inference latency, and compute cost in real time.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Concepts](#3-core-concepts)
4. [Setup & Installation](#4-setup--installation)
5. [Step-by-Step Execution Guide](#5-step-by-step-execution-guide)
   - [Step 1 — Profile the YOLO models](#step-1--profile-the-yolo-models)
   - [Step 2 — Train the RL routing policy](#step-2--train-the-rl-routing-policy)
   - [Step 3 — Test the trained policy](#step-3--test-the-trained-policy)
   - [Step 4 — Smoke-test the inference engine](#step-4--smoke-test-the-inference-engine)
   - [Step 5 — Start the FastAPI backend](#step-5--start-the-fastapi-backend)
   - [Step 6 — Launch the Streamlit dashboard](#step-6--launch-the-streamlit-dashboard)
   - [Step 7 — Verify MLflow tracking](#step-7--verify-mlflow-tracking)
   - [Step 8 — Docker deployment](#step-8--docker-deployment)
6. [Reproducibility with DVC](#6-reproducibility-with-dvc)
   - [Pipeline overview](#pipeline-overview)
   - [Clone and run — skip retraining](#clone-and-run--skip-retraining)
   - [Clone and reproduce from scratch](#clone-and-reproduce-from-scratch)
   - [Experiment with params.yaml](#experiment-with-paramsyaml)
   - [Configure a DVC remote](#configure-a-dvc-remote)
7. [Architecture Deep Dive](#7-architecture-deep-dive)
8. [Training Journey: What We Tried, What Failed, What Worked](#8-training-journey-what-we-tried-what-failed-what-worked)
9. [Final Policy Performance](#9-final-policy-performance)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Overview

```
Webcam frame
     │
     ▼
FeatureExtractor ─── 32×32 grayscale pixels (1024-dim) + Canny edge density
     │
     ▼
PPO Agent (BC-trained) ─── picks: Nano | Small | Large
     │                                 │
     ▼                                 ▼
YOLOv8 (selected)           YOLOv8-Small (fixed baseline)
     │                                 │
     └──────────┬──────────────────────┘
                ▼
     JSON result: adaptive + baseline paths
                ▼
     Streamlit UI  ◄──  WebSocket  ◄── FastAPI /ws/stream
                ▼
     MLflow session summary
```

**Key idea:** Instead of always running the same YOLO model, the system uses scene
complexity signals to pick the cheapest model that can still handle the current frame.
A parking lot scene → Nano (fast, cheap).  A busy intersection → Large (accurate).

---

## 2. Directory Structure

```
RL/
├── core/                          # RL components
│   ├── environment.py             # Gymnasium env — observation, action, reward
│   ├── features.py                # FeatureExtractor — 32×32 pixels + edge density
│   ├── agent.py                   # NeuralBanditAgent (early prototype, archived)
│   ├── reward_functions.py        # RewardCalculator (early prototype, archived)
│   └── buffer_manager.py          # WindowBufferManager (early prototype, archived)
│
├── serving/                       # Production inference stack
│   ├── engine.py                  # AdaptiveInferenceSystem — loads models, runs dual-path infer()
│   ├── app.py                     # FastAPI server — WebSocket /ws/stream endpoint
│   ├── ui.py                      # Streamlit dashboard — dual video + latency chart
│   └── tracking.py                # SessionTracker — logs MLflow summary on disconnect
│
├── training/                      # Training scripts
│   ├── profile_models.py          # Benchmarks YOLO n/s/l on the dataset → CSV
│   ├── train_rl.py                # Pure PPO training (produces collapsed policy)
│   └── pretrain_bc.py             # Behavioral Cloning warm-start → balanced policy
│
├── tests/                         # Validation scripts
│   ├── test_policy.py             # 1000-step rollout: action distribution + avg reward
│   ├── static_hypothesis_test.py  # Smoke-test on 3 fixed images
│   ├── test_rl.py                 # Environment unit tests
│   └── test_paths.py              # Checks all CSV image paths are resolvable
│
├── scripts/                       # Standalone utilities
│   ├── live_adaptive_inference.py # Live webcam with PPO agent (no web UI)
│   ├── live_inference.py          # Live webcam with neural bandit (older agent)
│   ├── evaluate.py                # Offline evaluation helper
│   └── orchestrator.py            # Multi-service orchestration helper
│
├── models/                        # Trained model weights
│   ├── PPO_v6/                    # CURRENT — BC-trained balanced policy
│   │   ├── final_adaptive_model.zip   ← production model
│   │   └── bc_init_model.zip          ← BC weights (before any fine-tuning)
│   ├── PPO_v5/                    # Pure PPO (Small-collapsed, ~71% Small)
│   ├── PPO_v4/                    # Ranking-norm reward (Small-collapsed, ~75%)
│   ├── PPO_v3/                    # Quality+efficiency reward (Small-collapsed, ~72%)
│   ├── PPO_v2/                    # Alpha=1.5 fix (Nano=0.8%, Small=55%, Large=44%)
│   ├── PPO/                       # First PPO run (alpha=3.0, Large=~0%)
│   └── rl_bandit_v1.pth           # Neural bandit baseline (early prototype)
│
├── model_performance_profile.csv  # Pre-computed YOLO benchmarks per image
├── Dockerfile                     # Production container (CUDA 12.1 + Ubuntu 22.04)
├── requirements.txt               # Training dependencies
├── requirements_deploy.txt        # Serving dependencies (FastAPI + Streamlit + MLflow)
└── yolov8n.pt / yolov8s.pt / yolov8l.pt   # YOLO weights (not in git)
```

---

## 3. Core Concepts

### Observation Space (1028-dim)

Every frame is converted into a flat vector before the RL agent sees it:

| Slice      | Size | Content                                              |
|------------|------|------------------------------------------------------|
| `[0:1024]` | 1024 | 32×32 grayscale downsampled image, flattened, ÷255   |
| `[1024]`   | 1    | Canny edge density × 10 (proxy for scene busyness)  |
| `[1025]`   | 1    | Previous action / 2.0 (normalised to [0, 1])         |
| `[1026]`   | 1    | Previous detection confidence                         |
| `[1027]`   | 1    | Padding zero                                          |

### Action Space

Three discrete actions:

| Action | Model        | Use case                          |
|--------|--------------|-----------------------------------|
| 0      | YOLOv8-Nano  | Simple/sparse scenes, fast ~4 ms  |
| 1      | YOLOv8-Small | Default balanced model ~8 ms      |
| 2      | YOLOv8-Large | Complex/dense scenes ~20+ ms      |

### Reward Function (environment.py)

For each step the reward is computed purely from the pre-profiled CSV — no live YOLO
inference during training:

```
quality[a]    = conf[a] × √(count[a] + 1)
efficiency[a] = quality[a] / latency[a]

score[a] = 0.84 × (quality[a] / max_quality) + 0.16 × (efficiency[a] / max_efficiency)

reward = (score[action] - min_score) / (max_score - min_score)   # ∈ [0, 1]
reward -= 0.02   if model switched from previous step
```

This ranking normalisation fills the full [0, 1] range every step regardless of scene
difficulty, giving PPO a strong advantage signal rather than a mean-shifted scalar.

### Dataset: model_performance_profile.csv

Generated by `training/profile_models.py`.  One row per image in the training split,
with pre-recorded confidence, latency, and detection count for all three YOLO models:

| Column    | Description                         |
|-----------|-------------------------------------|
| `path`    | Absolute path to image on disk      |
| `n_conf`  | Nano mean detection confidence       |
| `n_time`  | Nano inference latency (seconds)     |
| `n_count` | Nano detection count                 |
| `s_conf`  | Small mean detection confidence      |
| `s_time`  | Small inference latency              |
| `s_count` | Small detection count                |
| `l_conf`  | Large mean detection confidence      |
| `l_time`  | Large inference latency              |
| `l_count` | Large detection count                |

106,411 rows covering the COCO 2017 training split.

---

## 4. Setup & Installation

All commands assume you are in the `RL/` root:
```bash
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL
```

### Activate environment
```bash
conda activate IE7374
```

### Install training dependencies
```bash
pip install -r requirements.txt
```

### Install serving dependencies (FastAPI + Streamlit + MLflow)
```bash
pip install -r requirements_deploy.txt
```

> **protobuf note:** MLflow 2.10.2 requires `protobuf<4.0`.
> If you see `ImportError: cannot import name 'service' from 'google.protobuf'`, run:
> ```bash
> pip install "protobuf>=3.20,<4.0"
> ```

---

## 5. Step-by-Step Execution Guide

### Step 1 — Profile the YOLO models

Generates `model_performance_profile.csv` — the training data for the RL agent.
Skip this step if the file already exists.

```bash
python training/profile_models.py
```

This runs all three YOLOv8 models over every image in the COCO training split and
records confidence, latency, and detection count.  Takes ~30-60 minutes for 106K images.

Expected output:
```
Project Root Detected: /home/.../IE7374-MLOps-Adaptive-ML-Inference
Targeting Split File: .../Data-Pipeline/data/splits/train.txt
Loading YOLO models...
Profiling 106411 images. This measures actual CPU latency on your G14.
...
Profiling complete. Saved to model_performance_profile.csv
```

---

### Step 2 — Train the RL routing policy

Two strategies available.  **Option A is recommended** — it produces a balanced policy.

#### Option A — Behavioral Cloning warm-start (recommended)

```bash
python training/pretrain_bc.py
```

What it does:
1. Samples 15,000 images from the profiling CSV
2. Extracts 1028-dim observations for each image
3. Computes the analytically optimal model choice per image from the CSV metrics
4. Trains a supervised MLP classifier on `(obs → optimal_action)` — 30 epochs, ~43% accuracy
5. Injects the classifier weights into a PPO policy shell
6. Saves to `models/PPO_v6/final_adaptive_model.zip`

Expected output:
```
Extracting observations from 15000 images …
  0/15000
  2000/15000
  ...
Extraction done in ~90s.
Labels: Nano=32.7%  Small=34.4%  Large=32.9%

Training BC classifier...
  Epoch  1: loss=1.1376  acc=33.0%
  Epoch 10: loss=1.0891  acc=37.6%
  Epoch 30: loss=1.0533  acc=43.5%

BC prediction distribution:
  Nano:  35.1%
  Small: 31.6%
  Large: 33.3%

Post-BC-inject policy (500 steps, deterministic):
  Nano:  36.0%
  Small: 30.2%
  Large: 33.8%

Saved to models/PPO_v6/final_adaptive_model.zip
```

#### Option B — Pure PPO training (for reference)

```bash
python training/train_rl.py
```

Trains a PPO agent for 3,000,000 steps.  Saves to `models/PPO_v5/`.

> **Warning:** Pure PPO consistently collapses to always-Small (~70-90% of frames).
> This is a known limitation — see [Section 7](#7-training-journey-what-we-tried-what-failed-what-worked)
> for the full diagnosis.  Use Option A instead.

---

### Step 3 — Test the trained policy

```bash
# 1000-step rollout — action distribution + average reward
python tests/test_policy.py

# Smoke-test on 3 reference images (Parking / Wall / Computer)
python tests/static_hypothesis_test.py

# Verify all CSV image paths are resolvable on disk
python tests/test_paths.py
```

Expected output for `test_policy.py` (PPO_v6):
```
Testing model on 1000 samples...

--- TEST RESULTS ---
Action 0 (Nano) chosen:  343 times (34.3%)
Action 1 (Small) chosen: 341 times (34.1%)
Action 2 (Large) chosen: 316 times (31.6%)
Average Reward: 0.5082
```

---

### Step 4 — Smoke-test the inference engine

Verifies the engine loads and runs correctly before starting the server:

```bash
python - <<'EOF'
from serving.engine import AdaptiveInferenceSystem
import numpy as np

system = AdaptiveInferenceSystem(
    rl_model_path="models/PPO_v6/final_adaptive_model.zip",
    yolo_n_path="yolov8n.pt",
    yolo_s_path="yolov8s.pt",
    yolo_l_path="yolov8l.pt",
    device="cuda",   # use "cpu" if no GPU
)

dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = system.infer(dummy_frame)

print("Adaptive model:", result["adaptive"]["model_name"])
print("Adaptive latency:", result["adaptive"]["latency_ms"], "ms")
print("Baseline latency:", result["baseline"]["latency_ms"], "ms")
EOF
```

Expected output:
```
[Engine] Loading PPO agent from: models/PPO_v6/final_adaptive_model.zip
[Engine] Loading YOLO n/s/l on cuda …
[Engine] Warm-up complete.
[Engine] Ready.
Adaptive model: Nano    ← (or Small / Large)
Adaptive latency: ~4–15 ms
Baseline latency: ~4–12 ms
```

---

### Step 5 — Start the FastAPI backend

```bash
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

> Always use `python -m uvicorn` (not bare `uvicorn`).  The system `uvicorn` runs under
> system Python which doesn't have `cv2`, `torch`, or conda packages.

Wait for:
```
[Engine] Warm-up complete.
[Engine] Ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify:
```bash
curl http://localhost:8000/health
# → {"status":"ok","engine_ready":true}
```

**WebSocket protocol:**
- Client → Server: raw base64-encoded JPEG (no data-URL prefix)
- Server → Client: JSON with `adaptive` and `baseline` paths, each containing `model_name`,
  `detections`, `latency_ms`, `object_count`, `avg_confidence`

**Environment variables:**

| Variable           | Default                                | Description            |
|--------------------|----------------------------------------|------------------------|
| `RL_MODEL_PATH`    | `models/PPO_v6/final_adaptive_model.zip` | Path to PPO .zip       |
| `YOLO_N_PATH`      | `yolov8n.pt`                           | YOLOv8-Nano weights    |
| `YOLO_S_PATH`      | `yolov8s.pt`                           | YOLOv8-Small weights   |
| `YOLO_L_PATH`      | `yolov8l.pt`                           | YOLOv8-Large weights   |
| `INFERENCE_DEVICE` | `cuda`                                 | `cuda` or `cpu`        |

---

### Step 6 — Launch the Streamlit dashboard

Open a **second terminal** (keep the backend running):

```bash
python -m streamlit run serving/ui.py
```

Opens at `http://localhost:8501`.  Toggle **"Start Stream"** to activate the webcam.
The dashboard shows:
- **Left panel:** Frame annotated by the RL-selected YOLO model + model name + latency
- **Right panel:** Same frame annotated by the fixed YOLOv8-Small baseline
- **Chart:** Real-time latency comparison (adaptive vs baseline)
- **Metrics:** Per-model frame distribution, average latency savings

---

### Step 7 — Verify MLflow tracking

After stopping a stream session:

```bash
python -m mlflow ui --port 5000
```

Open `http://localhost:5000` → experiment **"adaptive_inference"**.

Logged metrics per session:

| Metric                      | Description                                        |
|-----------------------------|----------------------------------------------------|
| `avg_adaptive_latency_ms`   | Mean latency of the RL-selected model              |
| `avg_baseline_latency_ms`   | Mean latency of fixed YOLOv8-Small                 |
| `latency_savings_ms`        | baseline − adaptive (positive = adaptive faster)   |
| `total_frames`              | Total frames processed                              |
| `model_pct_nano/small/large`| % of frames routed to each YOLO variant            |

---

### Step 8 — Docker deployment

```bash
# Build
docker build -t adaptive-inference .

# Run FastAPI backend (GPU)
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/yolov8n.pt:/app/yolov8n.pt:ro \
  -v $(pwd)/yolov8s.pt:/app/yolov8s.pt:ro \
  -v $(pwd)/yolov8l.pt:/app/yolov8l.pt:ro \
  adaptive-inference

# Run Streamlit dashboard (pointing to a remote backend)
docker run -p 8501:8501 \
  -e WS_URL=ws://<backend-host>:8000/ws/stream \
  adaptive-inference \
  streamlit run serving/ui.py --server.port=8501 --server.address=0.0.0.0
```

Base image: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
Requires the NVIDIA Container Toolkit on the host.

---

## 6. Reproducibility with DVC

The RL pipeline uses [DVC](https://dvc.org) to make every step — from profiling to
training to evaluation — fully reproducible on any machine.

### Pipeline overview

```
profile  →  train  →  evaluate
```

| Stage      | Command                           | Inputs                                      | Outputs                                    |
|------------|-----------------------------------|---------------------------------------------|--------------------------------------------|
| `profile`  | `training/profile_models.py`      | COCO `train.txt` split + params             | `model_performance_profile.csv`            |
| `train`    | `training/pretrain_bc.py`         | CSV + `params.yaml` (bc / ppo / reward)     | `models/PPO_v6/final_adaptive_model.zip`   |
| `evaluate` | `training/evaluate_policy.py`     | Trained model + CSV                         | `metrics.json`                             |

DVC also tracks the YOLO weights as standalone versioned artifacts:
- `yolov8n.pt.dvc`
- `yolov8s.pt.dvc`
- `yolov8l.pt.dvc`

---

### Clone and run — skip retraining

If a DVC remote is configured (see [Configure a DVC remote](#configure-a-dvc-remote)),
you can fetch all pre-built artifacts and skip training entirely:

```bash
conda activate IE7374
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL

# Pull the CSV, trained model zip, and YOLO weights from remote storage
dvc pull

# Verify everything is healthy
dvc status           # should print: Data and pipelines are up to date.
dvc metrics show     # prints avg_reward, pct_nano, pct_small, pct_large
```

Then start the serving stack as normal:
```bash
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

---

### Clone and reproduce from scratch

If no DVC remote is available (or you want to fully reproduce):

```bash
conda activate IE7374

# Step 1 — reproduce the Data-Pipeline first (generates the COCO split files)
cd ~/IE7374-MLOps-Adaptive-ML-Inference/Data-Pipeline
dvc repro

# Step 2 — reproduce the RL pipeline
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL
pip install -r requirements.txt
dvc repro
```

DVC runs only the stages whose inputs have changed.  On a fresh clone all three
stages run in sequence:

```
Running stage 'profile':
> python training/profile_models.py
...
Running stage 'train':
> python training/pretrain_bc.py
...
Running stage 'evaluate':
> python training/evaluate_policy.py
...
--- EVALUATION RESULTS ---
Nano:   332  (33.2%)
Small:  335  (33.5%)
Large:  333  (33.3%)
Avg Reward: 0.5215
PASS: Policy is balanced.
```

The evaluate stage acts as a **quality gate** — it exits non-zero if any model variant
drops below 20% usage, failing the pipeline immediately so a broken re-train is caught
before deployment.

---

### Experiment with params.yaml

All tunable hyperparameters live in `params.yaml`.  Changing any value and running
`dvc repro` automatically re-runs only the affected downstream stages.

```
params.yaml
├── profile.device          → affects: profile
├── reward.*                → affects: train, evaluate
├── bc.*                    → affects: train, evaluate
├── ppo.*                   → affects: train, evaluate
└── paths.*                 → affects: train, evaluate
```

**Example — increase BC sample size:**

```bash
# Edit params.yaml: bc.sample_rows: 15000 → 30000
dvc repro
# DVC skips 'profile' (CSV unchanged), re-runs 'train' and 'evaluate'
```

**Compare metrics between experiments:**

```bash
# After changing a param and re-running:
dvc metrics diff
```

Output:
```
Path          Metric      HEAD    workspace    Change
metrics.json  avg_reward  0.5215  0.5341       0.0126
metrics.json  pct_nano    33.2    34.8         1.6
metrics.json  pct_small   33.5    33.1        -0.4
metrics.json  pct_large   33.3    32.1        -1.2
```

**Key params and their effect:**

| Param | Default | Effect of increasing |
|-------|---------|---------------------|
| `bc.sample_rows` | 15000 | More training data → higher BC accuracy → better routing |
| `bc.epochs` | 30 | More epochs → marginal accuracy gain (plateaus ~45%) |
| `reward.w_quality` | 0.84 | More weight on detection quality → Large chosen more often |
| `reward.w_efficiency` | 0.16 | More weight on speed → Nano chosen more often |
| `reward.switching_penalty` | 0.02 | Penalises frequent model switching → smoother routing |
| `ppo.finetune_steps` | 0 | >0 enables PPO fine-tuning after BC (risks balance collapse) |

---

### Configure a DVC remote

A DVC remote lets the team share cached artifacts (CSV, model weights) so nobody
has to re-run training from scratch after cloning.

**Google Drive (simplest):**
```bash
dvc remote add -d myremote gdrive://<folder-id>
dvc remote modify myremote gdrive_acknowledge_abuse true
dvc push    # upload all cached files
```

**S3:**
```bash
dvc remote add -d myremote s3://<bucket>/dvc-cache
dvc push
```

**SSH / local shared filesystem:**
```bash
dvc remote add -d myremote ssh://user@host:/path/to/cache
# or for a local shared drive:
dvc remote add -d myremote /mnt/shared/dvc-cache
dvc push
```

After pushing, anyone who clones the repo runs:
```bash
dvc pull   # fetches everything in one step
```

> If no remote is set up yet, run `dvc repro` to reproduce everything locally.
> Once a remote is added, `dvc push` to share, `dvc pull` on other machines.

---

## 7. Architecture Deep Dive


### serving/engine.py — AdaptiveInferenceSystem

The production-grade inference engine:

- Loaded once at server startup via FastAPI `lifespan`
- PPO agent runs on CPU; all three YOLO models run on GPU
- `decision_interval=5`: the RL agent re-evaluates every 5 frames to amortise its overhead
- `reset_state()` must be called at the start of each new WebSocket session so history
  from a previous client does not bleed into a new connection
- Applies a PyTorch 2.6+ `weights_only=False` patch at import time — `engine.py` **must**
  be imported before any SB3 or YOLO import

### serving/app.py — FastAPI WebSocket

```
POST /health     → liveness probe
WS   /ws/stream  → one WebSocket = one inference session
```

Each connection:
1. Calls `engine.reset_state()` for clean RL context
2. Starts a new MLflow run via `SessionTracker`
3. Receives base64-encoded JPEG frames
4. Calls `engine.infer(frame)` → returns dual-path JSON
5. On disconnect: calls `tracker.finalize()` to log session summary

### core/environment.py — AdaptiveInferenceEnv

Custom Gymnasium environment used only during training:

- **Observation:** 1028-dim vector (see Section 3)
- **Action:** Discrete(3) — Nano / Small / Large
- **Reward:** Ranking-normalised quality+efficiency score (see Section 3)
- **Episode length:** 2048 steps with a random start position in the CSV
  (avoids the value-function horizon collapse that 106K-step episodes cause)

### training/pretrain_bc.py — Behavioral Cloning Trainer

Key functions:
- `build_dataset()` — loads images, extracts 1028-dim observations, computes optimal action labels
- `train_bc()` — trains a 3-layer MLP (1028→256→256→3) via cross-entropy
- `inject_bc_weights()` — copies the MLP weights into the SB3 PPO policy's
  `mlp_extractor.policy_net` and `action_net` layers

---

## 8. Training Journey: What We Tried, What Failed, What Worked

This section documents every training attempt, the reasoning behind each change,
and why it succeeded or failed.  It is intended as a reference for anyone debugging
or extending the routing policy.

---

### The Goal

Train an RL agent that routes frames to the appropriate YOLO model with roughly
balanced usage: ~33% Nano / ~33% Small / ~33% Large.

Analytical verification showed the dataset supports this: for each image, computing
the optimal model from the CSV metrics gives Nano=31.7%, Small=35.2%, Large=33.1% —
a near-uniform distribution.

---

### Attempt 1 — First PPO run (PPO folder)

**Config:** `alpha=3.0` latency penalty inside reward, `ent_coef=0.01`, 1M steps.

**Result:**
```
Nano:  ~0%   Small: ~100%   Large: ~0%
```

**Why it failed:**
- `alpha=3.0` applied an exponential latency penalty that crushed Large's reward
  (Large is 3-5× slower than Small)
- The policy learned to never touch Large
- Small won by default on most frames

---

### Attempt 2 — Alpha reduction (PPO_v2)

**Change:** `alpha=1.5` (halved latency penalty), `ent_coef=0.03`.

**Result:**
```
Nano: 0.8%   Small: 55%   Large: 44%
```

**What improved:** Large recovered to 44% — the latency penalty was no longer
crushing it.

**What still failed:** Nano was still essentially ignored.

**Why:** Nano has the lowest average quality score (`conf × √count`) on the COCO
dataset because it detects fewer objects with lower confidence.  The reward function
didn't give Nano a strong enough signal for simple scenes.

---

### Attempt 3 — Efficiency term added (PPO_v3)

**Change:** Replaced the alpha-latency reward with:
```
score[a] = 0.84 × (quality[a] / max_quality) + 0.16 × (efficiency[a] / max_efficiency)
```
where `efficiency = quality / latency` (quality per unit time).

**Reasoning:** Nano processes ~67 quality-units/second vs Small's ~45 — Nano is
more efficient on simple scenes.  The 0.16 efficiency weight should reward it.

**Result:**
```
Nano: 0.3%   Small: 72%   Large: 28%
```

**Why it still failed:**
- Even though Nano is more efficient, Small's higher raw quality dominates the
  0.84 quality weight on most COCO frames (COCO is dense with objects)
- The 0.16 efficiency weight wasn't strong enough to compensate

---

### Attempt 4 — Ranking normalisation (PPO_v4)

**Change:** Reward replaced with rank-normalised score:
```
reward = (score[action] - min_score) / (max_score - min_score)   ∈ [0, 1]
```
This fills the full reward range every step regardless of scene difficulty.

**Reasoning:** Previously all three scores were clustered near the same value on
most frames, giving PPO weak gradient signal.  Rank normalisation spreads them to
[0, 1] always.

**Result:**
```
Nano: 3.6%   Small: 75%   Large: 21%
```

**Improvement:** Nano went from 0.3% to 3.6%, but still far from 33%.

---

### Attempt 5 — Short episodes + random starts (PPO_v5)

**Change:**
- `episode_length = 2048` (was 106K — the full dataset)
- Random start position each episode
- `ent_coef = 0.05`
- 3M training steps

**Reasoning:** With 106K-step episodes and `gamma=0.99`, the effective horizon is
`1 / (1 - 0.99) = 100` steps.  Beyond step ~460, `0.99^460 ≈ 0`.  The value
function had to predict returns over a horizon it mathematically couldn't see,
so it learned nothing (`explained_variance ≈ 0`).  Short episodes fix this.

**Result:**
```
Nano: 2.2%   Small: 71%   Large: 26%
Average Reward: 0.5754
explained_variance: -0.000242  (still near 0)
```

**Why it still failed:** The explained_variance stayed near 0 despite short
episodes.  Root cause: with the policy collapsing to Small early, the value
function sees near-constant rewards every episode (always Small = predictable
reward), so it has nothing to learn from.  This is a **degenerate training loop**:

```
bad value function
        ↓
no useful baseline for policy gradient
        ↓
policy drifts to highest-mean-reward action (Small)
        ↓
constant rewards every episode
        ↓
value function stays bad
```

---

### Root Cause Analysis

Before attempting further reward engineering, we investigated two questions:

**Question 1: Is the dataset actually balanced?**

Running the optimal-action analysis on the full 106K-row CSV:
```
Nano optimal:  33,739 rows  (31.7%)
Small optimal: 37,425 rows  (35.2%)
Large optimal: 35,247 rows  (33.1%)
```
Yes — the data supports a balanced policy.  The problem was in training, not the data.

**Question 2: Are the visual features informative?**

Edge density across optimal-action groups:
```
Nano-optimal:  mean edge density = 0.099
Small-optimal: mean edge density = 0.097
Large-optimal: mean edge density = 0.111
```
Edge density barely separates the classes.  The 1024-dim grayscale pixels have
more signal but PPO can't learn to use them because the value function never
converges.

**Question 3: What is the actual policy distribution?**

Inspecting the PPO_v5 probability outputs:
```
Mean P(Nano):  0.136 ± 0.078
Mean P(Small): 0.532 ± 0.183
Mean P(Large): 0.332 ± 0.176
Policy entropy: 0.87 / 1.10 max
```
The policy is diffuse (entropy near max) but systematically biased toward Small.
Nano gets 13.6% probability on average but is the argmax only 1% of the time —
because Small's probability is almost always above Nano's.

**Conclusion:** PPO is making statistically sound decisions given its limited
value function — it gravitates to the highest-mean-reward action when it can't
distinguish states.  No amount of reward shaping fixes this without fixing the
value function.

---

### Attempt 6 — PPO fine-tuning from BC init with low LR (failed)

**Change:** BC pre-training → inject into PPO → fine-tune with `ent_coef=0.02`, `lr=5e-5`.

**Result after 200K steps:**
```
Nano: 0.1%   Small: 93%   Large: 6.6%
```

**Why:** The PPO gradient signal (reward differential: Small=0.595 vs Nano=0.418)
immediately overpowered the BC initialisation.  Even starting balanced, PPO
re-collapses within 200K steps.

---

### Attempt 7 — PPO fine-tuning with high entropy (failed differently)

**Change:** BC pre-training → inject → fine-tune with `ent_coef=0.5`.

**Reasoning:** With a very high entropy bonus, the policy cost of collapsing
should outweigh the reward differential.

**Result after 200K steps:**
```
Nano: 1.4%   Small: 13%   Large: 86%
```

**Why:** High entropy suppressed learning generally, and the value function's
random initialisation happened to favour Large.  The high entropy just made the
policy oscillate erratically rather than converge.

---

### What Finally Worked — Behavioral Cloning (PPO_v6)

**Core insight:** PPO's value function failure is not fixable by reward engineering
alone.  The BC approach bypasses the RL training loop entirely by using **supervised
learning on analytically-derived labels**.

**Method:**
1. For each sampled image, compute which model is optimal using the profiling CSV
   (pure arithmetic — no model inference needed at train time)
2. Extract the same 1028-dim observation the environment would produce
3. Train a 3-layer MLP (1028 → 256 → 256 → 3) using cross-entropy loss
4. Copy the MLP weights into a PPO policy shell (the architectures match exactly)
5. No RL fine-tuning — the BC model is the final policy

**Why BC works where RL fails:**
- Dense supervision: every row has a ground-truth label (vs sparse RL reward signal)
- No value function: BC trains the policy network directly with gradient from labels
- No degenerate loop: the training signal doesn't depend on the current policy's quality

**Result:**
```
Nano:  343/1000 (34.3%)
Small: 341/1000 (34.1%)
Large: 316/1000 (31.6%)
Average Reward: 0.5082
```

**Trade-off:** The BC model's average reward (0.51) is lower than the collapsed PPO
model (0.58) because the PPO model's always-Small strategy is a safe bet —
it never picks the worst model, just rarely picks the best.  The BC model occasionally
picks a suboptimal action when its 43.5% classifier accuracy misses, but it achieves
the routing diversity that makes the system useful in production.

---

### Summary Table

| Version  | Approach                              | Nano  | Small | Large | Avg Reward | Notes                          |
|----------|---------------------------------------|-------|-------|-------|------------|-------------------------------|
| PPO      | Raw reward, alpha=3.0                 | ~0%   | ~100% | ~0%   | —          | Alpha crushed Large            |
| PPO_v2   | Alpha=1.5                             | 0.8%  | 55%   | 44%   | —          | Large recovered, Nano dead     |
| PPO_v3   | Quality + efficiency reward           | 0.3%  | 72%   | 28%   | —          | Efficiency term too weak       |
| PPO_v4   | Ranking normalisation                 | 3.6%  | 75%   | 21%   | —          | Better signal, still collapsed |
| PPO_v5   | Short episodes + random starts        | 2.2%  | 71%   | 26%   | 0.5754     | EV≈0, degenerate loop remains  |
| PPO_v6   | **Behavioral Cloning warm-start**     | **34.3%** | **34.1%** | **31.6%** | **0.5082** | **Production model** |

---

## 9. Final Policy Performance

Model: `models/PPO_v6/final_adaptive_model.zip`

```
Testing model on 1000 samples...

--- TEST RESULTS ---
Action 0 (Nano) chosen:  343 times (34.3%)
Action 1 (Small) chosen: 341 times (34.1%)
Action 2 (Large) chosen: 316 times (31.6%)
Average Reward: 0.5082
```

**What the policy has learned:**
- Routes scenes with few objects / high Nano confidence → Nano (fast, cheap)
- Routes complex / dense scenes → Large (accurate)
- Small fills the middle ground
- Routing is driven by the 32×32 grayscale downsampled image and edge density

**BC classifier accuracy on held-out data:** ~43.5% (vs 33.3% random baseline).
The 10% improvement above chance reflects real scene-complexity information encoded
in the downsampled grayscale features.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'core'` | Run from the `RL/` root directory, not a subdirectory |
| `ModuleNotFoundError: cv2` from uvicorn | Use `python -m uvicorn`, not bare `uvicorn` |
| `module 'websocket' has no attribute 'create_connection'` | `pip uninstall websocket -y && pip install websocket-client` |
| `No supported WebSocket library detected` | `pip install websockets`, then restart backend |
| `ImportError: cannot import name 'service' from 'google.protobuf'` | `pip install "protobuf>=3.20,<4.0"` |
| `Handshake status 404 Not Found` on `/ws/stream` | Restart backend; check terminal for import errors |
| `weights_only` TypeError on model load | `serving/engine.py` patches this at import time — ensure `engine` is imported before SB3/YOLO |
| `ValueError: Unexpected observation shape (1031,)` | Pre-existing bug in old `static_hypothesis_test.py` — fixed; env uses 1028-dim (1024+1+3) |
| Agent always picks YOLOv8-Small (90%+ of frames) | Pure PPO value-function collapse; retrain with `python training/pretrain_bc.py` |
| Webcam not found | Try `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)` |
| Adaptive latency higher than baseline | Expected if agent picks Large frequently; `decision_interval=5` amortises RL overhead |
| CUDA out of memory | `INFERENCE_DEVICE=cpu python -m uvicorn serving.app:app --port 8000` |
| Device mismatch: `Expected all tensors on same device` | Load PPO with `device='cpu'`: `PPO.load(path, device='cpu')` |
