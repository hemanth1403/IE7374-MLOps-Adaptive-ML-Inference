# Run Locally — Adaptive ML Inference System

## Directory Structure

```
RL/
├── core/        # RL components: agent, environment, features, reward_functions, buffer_manager
├── serving/     # Production stack: engine, app (FastAPI), ui (Streamlit), tracking
├── training/    # Train and profile scripts
├── scripts/     # Standalone scripts: live inference, orchestrator, evaluate
├── tests/       # Test scripts
├── assets/      # Diagrams and sample images
├── models/      # Trained model weights (PPO/, PPO_v1/, rl_bandit_v1.pth)
└── logs/        # Training logs
```

All commands below assume you are in the `RL/` root:
```bash
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL
```

---

## Step 1 — Activate the environment and install dependencies

```bash
conda activate IE7374

# Core RL + training dependencies
pip install -r requirements.txt

# Serving stack (FastAPI, Streamlit, MLflow, websocket)
pip install -r requirements_deploy.txt
```

> **Note:** `protobuf<4.0` is required — MLflow 2.10.2's generated stubs use
> `google.protobuf.service` which was removed in protobuf 4.x.

---

## Step 2 — Profile models (generates training data for the RL agent)

Only needed if `model_performance_profile.csv` does not exist:

```bash
python training/profile_models.py
```

Output: `model_performance_profile.csv` in the `RL/` root.

---

## Step 3 — Train the RL agent

Two training strategies are available:

### Option A — Behavioral Cloning (BC) warm-start (recommended)

Trains a supervised classifier on `(observation → optimal_action)` labels derived from the profiling
CSV, then optionally fine-tunes with PPO.  Produces a **balanced** routing policy (~33% each).

```bash
python training/pretrain_bc.py
```

Output: `models/PPO_v6/final_adaptive_model.zip`

> **How it works**: For each image the optimal model is computed analytically from the
> profiling CSV.  A 2-layer MLP (256-256) is trained via cross-entropy, then its weights
> are injected into a SB3 PPO shell.  This avoids the value-function collapse that causes
> pure RL to always pick YOLOv8-Small.

### Option B — Pure PPO RL training

```bash
python training/train_rl.py
```

Checkpoints are saved to `models/PPO_v5/`. Final model: `models/PPO_v5/final_adaptive_model.zip`.

> **Note**: Pure PPO tends to collapse to YOLOv8-Small (~70-90%) due to the value function
> failing to converge (explained_variance ≈ 0).  Use Option A for balanced routing.

---

## Step 4 — Test the trained policy

```bash
# Run 1000-step policy evaluation (model selection breakdown + avg reward)
python tests/test_policy.py

# Smoke-test with static images (Parking / Wall / Computer)
python tests/static_hypothesis_test.py

# Verify dataset image paths are resolvable
python tests/test_paths.py
```

Expected output for `test_policy.py` (BC-trained PPO_v6):
```
Testing model on 1000 samples...

--- TEST RESULTS ---
Action 0 (Nano) chosen:  ~340 times (~34.0%)
Action 1 (Small) chosen: ~340 times (~34.0%)
Action 2 (Large) chosen: ~320 times (~32.0%)
Average Reward: ~0.51
```

---

## Step 5 — Smoke-test the inference engine

Before starting the server, verify the engine loads and runs correctly:

```bash
python - <<'EOF'
from serving.engine import AdaptiveInferenceSystem
import numpy as np

system = AdaptiveInferenceSystem(
    rl_model_path="models/PPO/final_adaptive_model.zip",
    yolo_n_path="yolov8n.pt",
    yolo_s_path="yolov8s.pt",
    yolo_l_path="yolov8l.pt",
    device="cuda",         # use "cpu" if no GPU
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
[Engine] Loading PPO agent from: models/PPO/final_adaptive_model.zip
[Engine] Loading YOLO n/s/l on cuda …
[Engine] Warm-up complete.
[Engine] Ready.
Adaptive model: Nano        ← (or Small / Large)
Adaptive latency: ~4–15 ms
Baseline latency: ~4–12 ms
```

---

## Step 6 — Start the FastAPI backend

Open a terminal:

```bash
conda activate IE7374
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL

python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

> **Always use `python -m uvicorn`** (not bare `uvicorn`). The system `uvicorn`
> binary runs under the system Python and does not have `cv2`, `torch`, or conda packages.

Wait for:
```
[Engine] Warm-up complete.
[Engine] Ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify the health endpoint in another terminal:
```bash
curl http://localhost:8000/health
# → {"status":"ok","engine_ready":true}
```

---

## Step 7 — Start the Streamlit dashboard

Open a **second** terminal (keep the backend running):

```bash
conda activate IE7374
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL

python -m streamlit run serving/ui.py
```

> **Always use `python -m streamlit`** — same reason as uvicorn above.

Opens at `http://localhost:8501`. Toggle **"Start Stream"** — the webcam activates, both annotated feeds appear side-by-side, and the latency chart begins updating.

---

## Step 8 — Verify MLflow logged the session

After toggling the stream off (or closing the browser tab):

```bash
conda activate IE7374
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL

python -m mlflow ui --port 5000
```

Open `http://localhost:5000` → experiment **"adaptive_inference"** → run with:

| Metric | Description |
|---|---|
| `avg_adaptive_latency_ms` | Mean latency of the RL-selected model |
| `avg_baseline_latency_ms` | Mean latency of fixed YOLOv8-Small |
| `latency_savings_ms` | baseline − adaptive (positive = adaptive was faster) |
| `total_frames` | Total frames processed in the session |
| `model_pct_nano/small/large` | % of frames routed to each YOLO variant |

---

## Optional — Live inference scripts (no web UI)

```bash
# Live inference using PPO agent (webcam)
python scripts/live_adaptive_inference.py

# Live inference using neural bandit (older DQN agent)
python scripts/live_inference.py
```

---

## Docker

```bash
cd ~/IE7374-MLOps-Adaptive-ML-Inference/model_pipeline/src/RL

# Build
docker build -t adaptive-inference .

# Run FastAPI backend (GPU)
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/yolov8n.pt:/app/yolov8n.pt:ro \
  -v $(pwd)/yolov8s.pt:/app/yolov8s.pt:ro \
  -v $(pwd)/yolov8l.pt:/app/yolov8l.pt:ro \
  adaptive-inference

# Run Streamlit dashboard (connect to remote backend)
docker run -p 8501:8501 \
  -e WS_URL=ws://<backend-host>:8000/ws/stream \
  adaptive-inference \
  streamlit run serving/ui.py --server.port=8501 --server.address=0.0.0.0
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'core'` | Ensure you are running from the `RL/` root directory, not a subdirectory |
| `ModuleNotFoundError: cv2` from uvicorn | Use `python -m uvicorn` not bare `uvicorn` |
| `module 'websocket' has no attribute 'create_connection'` | `pip uninstall websocket -y && pip install websocket-client` |
| `No supported WebSocket library detected` (uvicorn warning) | `pip install websockets`, then restart the backend |
| `ImportError: cannot import name 'service' from 'google.protobuf'` | `pip install "protobuf>=3.20,<4.0"` |
| `Handshake status 404 Not Found` on `/ws/stream` | Restart backend; check backend terminal for import errors |
| `weights_only` TypeError | The patch in `serving/engine.py` handles this; ensure `engine` is imported first |
| `ValueError: Unexpected observation shape (1031,)` | Fixed — was a pre-existing bug in `static_hypothesis_test.py`; uses 1028-dim now |
| Webcam not found | Try `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)` if you have multiple cameras |
| Agent always picks YOLOv8-Small (90%+ of frames) | Pure PPO collapses due to value-function failure; retrain with `python training/pretrain_bc.py` |
| Adaptive latency consistently higher than baseline | Expected if agent picks Small/Large frequently; `decision_interval=5` amortises RL overhead |
| CUDA OOM | `INFERENCE_DEVICE=cpu python -m uvicorn serving.app:app --port 8000` |
