# Infrastructure

Production deployment infrastructure for the Adaptive ML Inference platform. Supports two deployment modes: Docker Compose (single-node) and Kubernetes (GKE).

---

## Directory Structure

```
infra/
├── k8s/                               # Kubernetes manifests (GKE)
│   ├── namespace.yaml                 # adaptive-inference namespace
│   ├── configmap.yaml                 # Environment variables (model paths, MLflow URI)
│   ├── pvc.yaml                       # PersistentVolumeClaim for model weights
│   ├── backend-deployment.yaml        # FastAPI inference server
│   ├── ui-deployment.yaml             # Streamlit dashboard
│   ├── mlflow-deployment.yaml         # MLflow tracking server
│   ├── backend-service.yaml           # ClusterIP for FastAPI
│   ├── ui-service.yaml                # ClusterIP for Streamlit
│   ├── mlflow-service.yaml            # ClusterIP for MLflow
│   ├── ingress.yaml                   # NGINX ingress with TLS + WebSocket support
│   ├── hpa.yaml                       # Horizontal Pod Autoscaler (UI only)
│   └── drift-detector-cronjob.yaml    # CronJob — runs drift detector every 6 hours
│
├── docker/
│   └── docker-compose.prod.yaml       # Full stack on a single GPU node
│
└── monitoring/
    ├── prometheus-configmap.yaml       # Prometheus scrape config + rule_files ref
    ├── prometheus-deployment.yaml      # Prometheus v2.48.0 with 7-day retention
    ├── prometheus-rules.yaml           # Alerting rules (latency, drift, routing)
    ├── grafana-deployment.yaml         # Grafana v10.2.0
    └── grafana-dashboard.json          # Pre-built inference metrics dashboard
```

---

## Option 1 — Docker Compose (single GPU node)

Runs the full stack (backend, UI, MLflow, Prometheus, Grafana) on one machine with a GPU.

**Prerequisites:** Docker, nvidia-container-toolkit, model weights at `model_pipeline/src/RL/models/`

```bash
cd infra/docker
docker compose -f docker-compose.prod.yaml up -d
```

| Service | URL |
|---|---|
| Inference backend | http://localhost:8000 |
| Streamlit dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

To stop:
```bash
docker compose -f docker-compose.prod.yaml down
```

---

## Option 2 — Kubernetes (GKE)

For the full GKE production deployment guide including cluster setup, CI/CD, TLS, and DNS, see [DEPLOYMENT_INTERNAL.md](../DEPLOYMENT_INTERNAL.md).

### Quick deploy to an existing cluster

```bash
# Create namespace and base resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml

# Deploy services
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/ui-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml

# Ingress and autoscaling
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Verify
kubectl get pods -n adaptive-inference
```

### Key manifest details

**`backend-deployment.yaml`** — FastAPI server (`serving/app.py`). Health checks at `/health/startup`, `/health/ready`, `/health`. Model weights loaded from PVC at `/app/models/`.

**`ui-deployment.yaml`** — Streamlit dashboard (`serving/ui.py`). Connects to backend via `ws://backend-service:8000/ws/stream`. Upload size limit 500MB.

**`ingress.yaml`** — NGINX ingress with:
- WebSocket passthrough (`proxy-read-timeout: 3600`)
- 500MB body limit for video upload
- TLS via `adaptive-inference-tls` secret
- `proxy-http-version: 1.1` for WebSocket compatibility

**`hpa.yaml`** — Autoscales backend on CPU utilization. Adjust `minReplicas`/`maxReplicas` as needed.

**`pvc.yaml`** — 10Gi PVC (`model-weights-pvc`) for ONNX/PyTorch model weights. Persists across pod restarts.

### Copying model weights to the PVC

```bash
BACKEND=$(kubectl get pod -n adaptive-inference -l app=backend -o jsonpath='{.items[0].metadata.name}')

# ONNX models (faster CPU inference)
kubectl cp yolov8n.onnx adaptive-inference/$BACKEND:/app/models/yolov8n.onnx
kubectl cp yolov8s.onnx adaptive-inference/$BACKEND:/app/models/yolov8s.onnx
kubectl cp yolov8l.onnx adaptive-inference/$BACKEND:/app/models/yolov8l.onnx
```

If the `/app/models/` directory is root-owned, fix permissions first:

```bash
kubectl run fix-perms --image=busybox --restart=Never -n adaptive-inference \
  --overrides='{"spec":{"securityContext":{"runAsUser":0},"volumes":[{"name":"m","persistentVolumeClaim":{"claimName":"model-weights-pvc"}}],"containers":[{"name":"fix","image":"busybox","command":["chmod","777","/mnt"],"volumeMounts":[{"name":"m","mountPath":"/mnt"}]}]}}'
kubectl wait --for=condition=Succeeded pod/fix-perms -n adaptive-inference --timeout=30s
kubectl delete pod fix-perms -n adaptive-inference
```

---

## Monitoring

### Prometheus

Scrapes metrics from the backend's `/metrics` endpoint (Prometheus client). Default scrape interval: 15s. Config is in `monitoring/prometheus-configmap.yaml`.

```bash
# Check what's being scraped
kubectl get configmap -n adaptive-inference prometheus-config -o yaml
```

### Grafana

Pre-built dashboard (`monitoring/grafana-dashboard.json`) shows:
- Inference latency (adaptive vs baseline)
- Model selection distribution (nano/small/large)
- Active WebSocket connections
- Request throughput

Import the dashboard: Grafana → Dashboards → Import → upload `grafana-dashboard.json`.

Default credentials: `admin / admin` (change on first login).

### Prometheus Alert Rules

`monitoring/prometheus-rules.yaml` defines four alerting rules loaded automatically by Prometheus:

| Alert | Condition | Severity |
|---|---|---|
| `HighAdaptiveLatencyP99` | p99 adaptive latency > 500ms for 5 min | warning |
| `HighBaselineLatencyP99` | p99 baseline latency > 1s for 5 min | warning |
| `LowFrameThroughput` | Frame rate < 0.01 fps for 10 min | info |
| `HighWebSocketConnectionCount` | > 20 concurrent connections for 2 min | warning |
| `ModelRoutingImbalance` | RL routes > 85% frames to Large for 30 min | warning |

---

## Drift Detection & Automated Retraining

### How it works

1. `infra/k8s/drift-detector-cronjob.yaml` runs every 6 hours as a K8s CronJob
2. It executes `monitoring/drift_detector.py` inside the same Docker image
3. The script pulls recent session metrics from MLflow and checks:
   - Threshold violations (latency > 150ms, savings < 5ms, confidence < 0.30)
   - Distribution drift via Evidently AI (compares recent vs. reference sessions)
4. If drift is detected, `monitoring/retrain_trigger.py` fires a GitHub `repository_dispatch` event
5. This starts the `retrain.yml` GitHub Actions workflow which:
   - Retrains the model with `dvc repro`
   - Runs the quality gate (`pct_nano >= 20%`, `pct_large >= 20%`)
   - Builds a new Docker image and does a rolling update on GKE
   - Sends Slack notifications at start, success, and failure

### Required secrets

Create the `retraining-secrets` K8s Secret before deploying the CronJob:

```bash
kubectl create secret generic retraining-secrets \
  --from-literal=github-token=<GitHub PAT with repo scope> \
  --from-literal=slack-webhook-url=<Slack incoming webhook URL> \
  -n adaptive-inference
```

`slack-webhook-url` is optional — leave it empty if Slack notifications are not needed.

### Required GitHub Actions secrets

| Secret | Purpose |
|---|---|
| `GKE_SA_KEY` | GCP service account JSON for GKE authentication |
| `GCP_SA_KEY_JSON` | Same SA JSON for DVC GCS remote access in `retrain.yml` |

### Tuning drift thresholds

Thresholds are set as environment variables in `drift-detector-cronjob.yaml` and can be changed without rebuilding the image:

| Variable | Default | Meaning |
|---|---|---|
| `LATENCY_THRESHOLD_MS` | 150 | Max acceptable avg adaptive latency |
| `SAVINGS_THRESHOLD_MS` | 5 | Min latency savings vs baseline |
| `CONFIDENCE_THRESHOLD` | 0.30 | Min average detection confidence |
| `DRIFT_THRESHOLD` | 0.3 | Evidently drift score to trigger retrain |
| `MIN_SESSIONS` | 10 | Min sessions before drift checks run |
