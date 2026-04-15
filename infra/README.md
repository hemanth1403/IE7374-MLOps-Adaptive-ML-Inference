# Infrastructure

Production deployment infrastructure for the Adaptive ML Inference platform. Supports two deployment modes: Docker Compose (single-node) and Kubernetes (GKE).

---

## Directory Structure

```
infra/
├── k8s/                           # Kubernetes manifests (GKE)
│   ├── namespace.yaml             # adaptive-inference namespace
│   ├── configmap.yaml             # Environment variables (model paths, MLflow URI)
│   ├── pvc.yaml                   # PersistentVolumeClaim for model weights
│   ├── backend-deployment.yaml    # FastAPI inference server
│   ├── ui-deployment.yaml         # Streamlit dashboard
│   ├── mlflow-deployment.yaml     # MLflow tracking server
│   ├── prometheus-deployment.yaml # Prometheus metrics
│   ├── grafana-deployment.yaml    # Grafana dashboards
│   ├── ingress.yaml               # NGINX ingress with TLS + WebSocket support
│   └── hpa.yaml                   # Horizontal Pod Autoscaler
│
├── docker/
│   └── docker-compose.prod.yaml   # Full stack on a single GPU node
│
└── monitoring/
    ├── prometheus-configmap.yaml   # Prometheus scrape config
    └── grafana-dashboard.json      # Pre-built inference metrics dashboard
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
