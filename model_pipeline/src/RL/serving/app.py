"""
app.py — FastAPI backend for the Adaptive ML Inference System.

Endpoints
---------
GET  /health          — liveness probe (always 200 if process is alive)
GET  /health/startup  — startup probe (200 after engine fully loaded + warmed up)
GET  /health/ready    — readiness probe (200 when engine is ready to serve)
GET  /metrics         — Prometheus metrics (latency, model selection, frame count)
WS   /ws/stream       — streaming inference over WebSocket

WebSocket protocol
------------------
Client → Server : JSON  {"frame": "<base64 JPEG>", "baseline_model": "Nano"|"Small"|"Large"}
Server → Client : JSON packet:
    {
        "adaptive": {
            "model_name":     "Nano" | "Small" | "Large",
            "detections":     [{bbox, confidence, class_id, class_name}, …],
            "latency_ms":     float,
            "object_count":   int,
            "avg_confidence": float
        },
        "baseline": { … same shape … }
    }

Environment variables
---------------------
RL_MODEL_PATH      path to PPO .zip  (default: models/PPO_v6/final_adaptive_model.zip)
YOLO_N_PATH        yolov8n.pt        (default: yolov8n.pt)
YOLO_S_PATH        yolov8s.pt        (default: yolov8s.pt)
YOLO_L_PATH        yolov8l.pt        (default: yolov8l.pt)
INFERENCE_DEVICE   cuda | cpu        (default: cuda)

Run from the RL root directory:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import os
import signal
import logging
import torch
# Ensure the RL root is on sys.path so package imports resolve correctly
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import base64
import json
from contextlib import asynccontextmanager
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pythonjsonlogger import jsonlogger

# engine.py applies the PyTorch patch at import time — import before SB3/YOLO
from serving.engine import AdaptiveInferenceSystem
from serving.tracking import SessionTracker

# ──────────────────────────────────────────────────────────────────────────────
# Structured JSON logging (GCP Cloud Logging compatible)
# ──────────────────────────────────────────────────────────────────────────────
_handler = logging.StreamHandler()
_handler.setFormatter(
    jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "time", "levelname": "severity"},
    )
)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)
log = logging.getLogger("adaptive_inference")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (via environment variables with sensible defaults)
# ──────────────────────────────────────────────────────────────────────────────
RL_MODEL_PATH = os.path.join(_RL_ROOT, "models", "PPO_v6", "final_adaptive_model")
YOLO_N_PATH   = os.getenv("YOLO_N_PATH",   "yolov8n.pt")
YOLO_S_PATH   = os.getenv("YOLO_S_PATH",   "yolov8s.pt")
YOLO_L_PATH   = os.getenv("YOLO_L_PATH",   "yolov8l.pt")
DEVICE = os.getenv("INFERENCE_DEVICE")

if DEVICE is None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────────────────────────────────────
FRAMES_TOTAL = Counter(
    "adaptive_inference_frames_total",
    "Total frames processed",
)
ADAPTIVE_LATENCY = Histogram(
    "adaptive_inference_adaptive_latency_seconds",
    "Adaptive path inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
BASELINE_LATENCY = Histogram(
    "adaptive_inference_baseline_latency_seconds",
    "Baseline path inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_SELECTIONS = Counter(
    "adaptive_inference_model_selections_total",
    "Number of times each YOLO model was selected by the RL agent",
    labelnames=["model"],
)
ACTIVE_CONNECTIONS = Gauge(
    "adaptive_inference_active_websocket_connections",
    "Number of currently active WebSocket connections",
)

# ──────────────────────────────────────────────────────────────────────────────
# Engine singleton — loaded once at startup, shared across all connections
# ──────────────────────────────────────────────────────────────────────────────
_engine: AdaptiveInferenceSystem | None = None
_shutdown_requested: bool = False


def _handle_sigterm(signum, frame):
    """Mark shutdown so readiness probe returns 503 during drain."""
    global _shutdown_requested
    log.info("SIGTERM received — marking as not ready for graceful drain")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_sigterm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    log.info("Loading AdaptiveInferenceSystem", extra={
        "rl_model_path": RL_MODEL_PATH,
        "device": DEVICE,
    })
    _engine = AdaptiveInferenceSystem(
        rl_model_path=RL_MODEL_PATH,
        yolo_n_path=YOLO_N_PATH,
        yolo_s_path=YOLO_S_PATH,
        yolo_l_path=YOLO_L_PATH,
        device=DEVICE,
    )
    log.info("Engine ready — serving requests")
    yield
    log.info("Shutting down — engine teardown")


app = FastAPI(title="Adaptive ML Inference API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _decode_frame(b64_payload: str) -> np.ndarray | None:
    """Decode a raw base64 JPEG string into a BGR numpy array."""
    try:
        img_bytes = base64.b64decode(b64_payload)
    except Exception:
        return None
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame  # None if imdecode fails


def _error(msg: str) -> str:
    return json.dumps({"error": msg})


# ──────────────────────────────────────────────────────────────────────────────
# Health / probe routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, Any]:
    """Liveness probe — returns 200 as long as the process is alive."""
    return {"status": "ok", "engine_ready": _engine is not None}


@app.get("/health/startup")
async def health_startup(response: Response) -> Dict[str, Any]:
    """
    Startup probe — returns 200 only after the engine has fully loaded
    (all YOLO models loaded + CUDA warmup complete). Kubernetes will
    keep restarting the pod until this succeeds; set failureThreshold
    high enough to allow up to 10 minutes for GPU warmup.
    """
    if _engine is None:
        response.status_code = 503
        return {"status": "starting", "engine_ready": False}
    return {"status": "ok", "engine_ready": True}


@app.get("/health/ready")
async def health_ready(response: Response) -> Dict[str, Any]:
    """
    Readiness probe — returns 503 while the engine is loading or
    after a SIGTERM (graceful shutdown drain period). Kubernetes will
    stop routing traffic to this pod until it returns 200.
    """
    if _engine is None or _shutdown_requested:
        response.status_code = 503
        return {"status": "not_ready", "engine_ready": _engine is not None, "shutting_down": _shutdown_requested}
    return {"status": "ready", "engine_ready": True}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint scraped by the monitoring stack."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket streaming inference
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    """
    One WebSocket connection = one inference session.

    - Resets RL state at the start of each session so history from a previous
      client does not bleed into a new one.
    - Starts an MLflow run; logs summary metrics on disconnect.
    """
    await websocket.accept()
    ACTIVE_CONNECTIONS.inc()
    log.info("WebSocket session started")

    _engine.reset_state()
    tracker = SessionTracker()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
                b64_frame = payload["frame"]
                baseline_model_name = payload.get("baseline_model", "Small")
            except (json.JSONDecodeError, KeyError):
                b64_frame = raw
                baseline_model_name = "Small"

            frame = _decode_frame(b64_frame)
            if frame is None:
                await websocket.send_text(_error("Could not decode frame"))
                continue

            result = _engine.infer(frame, baseline_model_name=baseline_model_name)
            tracker.record(result)

            # Update Prometheus metrics
            FRAMES_TOTAL.inc()
            adaptive = result["adaptive"]
            baseline = result["baseline"]
            ADAPTIVE_LATENCY.observe(adaptive["latency_ms"] / 1000.0)
            BASELINE_LATENCY.observe(baseline["latency_ms"] / 1000.0)
            MODEL_SELECTIONS.labels(model=adaptive["model_name"]).inc()

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        log.info("WebSocket session ended")
        tracker.finalize()
    finally:
        ACTIVE_CONNECTIONS.dec()
