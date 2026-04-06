"""
app.py — FastAPI backend for the Adaptive ML Inference System.

Endpoints
---------
GET  /health          — liveness probe
WS   /ws/stream       — streaming inference over WebSocket

WebSocket protocol
------------------
Client → Server : raw base64-encoded JPEG frame (no data-URL prefix)
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
RL_MODEL_PATH      path to PPO .zip  (default: models/PPO/final_adaptive_model.zip)
YOLO_N_PATH        yolov8n.pt        (default: yolov8n.pt)
YOLO_S_PATH        yolov8s.pt        (default: yolov8s.pt)
YOLO_L_PATH        yolov8l.pt        (default: yolov8l.pt)
INFERENCE_DEVICE   cuda | cpu        (default: cuda)
"""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# engine.py applies the PyTorch patch at import time — import before SB3/YOLO
from engine import AdaptiveInferenceSystem
from tracking import SessionTracker

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (via environment variables with sensible defaults)
# ──────────────────────────────────────────────────────────────────────────────
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "models/PPO/final_adaptive_model.zip")
YOLO_N_PATH   = os.getenv("YOLO_N_PATH",   "yolov8n.pt")
YOLO_S_PATH   = os.getenv("YOLO_S_PATH",   "yolov8s.pt")
YOLO_L_PATH   = os.getenv("YOLO_L_PATH",   "yolov8l.pt")
DEVICE        = os.getenv("INFERENCE_DEVICE", "cuda")

# ──────────────────────────────────────────────────────────────────────────────
# Engine singleton — loaded once at startup, shared across all connections
# ──────────────────────────────────────────────────────────────────────────────
_engine: AdaptiveInferenceSystem | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = AdaptiveInferenceSystem(
        rl_model_path=RL_MODEL_PATH,
        yolo_n_path=YOLO_N_PATH,
        yolo_s_path=YOLO_S_PATH,
        yolo_l_path=YOLO_L_PATH,
        device=DEVICE,
    )
    yield
    # No explicit teardown needed; OS reclaims GPU memory on process exit.


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
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "engine_ready": _engine is not None}


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    """
    One WebSocket connection = one inference session.

    - Resets RL state at the start of each session so history from a previous
      client does not bleed into a new one.
    - Starts an MLflow run; logs summary metrics on disconnect.
    """
    await websocket.accept()

    _engine.reset_state()
    tracker = SessionTracker()

    try:
        while True:
            raw = await websocket.receive_text()

            frame = _decode_frame(raw)
            if frame is None:
                await websocket.send_text(_error("Could not decode frame"))
                continue

            result = _engine.infer(frame)
            tracker.record(result)

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        tracker.finalize()
