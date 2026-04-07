"""
ui.py — Streamlit dashboard for the Adaptive ML Inference System.

Layout
------
┌──────────────────────────────────────────────────────────────────┐
│  Path A — RL Adaptive       │  Path B — Baseline (Small)         │
│  [live annotated feed]      │  [live annotated feed]             │
│  Model | Objects | Conf     │  Model | Objects | Conf            │
├──────────────────────────────────────────────────────────────────┤
│  Adp Latency │ Bsl Latency │ Savings │ Adp Conf │ Bsl Conf │ ΔConf │
├──────────────────────────────────────────────────────────────────┤
│  Latency (ms) chart  │  Confidence chart  — last 120 frames       │
└──────────────────────────────────────────────────────────────────┘

Usage (from RL root directory)
-----
    uvicorn serving.app:app --host 0.0.0.0 --port 8000   # backend first
    streamlit run serving/ui.py
    WS_URL=ws://192.168.1.10:8000/ws/stream streamlit run serving/ui.py
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import websocket  # websocket-client (synchronous)

WS_URL: str = os.getenv("WS_URL", "ws://localhost:8000/ws/stream")
MAX_CHART_FRAMES: int = 120
JPEG_QUALITY: int = 75

# BGR colours for bounding boxes / model labels
MODEL_COLORS: Dict[str, tuple] = {
    "Nano":  (0, 255, 0),
    "Small": (0, 255, 255),
    "Large": (0, 0, 255),
}
BASELINE_COLOR: tuple = (255, 150, 0)

# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_boxes(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    color: tuple,
) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(
            out, label, (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )
    return out


def _overlay_hud(
    frame: np.ndarray,
    model_name: str,
    latency_ms: float,
    avg_conf: float,
    is_adaptive: bool,
) -> np.ndarray:
    out = frame.copy()
    color = MODEL_COLORS.get(model_name, (200, 200, 200)) if is_adaptive else BASELINE_COLOR
    prefix = "RL" if is_adaptive else "Baseline"
    cv2.putText(
        out, f"{prefix}: {model_name}", (10, 32),
        cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2,
    )
    cv2.putText(
        out, f"{latency_ms:.1f} ms  |  conf: {avg_conf:.2f}", (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
    )
    return out


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive ML Inference",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Static layout — placeholders updated in-place during the streaming loop
# ──────────────────────────────────────────────────────────────────────────────
st.title("Adaptive ML Inference — Live Dashboard")
st.caption(
    "Path A uses the PPO RL agent to dynamically route frames to the optimal "
    "YOLOv8 variant. Path B always uses YOLOv8-Small as a fixed baseline."
)

run: bool = st.toggle("Start Stream", value=False)

# ── Video feeds ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Path A — RL Adaptive")
    feed_l = st.empty()
    pm_col1, pm_col2, pm_col3 = st.columns(3)
    ph_model_l = pm_col1.empty()
    ph_count_l = pm_col2.empty()
    ph_conf_l  = pm_col3.empty()

with col_r:
    st.subheader("Path B — Baseline (YOLOv8-Small)")
    feed_r = st.empty()
    pb_col1, pb_col2, pb_col3 = st.columns(3)
    ph_model_r = pb_col1.empty()
    ph_count_r = pb_col2.empty()
    ph_conf_r  = pb_col3.empty()

st.divider()

# ── Summary metrics (6 columns) ───────────────────────────────────────────────
st.subheader("Live Metrics")
s1, s2, s3, s4, s5, s6 = st.columns(6)
ph_lat_a   = s1.empty()
ph_lat_b   = s2.empty()
ph_saving  = s3.empty()
ph_aconf   = s4.empty()
ph_bconf   = s5.empty()
ph_cdelta  = s6.empty()

st.divider()

# ── Charts (side by side) ─────────────────────────────────────────────────────
chart_l, chart_r = st.columns(2)
with chart_l:
    st.caption("Latency (ms)")
    ph_lat_chart = st.empty()
with chart_r:
    st.caption("Confidence")
    ph_conf_chart = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Streaming loop — runs only while the toggle is ON
# ──────────────────────────────────────────────────────────────────────────────
if run:
    try:
        ws_conn = websocket.create_connection(WS_URL, timeout=5)
    except Exception as exc:
        st.error(f"Cannot connect to inference server at **{WS_URL}** — {exc}")
        st.stop()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        ws_conn.close()
        st.error("Webcam not found or already in use.")
        st.stop()

    adaptive_lats:  List[float] = []
    baseline_lats:  List[float] = []
    adaptive_confs: List[float] = []
    baseline_confs: List[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam read failed — stopping stream.")
                break

            # Encode frame → base64 JPEG → send to FastAPI WS
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            b64 = base64.b64encode(jpeg.tobytes()).decode()
            ws_conn.send(b64)

            raw = ws_conn.recv()
            result = json.loads(raw)

            if "error" in result:
                continue

            adp = result["adaptive"]
            bsl = result["baseline"]

            # ── Annotate frames ───────────────────────────────────────────────
            adp_color = MODEL_COLORS.get(adp["model_name"], (200, 200, 200))
            af = _draw_boxes(frame, adp["detections"], adp_color)
            af = _overlay_hud(af, adp["model_name"], adp["latency_ms"],
                              adp["avg_confidence"], is_adaptive=True)

            bf = _draw_boxes(frame, bsl["detections"], BASELINE_COLOR)
            bf = _overlay_hud(bf, bsl["model_name"], bsl["latency_ms"],
                              bsl["avg_confidence"], is_adaptive=False)

            # ── Update video feeds ────────────────────────────────────────────
            feed_l.image(_to_rgb(af), channels="RGB", use_container_width=True)
            feed_r.image(_to_rgb(bf), channels="RGB", use_container_width=True)

            # ── Per-path metrics ──────────────────────────────────────────────
            ph_model_l.metric("Model",   adp["model_name"])
            ph_count_l.metric("Objects", adp["object_count"])
            ph_conf_l.metric( "Confidence", f"{adp['avg_confidence']:.2f}")

            ph_model_r.metric("Model",   bsl["model_name"])
            ph_count_r.metric("Objects", bsl["object_count"])
            ph_conf_r.metric( "Confidence", f"{bsl['avg_confidence']:.2f}")

            # ── Summary metrics ───────────────────────────────────────────────
            lat_savings  = bsl["latency_ms"] - adp["latency_ms"]
            conf_delta   = adp["avg_confidence"] - bsl["avg_confidence"]

            ph_lat_a.metric("Adaptive Latency",  f"{adp['latency_ms']:.1f} ms")
            ph_lat_b.metric("Baseline Latency",  f"{bsl['latency_ms']:.1f} ms")
            ph_saving.metric(
                "Latency Savings",
                f"{lat_savings:.1f} ms",
                delta=f"{lat_savings:.1f} ms",
                delta_color="normal",
            )
            ph_aconf.metric("Adaptive Conf",  f"{adp['avg_confidence']:.2f}")
            ph_bconf.metric("Baseline Conf",  f"{bsl['avg_confidence']:.2f}")
            ph_cdelta.metric(
                "Conf Gain",
                f"{conf_delta:+.2f}",
                delta=f"{conf_delta:+.2f}",
                delta_color="normal",
            )

            # ── Rolling history ───────────────────────────────────────────────
            adaptive_lats.append(adp["latency_ms"])
            baseline_lats.append(bsl["latency_ms"])
            adaptive_confs.append(adp["avg_confidence"])
            baseline_confs.append(bsl["avg_confidence"])

            if len(adaptive_lats) > MAX_CHART_FRAMES:
                adaptive_lats.pop(0)
                baseline_lats.pop(0)
                adaptive_confs.pop(0)
                baseline_confs.pop(0)

            # ── Latency chart ─────────────────────────────────────────────────
            ph_lat_chart.line_chart(
                pd.DataFrame({"Adaptive": adaptive_lats, "Baseline": baseline_lats})
            )

            # ── Confidence chart ──────────────────────────────────────────────
            ph_conf_chart.line_chart(
                pd.DataFrame({"Adaptive": adaptive_confs, "Baseline": baseline_confs})
            )

    finally:
        cap.release()
        ws_conn.close()
