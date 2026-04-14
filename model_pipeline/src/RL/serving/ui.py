"""
ui.py — Streamlit dashboard for the Adaptive ML Inference System.

Uses st.camera_input to capture frames from the browser's webcam — works on
any deployed URL without requiring HTTPS or a physical camera on the server.

Layout
------
┌──────────────────────────────────────────────────────────────────┐
│  [Camera Feed]                                                   │
├──────────────────────────────────────────────────────────────────┤
│  Path A — RL Adaptive       │  Path B — Baseline (Small)         │
│  [annotated frame]          │  [annotated frame]                 │
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
import time
from typing import Any, Dict, List

import altair as alt
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
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "ws_conn" not in st.session_state:
    st.session_state.ws_conn = None
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "adaptive_lats" not in st.session_state:
    st.session_state.adaptive_lats: List[float] = []
if "baseline_lats" not in st.session_state:
    st.session_state.baseline_lats: List[float] = []
if "adaptive_confs" not in st.session_state:
    st.session_state.adaptive_confs: List[float] = []
if "baseline_confs" not in st.session_state:
    st.session_state.baseline_confs: List[float] = []

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive ML Inference",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Header controls
# ──────────────────────────────────────────────────────────────────────────────
st.title("Adaptive ML Inference — Live Dashboard")

run: bool = st.toggle("Start Stream", value=False)
baseline_choice: str = st.selectbox(
    "Baseline model (Path B)",
    ["Nano", "Small", "Large"],
    index=1,
    disabled=run,
)
st.caption(
    "Path A uses the PPO RL agent to dynamically route frames to the optimal "
    f"YOLOv8 variant. Path B uses YOLOv8-{baseline_choice} as a fixed baseline."
)

# ──────────────────────────────────────────────────────────────────────────────
# Camera input (browser webcam — no server camera needed)
# ──────────────────────────────────────────────────────────────────────────────
if run:
    st.info("Allow camera access when prompted by your browser.")
    # Changing the key on each frame forces a new capture automatically
    camera_frame = st.camera_input(
        "Live Camera",
        key=f"cam_{st.session_state.frame_count}",
        label_visibility="collapsed",
    )
else:
    camera_frame = None
    # Close WebSocket when stream is stopped
    if st.session_state.ws_conn is not None:
        try:
            st.session_state.ws_conn.close()
        except Exception:
            pass
        st.session_state.ws_conn = None
        st.session_state.frame_count = 0
        st.session_state.adaptive_lats = []
        st.session_state.baseline_lats = []
        st.session_state.adaptive_confs = []
        st.session_state.baseline_confs = []

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
    st.subheader(f"Path B — Baseline (YOLOv8-{baseline_choice})")
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
# Frame processing — runs when a camera frame is captured
# ──────────────────────────────────────────────────────────────────────────────
if run and camera_frame is not None:

    # Establish or reuse WebSocket connection
    if st.session_state.ws_conn is None:
        try:
            st.session_state.ws_conn = websocket.create_connection(WS_URL, timeout=5)
        except Exception as exc:
            st.error(f"Cannot connect to inference server at **{WS_URL}** — {exc}")
            st.stop()

    # Decode JPEG from browser camera → OpenCV BGR
    img_bytes = camera_frame.getvalue()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        st.warning("Could not decode camera frame.")
        st.stop()

    # Re-encode as JPEG at target quality and send to backend
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    try:
        st.session_state.ws_conn.send(
            json.dumps({"frame": b64, "baseline_model": baseline_choice})
        )
        raw = st.session_state.ws_conn.recv()
        result = json.loads(raw)
    except Exception as exc:
        st.session_state.ws_conn = None
        st.error(f"Connection lost — {exc}")
        st.stop()

    if "error" in result:
        st.session_state.frame_count += 1
        st.rerun()

    adp = result["adaptive"]
    bsl = result["baseline"]

    # ── Annotate frames ───────────────────────────────────────────────────────
    adp_color = MODEL_COLORS.get(adp["model_name"], (200, 200, 200))
    af = _draw_boxes(frame, adp["detections"], adp_color)
    af = _overlay_hud(af, adp["model_name"], adp["latency_ms"],
                      adp["avg_confidence"], is_adaptive=True)

    bf = _draw_boxes(frame, bsl["detections"], BASELINE_COLOR)
    bf = _overlay_hud(bf, bsl["model_name"], bsl["latency_ms"],
                      bsl["avg_confidence"], is_adaptive=False)

    # ── Update video feeds ────────────────────────────────────────────────────
    feed_l.image(_to_rgb(af), channels="RGB", use_container_width=True)
    feed_r.image(_to_rgb(bf), channels="RGB", use_container_width=True)

    # ── Per-path metrics ──────────────────────────────────────────────────────
    ph_model_l.metric("Model",      adp["model_name"])
    ph_count_l.metric("Objects",    adp["object_count"])
    ph_conf_l.metric( "Confidence", f"{adp['avg_confidence']:.2f}")

    ph_model_r.metric("Model",      bsl["model_name"])
    ph_count_r.metric("Objects",    bsl["object_count"])
    ph_conf_r.metric( "Confidence", f"{bsl['avg_confidence']:.2f}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    lat_savings = bsl["latency_ms"] - adp["latency_ms"]
    conf_delta  = adp["avg_confidence"] - bsl["avg_confidence"]

    ph_lat_a.metric("Adaptive Latency", f"{adp['latency_ms']:.1f} ms")
    ph_lat_b.metric("Baseline Latency", f"{bsl['latency_ms']:.1f} ms")
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

    # ── Rolling history ───────────────────────────────────────────────────────
    st.session_state.adaptive_lats.append(adp["latency_ms"])
    st.session_state.baseline_lats.append(bsl["latency_ms"])
    st.session_state.adaptive_confs.append(adp["avg_confidence"])
    st.session_state.baseline_confs.append(bsl["avg_confidence"])

    if len(st.session_state.adaptive_lats) > MAX_CHART_FRAMES:
        st.session_state.adaptive_lats.pop(0)
        st.session_state.baseline_lats.pop(0)
        st.session_state.adaptive_confs.pop(0)
        st.session_state.baseline_confs.pop(0)

    # ── Charts ────────────────────────────────────────────────────────────────
    _color_scale = alt.Scale(
        domain=["Baseline", "Adaptive"],
        range=["#1f77b4", "#FF6B35"],
    )
    _dash_scale = alt.Scale(
        domain=["Baseline", "Adaptive"],
        range=[[6, 4], [0, 0]],
    )

    frames = list(range(len(st.session_state.adaptive_lats)))

    lat_df = pd.DataFrame({
        "frame": frames,
        "Adaptive": st.session_state.adaptive_lats,
        "Baseline": st.session_state.baseline_lats,
    }).melt("frame", var_name="Series", value_name="Latency (ms)")

    ph_lat_chart.altair_chart(
        alt.Chart(lat_df).mark_line().encode(
            x=alt.X("frame:Q", axis=alt.Axis(title=None, labels=False)),
            y=alt.Y("Latency (ms):Q"),
            color=alt.Color("Series:N", scale=_color_scale),
            strokeDash=alt.StrokeDash("Series:N", scale=_dash_scale),
        ),
        use_container_width=True,
    )

    conf_df = pd.DataFrame({
        "frame": frames,
        "Adaptive": st.session_state.adaptive_confs,
        "Baseline": st.session_state.baseline_confs,
    }).melt("frame", var_name="Series", value_name="Confidence")

    ph_conf_chart.altair_chart(
        alt.Chart(conf_df).mark_line().encode(
            x=alt.X("frame:Q", axis=alt.Axis(title=None, labels=False)),
            y=alt.Y("Confidence:Q"),
            color=alt.Color("Series:N", scale=_color_scale),
            strokeDash=alt.StrokeDash("Series:N", scale=_dash_scale),
        ),
        use_container_width=True,
    )

    # Advance frame counter and rerun to capture next frame automatically
    st.session_state.frame_count += 1
    time.sleep(0.05)
    st.rerun()
