"""
ui.py — Streamlit dashboard for the Adaptive ML Inference System.

Two camera modes (controlled by CAMERA_MODE env var):
  - "local"   (default): OpenCV continuous live stream from webcam index 0
  - "browser": streamlit-webrtc — true live stream via browser camera (WebRTC),
               works on deployed GKE without a physical webcam on the server

Usage (from RL root directory)
-----
    # Local live stream (default)
    streamlit run serving/ui.py

    # Deployed / browser camera (requires: pip install streamlit-webrtc aiortc av)
    CAMERA_MODE=browser streamlit run serving/ui.py
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

WS_URL: str       = os.getenv("WS_URL", "ws://localhost:8000/ws/stream")
CAMERA_MODE: str  = os.getenv("CAMERA_MODE", "local")   # "local" | "browser"
MAX_CHART_FRAMES  = 120
JPEG_QUALITY      = 75

MODEL_COLORS: Dict[str, tuple] = {
    "Nano":  (0, 255, 0),
    "Small": (0, 255, 255),
    "Large": (0, 0, 255),
}
BASELINE_COLOR: tuple = (255, 150, 0)

# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_boxes(frame, detections, color):
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def _overlay_hud(frame, model_name, latency_ms, avg_conf, is_adaptive):
    out = frame.copy()
    color = MODEL_COLORS.get(model_name, (200, 200, 200)) if is_adaptive else BASELINE_COLOR
    prefix = "RL" if is_adaptive else "Baseline"
    cv2.putText(out, f"{prefix}: {model_name}", (10, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)
    cv2.putText(out, f"{latency_ms:.1f} ms  |  conf: {avg_conf:.2f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return out


def _to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
for _k, _v in [
    ("adaptive_lats", []), ("baseline_lats", []),
    ("adaptive_confs", []), ("baseline_confs", []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Adaptive ML Inference",
                   layout="wide", initial_sidebar_state="collapsed")
st.title("Adaptive ML Inference — Live Dashboard")

run: bool = st.toggle("Start Stream", value=False)
baseline_choice: str = st.selectbox(
    "Baseline model (Path B)", ["Nano", "Small", "Large"], index=1, disabled=run)
st.caption(
    "Path A uses the PPO RL agent to dynamically route frames to the optimal "
    f"YOLOv8 variant. Path B uses YOLOv8-{baseline_choice} as a fixed baseline.")

# ── Layout placeholders ───────────────────────────────────────────────────────
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Path A — RL Adaptive")
    feed_l = st.empty()
    c1, c2, c3 = st.columns(3)
    ph_model_l, ph_count_l, ph_conf_l = c1.empty(), c2.empty(), c3.empty()
with col_r:
    st.subheader(f"Path B — Baseline (YOLOv8-{baseline_choice})")
    feed_r = st.empty()
    c1, c2, c3 = st.columns(3)
    ph_model_r, ph_count_r, ph_conf_r = c1.empty(), c2.empty(), c3.empty()

st.divider()
st.subheader("Live Metrics")
s1, s2, s3, s4, s5, s6 = st.columns(6)
ph_lat_a  = s1.empty(); ph_lat_b  = s2.empty(); ph_saving = s3.empty()
ph_aconf  = s4.empty(); ph_bconf  = s5.empty(); ph_cdelta = s6.empty()

st.divider()
cl, cr = st.columns(2)
with cl:
    st.caption("Latency (ms)")
    ph_lat_chart = st.empty()
with cr:
    st.caption("Confidence")
    ph_conf_chart = st.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Shared: update metrics and charts from a result dict
# update_left=False skips feed_l (used in browser mode where WebRTC shows it)
# ──────────────────────────────────────────────────────────────────────────────
def _update_ui(frame, result, lats_a, lats_b, confs_a, confs_b, update_left=True):
    adp = result["adaptive"]
    bsl = result["baseline"]

    adp_color = MODEL_COLORS.get(adp["model_name"], (200, 200, 200))
    af = _overlay_hud(_draw_boxes(frame, adp["detections"], adp_color),
                      adp["model_name"], adp["latency_ms"], adp["avg_confidence"], True)
    bf = _overlay_hud(_draw_boxes(frame, bsl["detections"], BASELINE_COLOR),
                      bsl["model_name"], bsl["latency_ms"], bsl["avg_confidence"], False)

    if update_left:
        feed_l.image(_to_rgb(af), channels="RGB", use_container_width=True)
    feed_r.image(_to_rgb(bf), channels="RGB", use_container_width=True)

    ph_model_l.metric("Model", adp["model_name"])
    ph_count_l.metric("Objects", adp["object_count"])
    ph_conf_l.metric("Confidence", f"{adp['avg_confidence']:.2f}")
    ph_model_r.metric("Model", bsl["model_name"])
    ph_count_r.metric("Objects", bsl["object_count"])
    ph_conf_r.metric("Confidence", f"{bsl['avg_confidence']:.2f}")

    lat_savings = bsl["latency_ms"] - adp["latency_ms"]
    conf_delta  = adp["avg_confidence"] - bsl["avg_confidence"]
    ph_lat_a.metric("Adaptive Latency",  f"{adp['latency_ms']:.1f} ms")
    ph_lat_b.metric("Baseline Latency",  f"{bsl['latency_ms']:.1f} ms")
    ph_saving.metric("Latency Savings",  f"{lat_savings:.1f} ms",
                     delta=f"{lat_savings:.1f} ms", delta_color="normal")
    ph_aconf.metric("Adaptive Conf",     f"{adp['avg_confidence']:.2f}")
    ph_bconf.metric("Baseline Conf",     f"{bsl['avg_confidence']:.2f}")
    ph_cdelta.metric("Conf Gain",        f"{conf_delta:+.2f}",
                     delta=f"{conf_delta:+.2f}", delta_color="normal")

    lats_a.append(adp["latency_ms"]); lats_b.append(bsl["latency_ms"])
    confs_a.append(adp["avg_confidence"]); confs_b.append(bsl["avg_confidence"])
    for lst in [lats_a, lats_b, confs_a, confs_b]:
        if len(lst) > MAX_CHART_FRAMES:
            lst.pop(0)

    _cs = alt.Scale(domain=["Baseline","Adaptive"], range=["#1f77b4","#FF6B35"])
    _ds = alt.Scale(domain=["Baseline","Adaptive"], range=[[6,4],[0,0]])
    frames = list(range(len(lats_a)))

    lat_df = pd.DataFrame({"frame": frames, "Adaptive": lats_a, "Baseline": lats_b}
                          ).melt("frame", var_name="Series", value_name="Latency (ms)")
    ph_lat_chart.altair_chart(
        alt.Chart(lat_df).mark_line().encode(
            x=alt.X("frame:Q", axis=alt.Axis(title=None, labels=False)),
            y="Latency (ms):Q",
            color=alt.Color("Series:N", scale=_cs),
            strokeDash=alt.StrokeDash("Series:N", scale=_ds)),
        use_container_width=True)

    conf_df = pd.DataFrame({"frame": frames, "Adaptive": confs_a, "Baseline": confs_b}
                           ).melt("frame", var_name="Series", value_name="Confidence")
    ph_conf_chart.altair_chart(
        alt.Chart(conf_df).mark_line().encode(
            x=alt.X("frame:Q", axis=alt.Axis(title=None, labels=False)),
            y="Confidence:Q",
            color=alt.Color("Series:N", scale=_cs),
            strokeDash=alt.StrokeDash("Series:N", scale=_ds)),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE A — Local: OpenCV continuous live stream
# ══════════════════════════════════════════════════════════════════════════════
if run and CAMERA_MODE == "local":
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

    lats_a: List[float] = []
    lats_b: List[float] = []
    confs_a: List[float] = []
    confs_b: List[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam read failed — stopping stream.")
                break

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            b64 = base64.b64encode(jpeg.tobytes()).decode()
            ws_conn.send(json.dumps({"frame": b64, "baseline_model": baseline_choice}))

            raw    = ws_conn.recv()
            result = json.loads(raw)
            if "error" in result:
                continue

            _update_ui(frame, result, lats_a, lats_b, confs_a, confs_b)
    finally:
        cap.release()
        ws_conn.close()

# ══════════════════════════════════════════════════════════════════════════════
# MODE B — Browser: streamlit-webrtc (true live stream for GKE / deployed)
#
# The browser camera is accessed via WebRTC — no server-side webcam needed.
# Each frame is sent to the FastAPI backend via WebSocket; the annotated
# adaptive frame is returned through the WebRTC stream. The baseline feed
# and all metrics are updated via a 10 fps polling loop (st.rerun).
#
# Requires: pip install streamlit-webrtc aiortc av
# HTTPS note: Chrome blocks getUserMedia on HTTP. Use Firefox, or access via
#   kubectl port-forward svc/ui-service 8501:80 -n adaptive-inference
#   then open http://localhost:8501
# ══════════════════════════════════════════════════════════════════════════════
elif run and CAMERA_MODE == "browser":
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av as _av
        import threading as _threading
    except ImportError:
        st.error(
            "**streamlit-webrtc** is not installed. "
            "Run: `pip install streamlit-webrtc aiortc av`"
        )
        st.stop()

    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })

    class _InferenceProcessor(VideoProcessorBase):
        """Sends each WebRTC frame to the FastAPI backend; returns annotated frame."""

        def __init__(self) -> None:
            self._data_lock = _threading.Lock()  # guards _latest
            self._ws_lock   = _threading.Lock()  # guards WebSocket from concurrent recv()
            self._ws: Any   = None
            self._latest: Any = None             # (raw_bgr_frame, result_dict) | None
            self._baseline: str = baseline_choice
            self._connect()

        def _connect(self) -> None:
            try:
                self._ws = websocket.create_connection(WS_URL, timeout=5)
            except Exception:
                self._ws = None

        def recv(self, frame: "_av.VideoFrame") -> "_av.VideoFrame":
            img = frame.to_ndarray(format="bgr24")
            with self._ws_lock:
                if self._ws is None:
                    self._connect()
                if self._ws:
                    try:
                        _, jpeg = cv2.imencode(
                            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                        )
                        b64 = base64.b64encode(jpeg.tobytes()).decode()
                        self._ws.send(
                            json.dumps({"frame": b64, "baseline_model": self._baseline})
                        )
                        result = json.loads(self._ws.recv())
                        if "error" not in result:
                            with self._data_lock:
                                self._latest = (img.copy(), result)
                            adp   = result["adaptive"]
                            color = MODEL_COLORS.get(adp["model_name"], (200, 200, 200))
                            out   = _overlay_hud(
                                _draw_boxes(img, adp["detections"], color),
                                adp["model_name"], adp["latency_ms"],
                                adp["avg_confidence"], True,
                            )
                            return _av.VideoFrame.from_ndarray(_to_rgb(out), format="rgb24")
                    except Exception:
                        self._ws = None
            return _av.VideoFrame.from_ndarray(img, format="bgr24")

        def get_latest(self) -> Any:
            with self._data_lock:
                return self._latest

        def __del__(self) -> None:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass

    # ── Launch WebRTC streamer in the left-column feed placeholder ─────────────
    # feed_l.container() places the WebRTC player at the correct position
    # (above the per-feed metric row) without extra empty space.
    with feed_l.container():
        ctx = webrtc_streamer(
            key="inference",
            video_processor_factory=_InferenceProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )

    # Keep the baseline model selection in sync with the live processor
    if ctx.video_processor:
        ctx.video_processor._baseline = baseline_choice

    # ── Poll processor for results; update baseline feed + all metrics ─────────
    if ctx.state.playing and ctx.video_processor:
        latest = ctx.video_processor.get_latest()
        if latest is not None:
            frame, result = latest
            _update_ui(
                frame, result,
                st.session_state.adaptive_lats,
                st.session_state.baseline_lats,
                st.session_state.adaptive_confs,
                st.session_state.baseline_confs,
                update_left=False,   # WebRTC element already shows the adaptive feed
            )
        time.sleep(0.1)   # ~10 fps metrics refresh
        st.rerun()

    elif not ctx.state.playing:
        st.session_state.adaptive_lats  = []
        st.session_state.baseline_lats  = []
        st.session_state.adaptive_confs = []
        st.session_state.baseline_confs = []

else:
    # Stream stopped — reset metrics history
    st.session_state.adaptive_lats  = []
    st.session_state.baseline_lats  = []
    st.session_state.adaptive_confs = []
    st.session_state.baseline_confs = []
