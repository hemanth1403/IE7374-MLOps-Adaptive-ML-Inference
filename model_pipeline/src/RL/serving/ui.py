"""
ui.py — Streamlit dashboard for the Adaptive ML Inference System.

Two camera modes (controlled by CAMERA_MODE env var):
  - "local"   (default): OpenCV continuous live stream from webcam index 0
  - "browser": Pure JavaScript getUserMedia + WebSocket — no WebRTC/STUN/TURN needed.
               The browser captures the camera, sends JPEG frames directly to the
               FastAPI backend via ws://<host>/ws/stream (routed by NGINX ingress).

Usage (from RL root directory)
-----
    # Local live stream (default)
    streamlit run serving/ui.py

    # Deployed / browser camera
    CAMERA_MODE=browser streamlit run serving/ui.py
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from typing import Any, Dict, List

import altair as alt
import cv2
import numpy as np
import inspect
import pandas as pd
import streamlit as st
import websocket  # websocket-client (synchronous)

# st.image gained use_container_width in 1.35+; older versions only have use_column_width
_IMG_KW = (
    {"use_container_width": True}
    if "use_container_width" in inspect.signature(st.image).parameters
    else {"use_column_width": True}
)

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

input_mode: str = st.radio(
    "Input Source", ["Live Camera", "Upload Video"],
    horizontal=True)

uploaded_video = None
frame_skip = 1
if input_mode == "Upload Video":
    uploaded_video = st.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        st.caption(f"File: **{uploaded_video.name}**")
    frame_skip = st.select_slider(
        "Process every N frames",
        options=[1, 2, 4, 8, 16],
        value=4,
        help="Higher = faster processing with fewer data points. 4 is a good balance."
    )

_toggle_label = "Process Video" if input_mode == "Upload Video" else "Start Stream"
_toggle_disabled = (input_mode == "Upload Video" and uploaded_video is None)
run: bool = st.toggle(_toggle_label, value=False, disabled=_toggle_disabled)

baseline_choice: str = st.selectbox(
    "Baseline model (Path B)", ["Nano", "Small", "Large"], index=1, disabled=run)
st.caption(
    "Path A uses the PPO RL agent to dynamically route frames to the optimal "
    f"YOLOv8 variant. Path B uses YOLOv8-{baseline_choice} as a fixed baseline.")

# ── Placeholder for browser camera component (full-width, sits above columns) ─
browser_ph = st.empty()

# ── Layout placeholders (used by local mode) ─────────────────────────────────
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
# Shared: update metrics and charts (local mode)
# update_left=False skips feed_l (browser mode handles adaptive display in JS)
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
        feed_l.image(_to_rgb(af), channels="RGB", **_IMG_KW)
    feed_r.image(_to_rgb(bf), channels="RGB", **_IMG_KW)

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
if run and input_mode == "Live Camera" and CAMERA_MODE == "local":
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
# MODE B — Browser: pure JavaScript camera + WebSocket (no WebRTC/STUN/TURN)
#
# The browser captures camera via getUserMedia(), draws frames to a canvas,
# and sends base64 JPEG frames directly to the FastAPI backend WebSocket at
# ws://<host>/ws/stream (routed via NGINX ingress — backend never exposed publicly).
#
# No peer-to-peer connection needed, so no NAT issues. Works on any browser
# that supports getUserMedia (all modern browsers over HTTPS; Firefox also on HTTP).
# ══════════════════════════════════════════════════════════════════════════════
elif run and input_mode == "Live Camera" and CAMERA_MODE == "browser":

    _component_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: monospace; color: #eee; padding: 4px; }}
  #status {{ font-size: 12px; color: #aaa; margin-bottom: 8px; }}
  .feeds {{ display: flex; gap: 8px; width: 100%; }}
  .feed-wrap {{ flex: 1; }}
  .feed-label {{ font-size: 12px; color: #aaa; margin-bottom: 4px; }}
  canvas {{ width: 100%; aspect-ratio: 4/3; display: block;
            border-radius: 4px; background: #0e1117; }}
  .metrics {{ display: flex; margin-top: 10px; border-top: 1px solid #333; }}
  .metric {{ flex: 1; padding: 8px 4px; text-align: center;
             border-right: 1px solid #333; }}
  .metric:last-child {{ border-right: none; }}
  .mval {{ font-size: 1.05em; font-weight: bold; color: #FF6B35; }}
  .mlbl {{ font-size: 10px; color: #888; margin-top: 2px; }}
</style>
</head>
<body>
<div id="status">Starting camera...</div>
<div class="feeds">
  <div class="feed-wrap">
    <div class="feed-label">Path A — RL Adaptive</div>
    <canvas id="cA"></canvas>
  </div>
  <div class="feed-wrap">
    <div class="feed-label">Path B — Baseline (YOLOv8-{baseline_choice})</div>
    <canvas id="cB"></canvas>
  </div>
</div>
<div class="metrics">
  <div class="metric"><div class="mval" id="mModel">—</div><div class="mlbl">RL Model</div></div>
  <div class="metric"><div class="mval" id="mLatA">—</div><div class="mlbl">Adaptive Latency</div></div>
  <div class="metric"><div class="mval" id="mLatB">—</div><div class="mlbl">Baseline Latency</div></div>
  <div class="metric"><div class="mval" id="mSave">—</div><div class="mlbl">Latency Saved</div></div>
  <div class="metric"><div class="mval" id="mConfA">—</div><div class="mlbl">Adaptive Conf</div></div>
  <div class="metric"><div class="mval" id="mConfB">—</div><div class="mlbl">Baseline Conf</div></div>
</div>
<video id="vid" autoplay playsinline muted style="display:none"></video>

<script>
const BASELINE = '{baseline_choice}';
const FPS      = 10;
const W = 640, H = 480;
const MODEL_COLORS = {{Nano:'#00ff55', Small:'#00ffff', Large:'#4466ff'}};
const BL_COLOR = '#ff9600';

const vid = document.getElementById('vid');
const cA  = document.getElementById('cA');
const cB  = document.getElementById('cB');
const xA  = cA.getContext('2d');
const xB  = cB.getContext('2d');
const status = document.getElementById('status');

cA.width = W; cA.height = H;
cB.width = W; cB.height = H;

let ws = null;
let lastResult = null;

async function init() {{
  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{video:true, audio:false}});
    vid.srcObject = stream;
    await new Promise(r => vid.onloadedmetadata = r);
    status.textContent = 'Camera ready — connecting to backend...';
  }} catch(e) {{
    status.textContent = 'Camera blocked: ' + e.message + ' (use HTTPS or Firefox)';
    return;
  }}
  connect();
  requestAnimationFrame(renderLoop);
  setInterval(sendFrame, Math.round(1000 / FPS));
}}

function connect() {{
  const host  = (window.parent || window).location.host;
  const proto = (window.parent || window).location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + host + '/ws/stream');
  ws.onopen  = () => {{ status.textContent = 'Live \u2014 RL adaptive routing active \u2713'; }};
  ws.onclose = () => {{ status.textContent = 'Reconnecting...'; setTimeout(connect, 2000); }};
  ws.onerror = () => {{ status.textContent = 'Backend unreachable \u2014 retrying...'; }};
  ws.onmessage = evt => {{
    try {{
      const d = JSON.parse(evt.data);
      if (!d.error) {{ lastResult = d; updateMetrics(d); }}
    }} catch(_) {{}}
  }};
}}

function sendFrame() {{
  if (!ws || ws.readyState !== 1 || vid.readyState < 2) return;
  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  tmp.getContext('2d').drawImage(vid, 0, 0, W, H);
  tmp.toBlob(blob => {{
    const reader = new FileReader();
    reader.onloadend = () => {{
      if (ws.readyState !== 1) return;
      const b64 = reader.result.split(',')[1];
      ws.send(JSON.stringify({{frame: b64, baseline_model: BASELINE}}));
    }};
    reader.readAsDataURL(blob);
  }}, 'image/jpeg', 0.75);
}}

function renderLoop() {{
  if (vid.readyState >= 2) {{
    xA.drawImage(vid, 0, 0, W, H);
    xB.drawImage(vid, 0, 0, W, H);
    if (lastResult) {{
      const adp = lastResult.adaptive;
      const bsl = lastResult.baseline;
      overlayResult(xA, adp.detections, MODEL_COLORS[adp.model_name] || '#ccc',
                    'RL: ' + adp.model_name, adp.latency_ms, adp.avg_confidence);
      overlayResult(xB, bsl.detections, BL_COLOR,
                    'Baseline: ' + bsl.model_name, bsl.latency_ms, bsl.avg_confidence);
    }}
  }}
  requestAnimationFrame(renderLoop);
}}

function overlayResult(ctx, dets, color, label, lat, conf) {{
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, W, 52);
  ctx.fillStyle = color;
  ctx.font = 'bold 15px monospace';
  ctx.fillText(label, 10, 24);
  ctx.fillStyle = '#cccccc';
  ctx.font = '11px monospace';
  ctx.fillText(lat.toFixed(1) + ' ms  |  conf: ' + conf.toFixed(2), 10, 44);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.fillStyle = color;
  ctx.font = '11px monospace';
  (dets || []).forEach(d => {{
    const [x1,y1,x2,y2] = d.bbox;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    ctx.fillText(d.class_name + ' ' + d.confidence.toFixed(2), x1, Math.max(y1-3,12));
  }});
}}

function updateMetrics(d) {{
  const save = d.baseline.latency_ms - d.adaptive.latency_ms;
  document.getElementById('mModel').textContent  = d.adaptive.model_name;
  document.getElementById('mLatA').textContent   = d.adaptive.latency_ms.toFixed(1) + ' ms';
  document.getElementById('mLatB').textContent   = d.baseline.latency_ms.toFixed(1)  + ' ms';
  document.getElementById('mSave').textContent   = save.toFixed(1) + ' ms';
  document.getElementById('mConfA').textContent  = d.adaptive.avg_confidence.toFixed(2);
  document.getElementById('mConfB').textContent  = d.baseline.avg_confidence.toFixed(2);
}}

init();
</script>
</body>
</html>"""

    with browser_ph.container():
        st.components.v1.html(_component_html, height=580, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# MODE C — Upload Video: process frames from an uploaded video file
# ══════════════════════════════════════════════════════════════════════════════
elif run and input_mode == "Upload Video":
    # Save uploaded bytes to a temp file so OpenCV can open it
    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_video.read())
    tmp.flush()
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st.caption(f"Video info: **{total_frames} frames** @ **{fps:.1f} fps**")

    try:
        ws_conn = websocket.create_connection(WS_URL, timeout=5)
    except Exception as exc:
        cap.release()
        os.unlink(tmp.name)
        st.error(f"Cannot connect to inference server at **{WS_URL}** — {exc}")
        st.stop()

    lats_a: List[float] = []
    lats_b: List[float] = []
    confs_a: List[float] = []
    confs_b: List[float] = []
    progress_bar = st.progress(0, text="Processing frames…")
    frame_idx = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                pct = min(frame_idx / max(total_frames, 1), 1.0)
                progress_bar.progress(pct, text=f"Scanning… {frame_idx} / {total_frames}")
                continue

            _, jpeg = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            b64 = base64.b64encode(jpeg.tobytes()).decode()
            ws_conn.send(json.dumps({"frame": b64, "baseline_model": baseline_choice}))

            raw    = ws_conn.recv()
            result = json.loads(raw)
            processed += 1

            if "error" in result:
                continue

            _update_ui(frame, result, lats_a, lats_b, confs_a, confs_b)

            pct = min(frame_idx / max(total_frames, 1), 1.0)
            progress_bar.progress(pct, text=f"Frame {frame_idx} / {total_frames} — {processed} processed")

        progress_bar.progress(1.0, text="Done.")
        st.success(f"Processed **{processed} frames** (1 of every {frame_skip}) from **{uploaded_video.name}**.")

        # ── Final summary metrics ──────────────────────────────────────────
        if lats_a:
            st.subheader("Video Processing Summary")
            avg_lat_a  = sum(lats_a)  / len(lats_a)
            avg_lat_b  = sum(lats_b)  / len(lats_b)
            avg_conf_a = sum(confs_a) / len(confs_a)
            avg_conf_b = sum(confs_b) / len(confs_b)
            avg_saving = avg_lat_b - avg_lat_a

            sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
            sm1.metric("Frames Processed",     processed)
            sm2.metric("Avg Adaptive Latency", f"{avg_lat_a:.1f} ms")
            sm3.metric("Avg Baseline Latency", f"{avg_lat_b:.1f} ms")
            sm4.metric("Avg Latency Savings",  f"{avg_saving:.1f} ms",
                       delta=f"{avg_saving:.1f} ms", delta_color="normal")
            sm5.metric("Avg Adaptive Conf",    f"{avg_conf_a:.2f}")
            sm6.metric("Avg Baseline Conf",    f"{avg_conf_b:.2f}")
    finally:
        cap.release()
        ws_conn.close()
        os.unlink(tmp.name)

else:
    # Stream stopped — reset history
    st.session_state.adaptive_lats  = []
    st.session_state.baseline_lats  = []
    st.session_state.adaptive_confs = []
    st.session_state.baseline_confs = []
