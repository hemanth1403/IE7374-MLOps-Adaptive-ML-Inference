# ─────────────────────────────────────────────────────────────────────────────
# Adaptive ML Inference System — Production Container
#
# Base: nvidia/cuda:12.1.0-runtime-ubuntu22.04
#   "runtime" includes libcudart needed by PyTorch without the full dev toolkit.
#   Requires NVIDIA Container Toolkit on the host:
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#
# Build (from model_pipeline/src/RL/):
#   docker build -t adaptive-inference .
#
# Run FastAPI backend (GPU):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     -v $(pwd)/yolov8n.pt:/app/yolov8n.pt:ro \
#     -v $(pwd)/yolov8s.pt:/app/yolov8s.pt:ro \
#     -v $(pwd)/yolov8l.pt:/app/yolov8l.pt:ro \
#     adaptive-inference
#
# Run Streamlit dashboard (connect to a remote backend):
#   docker run -p 8501:8501 \
#     -e WS_URL=ws://<backend-host>:8000/ws/stream \
#     adaptive-inference \
#     streamlit run serving/ui.py --server.port=8501 --server.address=0.0.0.0
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ── System packages ───────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        # OpenCV runtime dependencies
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── NVIDIA container toolkit env hints ────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_deploy.txt

# ── Application source ────────────────────────────────────────────────────────
# Model weights (.pt / .zip) are large binaries — mount as read-only volumes
# at runtime rather than baking into the image layer.
COPY core/ ./core/
COPY serving/ ./serving/

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8000 — FastAPI / uvicorn
# 8501 — Streamlit
EXPOSE 8000 8501

# ── Default: start the FastAPI inference server ───────────────────────────────
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
