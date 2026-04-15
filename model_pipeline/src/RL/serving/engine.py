"""
engine.py — Modular inference engine for the Adaptive ML Inference System.

Handles:
  - PyTorch 2.6+ weights_only=False patch (applied at import time)
  - YOLO model loading on CUDA
  - PPO agent loading on CPU
  - 1028-dim observation construction (must match environment.py exactly)
  - Dual-path inference: RL-adaptive and YOLOv8-Small baseline
"""

# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH 2.6+ SECURITY PATCH
# Must be applied before any call that triggers torch.load (YOLO / SB3 load).
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.serialization

_original_torch_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_load
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np
from stable_baselines3 import PPO
from ultralytics import YOLO

from core.features import FeatureExtractor

MODEL_NAMES: List[str] = ["Nano", "Small", "Large"]

# Color palette used by the visualisation layer (BGR)
MODEL_COLORS: Dict[str, tuple] = {
    "Nano":  (0, 255, 0),    # green
    "Small": (0, 255, 255),  # yellow
    "Large": (0, 0, 255),    # red
}


@dataclass
class InferenceResult:
    model_name: str
    detections: List[Dict[str, Any]]
    latency_ms: float
    object_count: int
    avg_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name":     self.model_name,
            "detections":     self.detections,
            "latency_ms":     round(self.latency_ms, 2),
            "object_count":   self.object_count,
            "avg_confidence": round(self.avg_confidence, 4),
        }


class AdaptiveInferenceSystem:
    """
    Standalone engine that routes frames between three YOLOv8 variants
    (Nano / Small / Large) using a trained PPO reinforcement learning agent.

    Observation space (1028-dim) matches environment.py exactly:
        [1024 visual features | 1 scaled edge density (×10) | 3 metadata]

    Parameters
    ----------
    rl_model_path : str
        Path to the trained PPO .zip file.
    yolo_n_path, yolo_s_path, yolo_l_path : str
        Paths to the YOLOv8 nano / small / large .pt weights.
    device : str
        Torch device for YOLO inference ("cuda" or "cpu").
    """

    def __init__(
        self,
        rl_model_path: str,
        yolo_n_path: str = "yolov8n.pt",
        yolo_s_path: str = "yolov8s.pt",
        yolo_l_path: str = "yolov8l.pt",
        device: str = "cuda",
        decision_interval: int = 5,
    ) -> None:
        self.device = device
        self.extractor = FeatureExtractor()

        # RL agent on CPU — keeps GPU headroom for YOLO inference
        print(f"[Engine] Loading PPO agent from: {rl_model_path}")
        self.agent = PPO.load(rl_model_path, device="cpu")

        # Three YOLO variants — prefer .onnx (faster CPU) over .pt when available
        print(f"[Engine] Loading YOLO n/s/l on {device} …")
        _pairs = [
            AdaptiveInferenceSystem._load_yolo(yolo_n_path, device),
            AdaptiveInferenceSystem._load_yolo(yolo_s_path, device),
            AdaptiveInferenceSystem._load_yolo(yolo_l_path, device),
        ]
        self.models: List[YOLO] = [m for m, _ in _pairs]
        self._is_onnx: List[bool] = [f for _, f in _pairs]

        # RL state — must be reset between independent sessions
        self.prev_action: int = 0
        self.prev_conf: float = 0.5

        # Decision throttling: re-evaluate the RL policy only every N frames.
        self.decision_interval: int = max(1, decision_interval)
        self._frame_count: int = 0
        self._current_action: int = 0

        # Warm up CUDA kernels on all three models.
        self._warmup()

        print("[Engine] Ready.")

    @staticmethod
    def _load_yolo(path: str, device: str):
        """Prefer .onnx over .pt for faster CPU inference (deployed only).
        Returns (YOLO model, is_onnx). ONNX models skip .to(device)."""
        onnx_path = os.path.splitext(path)[0] + ".onnx"
        if os.path.exists(onnx_path):
            print(f"[Engine] Using ONNX: {onnx_path}")
            return YOLO(onnx_path), True
        print(f"[Engine] Using PyTorch: {path}")
        return YOLO(path).to(device), False

    def _warmup(self) -> None:
        print("[Engine] Warming up models (n/s/l) …")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for i, model in enumerate(self.models):
            if self._is_onnx[i]:
                model(dummy, verbose=False)
            else:
                model(dummy, verbose=False, device=self.device)
        print("[Engine] Warm-up complete.")

    def _build_obs(self, frame: np.ndarray) -> np.ndarray:
        """
        Build the 1028-dim observation vector that matches the training
        environment (environment.py → _get_obs):

          [visual_feats (1024)] + [edge * 10.0 (1)] + [prev_action/2, prev_conf, 0.0 (3)]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        vis_feats = (resized.flatten() / 255.0).astype(np.float32)

        edges = cv2.Canny(gray, 100, 200)
        edge_val = float(np.sum(edges > 0)) / (edges.shape[0] * edges.shape[1])
        scaled_edge = np.array([edge_val * 10.0], dtype=np.float32)

        metadata = np.array(
            [self.prev_action / 2.0, self.prev_conf, 0.0], dtype=np.float32
        )

        return np.concatenate([vis_feats, scaled_edge, metadata]).astype(np.float32)

    def _run_yolo(self, model: YOLO, frame: np.ndarray) -> InferenceResult:
        """Run one YOLO model on a frame and return structured detections."""
        t0 = time.perf_counter()
        idx = self.models.index(model)
        if self._is_onnx[idx]:
            results = model(frame, verbose=False)
        else:
            results = model(frame, verbose=False, device=self.device)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        boxes = results[0].boxes
        if len(boxes) > 0:
            avg_conf = float(torch.mean(boxes.conf).item())
            count = len(boxes)
            detections = [
                {
                    "bbox":       box.xyxy[0].tolist(),
                    "confidence": float(box.conf.item()),
                    "class_id":   int(box.cls.item()),
                    "class_name": results[0].names[int(box.cls.item())],
                }
                for box in boxes
            ]
        else:
            avg_conf, count, detections = 0.0, 0, []

        return InferenceResult(
            model_name="",
            detections=detections,
            latency_ms=latency_ms,
            object_count=count,
            avg_confidence=avg_conf,
        )

    def infer(self, frame: np.ndarray, baseline_model_name: str = "Small") -> Dict[str, Any]:
        """
        Dual-path inference on a single BGR frame.

        Path A — Adaptive: PPO agent selects the optimal YOLO variant.
        Path B — Baseline: runs the model specified by baseline_model_name.
        """
        self._frame_count += 1
        if self._frame_count % self.decision_interval == 1:
            obs = self._build_obs(frame)
            action_arr, _ = self.agent.predict(obs, deterministic=True)
            self._current_action = int(action_arr)

        action = self._current_action

        adaptive = self._run_yolo(self.models[action], frame)
        adaptive.model_name = MODEL_NAMES[action]

        self.prev_action = action
        self.prev_conf = adaptive.avg_confidence

        _baseline_index = {"Nano": 0, "Small": 1, "Large": 2}
        baseline_idx = _baseline_index.get(baseline_model_name, 1)
        baseline = self._run_yolo(self.models[baseline_idx], frame)
        baseline.model_name = baseline_model_name

        return {"adaptive": adaptive.to_dict(), "baseline": baseline.to_dict()}

    def reset_state(self) -> None:
        """Reset RL state tracking. Call at the start of each new session."""
        self.prev_action = 0
        self.prev_conf = 0.5
        self._frame_count = 0
        self._current_action = 0
