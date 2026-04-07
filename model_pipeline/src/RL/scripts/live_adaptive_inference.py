import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import torch

# ==========================================
# PYTORCH 2.6+ SECURITY BYPASS
# This MUST be at the very top of your script
# ==========================================
import torch.serialization
original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load
# ==========================================

import cv2
import numpy as np
from ultralytics import YOLO
from stable_baselines3 import PPO
from core.features import FeatureExtractor
import time

class AdaptiveInferenceSystem:
    def __init__(self, rl_model_path):
        print(f"Loading RL Strategy from {rl_model_path}...")
        self.agent = PPO.load(rl_model_path, device='cpu')

        print("Loading YOLO models (Nano, Small, Large)...")
        self.yolo_n = YOLO("yolov8n.pt").to('cuda')
        self.yolo_s = YOLO("yolov8s.pt").to('cuda')
        self.yolo_l = YOLO("yolov8l.pt").to('cuda')

        self.models = [self.yolo_n, self.yolo_s, self.yolo_l]
        self.extractor = FeatureExtractor()

        self.prev_action = 0
        self.prev_conf = 0.5

    def run_live(self):
        cap = cv2.VideoCapture(0)

        print("\n--- System Live on Zephyrus G14 ---")
        print("Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            # --- 1. FEATURE EXTRACTION (1028-dim) ---
            vis_feats = self.extractor.get_visual_features(frame).flatten()

            edge_val = float(np.atleast_1d(self.extractor.get_edge_density(frame))[0])
            scaled_edge = np.array([edge_val * 10.0], dtype=np.float32)

            metadata = np.array([
                float(self.prev_action / 2.0),
                float(self.prev_conf),
                0.0
            ], dtype=np.float32)

            state = np.concatenate([vis_feats, scaled_edge, metadata]).astype(np.float32)

            # --- 2. RL STRATEGY PHASE ---
            action_output, _ = self.agent.predict(state, deterministic=True)
            action = int(action_output)

            model_names = ["Nano", "Small", "Large"]
            model_name = model_names[action]

            # --- 3. INFERENCE PHASE ---
            results = self.models[action](frame, verbose=False, device='cuda')

            latency_ms = (time.time() - start_time) * 1000

            # --- 4. DATA UPDATE ---
            if len(results[0].boxes) > 0:
                conf = float(torch.mean(results[0].boxes.conf))
                count = len(results[0].boxes)
            else:
                conf, count = 0.0, 0

            self.prev_action = action
            self.prev_conf = conf

            # --- 5. VISUALIZATION ---
            annotated_frame = results[0].plot()

            colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}
            current_color = colors.get(action, (255, 255, 255))

            cv2.putText(annotated_frame, f"RL MODEL: {model_name}", (20, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, current_color, 2)
            cv2.putText(annotated_frame, f"Latency: {latency_ms:.1f} ms", (20, 95),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow("Adaptive MLOps Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = os.path.join(_RL_ROOT, "models", "PPO_v1", "final_adaptive_model.zip")
    system = AdaptiveInferenceSystem(rl_model_path=MODEL_PATH)
    system.run_live()
