import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import torch
import numpy as np
import cv2
from stable_baselines3 import PPO
from core.features import FeatureExtractor

TEST_IMAGES_DIR = os.path.join(_RL_ROOT, "test_images")

# Use latest available model: v6 > v5 > v4 > … > PPO
for _version in ["PPO_v6", "PPO_v5", "PPO_v4", "PPO_v3", "PPO_v2", "PPO"]:
    _candidate = os.path.join(_RL_ROOT, "models", _version, "final_adaptive_model.zip")
    if os.path.exists(_candidate):
        MODEL_PATH = _candidate
        break

model = PPO.load(MODEL_PATH, device="cpu")
extractor = FeatureExtractor()

def run_hypothesis(image_path, label):
    if not os.path.exists(image_path):
        print(f"Skipping {label}: {image_path} not found.")
        return

    frame = cv2.imread(image_path)

    vis_feats = extractor.get_visual_features(frame).flatten()

    edge_raw_val = float(np.atleast_1d(extractor.get_edge_density(frame))[0])
    scaled_edge = np.array([edge_raw_val * 10.0], dtype=np.float32)  # 1-dim, matching environment.py

    decisions = []
    prev_action = 0
    prev_conf = 0.5

    for _ in range(10):
        metadata = np.array([prev_action / 2.0, prev_conf, 0.0], dtype=np.float32)
        obs = np.concatenate([vis_feats, scaled_edge, metadata]).astype(np.float32)  # 1024+1+3=1028

        action, _ = model.predict(obs, deterministic=True)
        decisions.append(["Nano", "Small", "Large"][int(action)])
        prev_action = action

    print(f"\n--- {label} ---")
    print(f"Edge Density: {edge_raw_val:.4f} (Amplified ×10)")
    print(f"Agent Decisions: {decisions}")

if __name__ == "__main__":
    run_hypothesis(os.path.join(TEST_IMAGES_DIR, "Parking.jpeg"), "Hypothesis C (Complex)")
    run_hypothesis(os.path.join(TEST_IMAGES_DIR, "Wall.png"),     "Hypothesis A (Simple)")
    run_hypothesis(os.path.join(TEST_IMAGES_DIR, "Computer.png"), "Hypothesis B (Medium)")
