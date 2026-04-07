import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import cv2
import torch
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO

def get_project_paths():
    """
    Dynamically locates the project root and necessary data folders.
    """
    project_root = _RL_ROOT
    while project_root != os.path.dirname(project_root):
        if os.path.basename(project_root) == "IE7374-MLOps-Adaptive-ML-Inference":
            break
        project_root = os.path.dirname(project_root)

    data_dir = os.path.join(project_root, "Data-Pipeline")
    split_file = os.path.join(data_dir, "data/splits/train.txt")

    return project_root, data_dir, split_file

def profile_dataset():
    root, data_dir, split_file = get_project_paths()

    print(f"Project Root Detected: {root}")
    print(f"Targeting Split File: {split_file}")

    if not os.path.exists(split_file):
        print(f"ERROR: Could not find {split_file}")
        print("Please ensure your 'Data-Pipeline' folder is in the project root.")
        return

    # Load Models onto CPU (Safe for RTX 5060 mismatch)
    print("Loading YOLO models...")
    models = {
        'n': YOLO("yolov8n.pt").to('cpu'),
        's': YOLO("yolov8s.pt").to('cpu'),
        'l': YOLO("yolov8l.pt").to('cpu')
    }

    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    results = []
    total = len(lines)
    print(f"Profiling {total} images. This measures actual CPU latency on your G14.")

    for i, relative_path in enumerate(lines):
        img_path = os.path.join(data_dir, relative_path)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image at {img_path}")
            continue

        row = {'path': img_path}

        for name, model in models.items():
            start = time.perf_counter()
            res = model(frame, verbose=False)
            latency = time.perf_counter() - start

            if len(res[0].boxes) > 0:
                conf = float(torch.mean(res[0].boxes.conf))
                count = len(res[0].boxes)
            else:
                conf, count = 0.0, 0

            row[f'{name}_conf'] = conf
            row[f'{name}_time'] = latency
            row[f'{name}_count'] = count

        results.append(row)

        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{total} profiled...")

    # Save CSV to RL root so train_rl.py can find it
    output_path = os.path.join(_RL_ROOT, "model_performance_profile.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"\nSUCCESS! Profile saved to: {output_path}")
    print("You can now use this CSV to train your RL agent 100x faster.")

if __name__ == "__main__":
    profile_dataset()
