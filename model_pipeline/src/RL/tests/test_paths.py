import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import cv2

def test_image_loading():
    # Climb from RL root to project root
    project_root = _RL_ROOT
    while project_root != os.path.dirname(project_root):
        if os.path.basename(project_root) == "IE7374-MLOps-Adaptive-ML-Inference":
            break
        project_root = os.path.dirname(project_root)

    split_file = os.path.join(project_root, "Data-Pipeline/data/splits/train.txt")

    if not os.path.exists(split_file):
        print(f"Split file NOT FOUND at: {split_file}")
        return

    with open(split_file, 'r') as f:
        first_image_rel = f.readline().strip()

    full_path = os.path.join(project_root, "Data-Pipeline", first_image_rel)

    print(f"Checking image path: {full_path}")

    img = cv2.imread(full_path)
    if img is not None:
        print(f"SUCCESS! Image shape: {img.shape}")
    else:
        print(f"FAILURE! Still cannot read the image.")
        parent_dir = os.path.dirname(full_path)
        if os.path.exists(parent_dir):
            print(f"Parent directory exists. First 5 files: {os.listdir(parent_dir)[:5]}")
        else:
            print(f"Parent directory DOES NOT EXIST: {parent_dir}")

if __name__ == "__main__":
    test_image_loading()
