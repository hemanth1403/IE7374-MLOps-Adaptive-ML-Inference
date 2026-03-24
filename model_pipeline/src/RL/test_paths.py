import os
import cv2

def test_image_loading():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Project Root: ~/IE7374-MLOps-Adaptive-ML-Inference/
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    
    # 2. Path to the split file
    # Location: ~/IE7374-MLOps-Adaptive-ML-Inference/Data-Pipeline/data/splits/train.txt
    split_file = os.path.join(project_root, "Data-Pipeline/data/splits/train.txt")

    if not os.path.exists(split_file):
        print(f"Split file NOT FOUND at: {split_file}")
        return

    with open(split_file, 'r') as f:
        # first_image_rel is "data/processed/images/train2017/000000000009.jpg"
        first_image_rel = f.readline().strip()

    # 3. Construct the Full Path
    # Result: ~/IE7374-MLOps-Adaptive-ML-Inference/Data-Pipeline/data/processed/images/train2017/000000000009.jpg
    full_path = os.path.join(project_root, "Data-Pipeline", first_image_rel)
    
    print(f"Checking image path: {full_path}")
    
    img = cv2.imread(full_path)
    if img is not None:
        print(f"SUCCESS! Image shape: {img.shape}")
    else:
        print(f"FAILURE! Still cannot read the image.")
        # Diagnostic: List files in that directory to see what's wrong
        parent_dir = os.path.dirname(full_path)
        if os.path.exists(parent_dir):
            print(f"Parent directory exists. First 5 files: {os.listdir(parent_dir)[:5]}")
        else:
            print(f"Parent directory DOES NOT EXIST: {parent_dir}")

if __name__ == "__main__":
    test_image_loading()