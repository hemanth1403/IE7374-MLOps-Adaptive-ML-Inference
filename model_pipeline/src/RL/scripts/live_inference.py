import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from core.agent import NeuralBanditAgent
from core.features import FeatureExtractor

def run_live_inference(model_path=None, window_size=10):
    if model_path is None:
        model_path = os.path.join(_RL_ROOT, "models", "rl_bandit_v1.pth")

    models = {
        0: YOLO("yolov8n.pt").to('cpu'),
        1: YOLO("yolov8s.pt").to('cpu'),
        2: YOLO("yolov8l.pt").to('cpu')
    }

    # Based on features.py (1024 + 1 + 3 = 1028 dimensions)
    agent = NeuralBanditAgent(input_dim=1028, epsilon=0.0)
    agent.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.model.to('cpu')
    agent.device = 'cpu'
    agent.model.eval()

    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)

    current_model_idx = 0
    avg_conf = 0.0
    obj_count_delta = 0.0
    prev_obj_count = 0
    window_counter = 0

    print("Starting Live Adaptive Inference...")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if window_counter % window_size == 0:
            vis_vec = extractor.get_visual_features(frame)
            edge_val = extractor.get_edge_density(frame)
            metadata = np.array([current_model_idx, avg_conf, obj_count_delta], dtype=np.float32)

            state = extractor.construct_state(vis_vec, edge_val, metadata)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                current_model_idx = agent.select_action(state_tensor)

        current_model = models[current_model_idx]
        results = current_model(frame, verbose=False)

        if len(results[0].boxes) > 0:
            avg_conf = float(torch.mean(results[0].boxes.conf))
            current_count = len(results[0].boxes)
            obj_count_delta = float(current_count - prev_obj_count)
            prev_obj_count = current_count
        else:
            avg_conf = 0.0
            obj_count_delta = float(-prev_obj_count)
            prev_obj_count = 0

        annotated_frame = results[0].plot()
        model_names = ["Nano", "Small", "Large"]

        cv2.putText(annotated_frame, f"RL Selected: {model_names[current_model_idx]}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Adaptive RL Inference", annotated_frame)

        window_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference()
