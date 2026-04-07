import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import cv2
import time
import torch
from ultralytics import YOLO
from core.features import FeatureExtractor
from core.agent import NeuralBanditAgent
from core.buffer_manager import WindowBufferManager

class MLOpsOrchestrator:
    def __init__(self, window_size=10):
        self.device = 'cpu'

        print("Loading YOLO variants...")
        self.models = {
            0: YOLO('yolov8n.pt').to(self.device),
            1: YOLO('yolov8s.pt').to(self.device),
            2: YOLO('yolov8l.pt').to(self.device)
        }

        self.extractor = FeatureExtractor()
        self.agent = NeuralBanditAgent()
        self.buffer = WindowBufferManager(window_size=window_size)

        self.current_model_idx = 0  # Start with Nano
        self.window_size = window_size

    def run_inference(self, video_path):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            vis_feat = self.extractor.get_visual_features(frame)
            edge_val = self.extractor.get_edge_density(frame)

            results = self.models[self.current_model_idx](frame, verbose=False)

            conf = results[0].probs.top1conf.item() if results[0].probs else 0.5
            obj_count = len(results[0].boxes)

            self.buffer.add_frame_data(vis_feat, edge_val, conf, obj_count)

            if self.buffer.is_window_complete():
                avg_vis, avg_edge, metadata = self.buffer.get_aggregated_state(self.current_model_idx)
                state = self.extractor.construct_state(avg_vis, [avg_edge], metadata)

                self.current_model_idx = self.agent.select_action(state)
                print(f"Switching to Model: {self.current_model_idx} for next window.")

                self.buffer.reset_window()

        cap.release()
        return metrics

if __name__ == "__main__":
    orchestrator = MLOpsOrchestrator(window_size=10)
    orchestrator.run_inference("test_video.mp4")
