import cv2
import torch
from ultralytics import YOLO
from features import FeatureExtractor
from agent import NeuralBanditAgent
from buffer_manager import WindowBufferManager

class MLOpsOrchestrator:
    def __init__(self, window_size=10):
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        
        # 1. Load YOLO Models (n, s, l)
        print("Loading YOLO variants...")
        self.models = {
            0: YOLO('yolov8n.pt').to(self.device),
            1: YOLO('yolov8s.pt').to(self.device),
            2: YOLO('yolov8l.pt').to(self.device)
        }
        
        # 2. Initialize Components
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

            # --- STEP 1: Feature Extraction ---
            vis_feat = self.extractor.get_visual_features(frame)
            edge_val = self.extractor.get_edge_density(frame)
            
            # --- STEP 2: Inference with CURRENT model ---
            # We use the model decided by the RL agent in the PREVIOUS window
            results = self.models[self.current_model_idx](frame, verbose=False)
            
            # Extract metrics for the buffer
            conf = results[0].probs.top1conf.item() if results[0].probs else 0.5
            obj_count = len(results[0].boxes)
            
            # --- STEP 3: Buffer Update ---
            self.buffer.add_frame_data(vis_feat, edge_val, conf, obj_count)
            
            # --- STEP 4: RL Decision Point ---
            if self.buffer.is_window_complete():
                avg_vis, avg_edge, metadata = self.buffer.get_aggregated_state(self.current_model_idx)
                state = self.extractor.construct_state(avg_vis, [avg_edge], metadata) # simplifying edge for now
                
                # Agent picks model for the NEXT window
                self.current_model_idx = self.agent.select_action(state)
                print(f"Switching to Model: {self.current_model_idx} for next window.")
                
                self.buffer.reset_window()

            # Optional: Visualization
            # cv2.imshow("Adaptive Inference", results[0].plot())
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()

if __name__ == "__main__":
    orchestrator = MLOpsOrchestrator(window_size=10)
    orchestrator.run_inference("test_video.mp4")






#######################

# Key Technical Considerations
# Memory Management: Loading three YOLO models and a Neural Network onto an GPU, as YOLOv8 models are quite small. 
# However, ensure you don't have other heavy GPU tasks running in the background.

# The "Feedback Loop": Notice that the RL Agent uses the performance of the current model to decide the next model. 
# This creates a temporal link—if the "Small" model starts struggling with confidence, the agent sees that in the buffer and upgrades to "Large" for the next 10 frames.

# Real-time vs. Training: In this script, the agent.update() call is missing. 
# Usually, you would run a "Training" version of this script where you also have access to the Ground Truth labels so you can calculate the reward and call agent.update().