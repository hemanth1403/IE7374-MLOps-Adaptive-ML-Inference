import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import cv2
from pathlib import Path

class AdaptiveInferenceEnv(gym.Env):
    def __init__(self, dataset_path, window_size=10, latency_weight=0.5):
        super(AdaptiveInferenceEnv, self).__init__()
        self.window_size = window_size
        self.alpha = latency_weight 
        self.action_space = spaces.Discrete(3)
        
        # FIX: Changed from 1027 to 1028
        self.observation_shape = 1028 
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.observation_shape,), dtype=np.float32
        )

        self.data = self._load_dataset(dataset_path)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.zeros(self.observation_shape, dtype=np.float32), {}

    def step(self, action):
        obs = self._get_next_obs() 
        edge_density = obs[1024] # 1025th element
        
        base_acc = {0: 0.70, 1: 0.85, 2: 0.95}[action]
        latency = {0: 0.008, 1: 0.020, 2: 0.045}[action]
        
        # Simulated penalty: Nano struggles with complex scenes
        if action == 0 and edge_density > 0.4:
            accuracy = base_acc - (edge_density * 0.3) 
        else:
            accuracy = base_acc

        reward = accuracy - (self.alpha * latency)
        self.current_step += self.window_size
        done = self.current_step + self.window_size >= len(self.data)
        
        return obs, reward, done, False, {}

    def _get_next_obs(self):
        if not self.data or self.current_step >= len(self.data):
            return np.zeros(self.observation_shape, dtype=np.float32)

        img_rel_path = self.data[self.current_step].strip()
        project_root = Path(__file__).resolve().parents[3]
        full_path = project_root / "Data-Pipeline" / img_rel_path
        frame = cv2.imread(str(full_path))
        if frame is None:
            # If image fails, return 1028 zeros to prevent crash but log it
            if self.current_step == 0:
                print(f"CRITICAL ERROR: Could not load image at {full_path}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        from features import FeatureExtractor
        fe = FeatureExtractor()
        vis = fe.get_visual_features(frame) # 1024
        edge = fe.get_edge_density(frame)   # 1
        metadata = np.array([0.0, 0.8, 0.0], dtype=np.float32) # 3
        
        return fe.construct_state(vis, edge, metadata) # Total 1028

    def _load_dataset(self, path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        split_file = Path(__file__).resolve().parents[3] / "Data-Pipeline" / "data" / "splits" / "train.txt"
        
        if os.path.exists(split_file):
            with open(str(split_file), "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                print(f"SUCCESS: Loaded {len(lines)} images from train.txt")
                return lines
        return []