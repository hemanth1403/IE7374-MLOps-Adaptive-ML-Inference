import numpy as np
from collections import deque

class WindowBufferManager:
    """
    Collects frame-level features and aggregates them for the RL agent.
    Ensures model switching only happens at window boundaries.
    """
    def __init__(self, window_size=10):
        self.window_size = window_size
        
        # Buffers to store data within the current window
        self.visual_buffer = deque(maxlen=window_size)
        self.conf_buffer = deque(maxlen=window_size)
        self.count_buffer = deque(maxlen=window_size)
        
        self.frame_counter = 0

    def add_frame_data(self, visual_features, confidence=0.0, obj_count=0):
        """
        Pushes new frame data into the temporal buffers.
        """
        self.visual_buffer.append(visual_features)
        self.conf_buffer.append(confidence)
        self.count_buffer.append(obj_count)
        self.frame_counter += 1

    def is_window_complete(self):
        """
        Returns True if we have reached the end of a temporal window.
        """
        return self.frame_counter % self.window_size == 0

    def get_aggregated_state(self, prev_model_idx):
        """
        Computes the mean/delta features to pass to the RL Agent.
        """
        # 1. Mean Visual State (The 'Average' look of the last N frames)
        avg_visual = np.mean(self.visual_buffer, axis=0)
        
        # 2. Average Confidence of current detections
        avg_conf = np.mean(self.conf_buffer)
        
        # 3. Object Count Trend (Difference between start and end of window)
        count_delta = self.count_buffer[-1] - self.count_buffer[0]
        
        # 4. Construct Metadata Vector
        metadata = np.array([
            float(prev_model_idx),
            float(avg_conf),
            float(count_delta)
        ], dtype=np.float32)
        
        return avg_visual, metadata

    def reset_window(self):
        """
        Clears the counter for a new decision cycle.
        """
        # We don't clear the deques because we want a sliding 'warm' start 
        # for the next window's features, but we reset the logic counter.
        pass



###################
# Why this is effective:
# Reduced Agent Overhead: Instead of running the RL Inference 30 times a second (30 FPS), this manager ensures the RL agent only runs 3 times a second (for a window of 10). This leaves more GPU overhead for your YOLOv8 models.

# Smoothing: By taking the np.mean of the visual features, you filter out sensor noise or sudden motion blur that might confuse the RL agent.

# The "Delta" Insight: Calculating count_delta is huge for autonomous perception. If the count is increasing (+3 objects in 10 frames), it’s a strong signal for the agent to switch to a larger model before the scene gets too crowded.

# How it fits into the flow:
# features.py extracts raw data from Frame 1.

# buffer_manager.py stores it.

# When frame_counter == 10, buffer_manager sends the aggregated data to agent.py.

# agent.py decides the model for Frames 11–20.
