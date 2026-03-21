import cv2
import numpy as np

class FeatureExtractor:
    """
    Extracts lightweight features from a frame to act as RL Context.
    Goal: Execution time < 2ms.
    """
    def __init__(self, resize_dim=(32, 32)):
        self.resize_dim = resize_dim

    def get_visual_features(self, frame):
        """
        Processes raw BGR frame into a flattened grayscale vector.
        """
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Downsample (32x32 is usually enough to sense 'clutter')
        resized = cv2.resize(gray, self.resize_dim, interpolation=cv2.INTER_AREA)
        
        # 3. Normalize (0 to 1) and flatten
        return (resized.flatten() / 255.0).astype(np.float32)

    def get_edge_density(self, frame):
        """
        Calculates Canny edge density as a proxy for scene complexity.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return np.array([density], dtype=np.float32)

    def construct_state(self, visual_vec, edge_val, metadata):
        """
        Combines visual data with temporal metadata.
        metadata: [prev_model_idx, avg_conf, obj_count_delta]
        """
        # Concatenate everything into a single 1027-length vector
        state = np.concatenate([visual_vec, edge_val, metadata])
        return state

# --- Usage Example ---
# extractor = FeatureExtractor()
# vis_vec = extractor.get_visual_features(raw_frame)
# edge_val = extractor.get_edge_density(raw_frame)
# full_state = extractor.construct_state(vis_vec, edge_val, [0, 0.85, 2])



###########################


# Why these specific features?  

# 32x32 Grayscale: 
# This captures the "blobbiness" of a scene. 
# If the frame is mostly a gray road, the RL agent learns that's "simple." 
# If there are many high-contrast shapes, it's "complex."

# Canny Edge Density: 
# This is a classic "cheap" computer vision trick. 
# More edges almost always mean more objects or a busier background, which is a strong signal that you might need YOLOv8-Large.

# Float32 Precision: 
# We use np.float32 because neural networks (which we'll build in agent.py) are optimized for 32-bit floats. 
# Using 64-bit would just double your memory usage for no gain in accuracy.