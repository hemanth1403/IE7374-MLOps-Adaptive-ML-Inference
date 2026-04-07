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