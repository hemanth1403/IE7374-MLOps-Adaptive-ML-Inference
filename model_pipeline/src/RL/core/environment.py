import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import cv2
import os
import yaml
from core.features import FeatureExtractor

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_params():
    params_path = os.path.join(_RL_ROOT, "params.yaml")
    if os.path.exists(params_path):
        with open(params_path) as f:
            return yaml.safe_load(f)
    return {}

class AdaptiveInferenceEnv(gym.Env):
    def __init__(self, csv_path):
        super(AdaptiveInferenceEnv, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.extractor = FeatureExtractor()
        self.action_space = spaces.Discrete(3)

        # 1028-dim: 1024 visual + 1 edge density + 3 metadata
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1028,), dtype=np.float32)

        P = _load_params().get("reward", {})
        self.w_quality         = P.get("w_quality",         0.84)
        self.w_efficiency      = P.get("w_efficiency",      0.16)
        self.switching_penalty = P.get("switching_penalty", 0.02)
        self.episode_length    = P.get("episode_length",    2048)

        self.current_step = 0
        self.episode_start = 0
        self.prev_action  = 0
        self.prev_conf    = 0.5

    def _get_obs(self):
        row      = self.df.iloc[self.current_step]
        img_path = row['path']

        if os.path.exists(img_path):
            frame     = cv2.imread(img_path)
            vis_feats = self.extractor.get_visual_features(frame).flatten()
            edge_val  = float(np.atleast_1d(self.extractor.get_edge_density(frame))[0])
        else:
            vis_feats = np.zeros(1024, dtype=np.float32)
            edge_val  = 0.0

        scaled_edge = np.array([edge_val * 10.0], dtype=np.float32)
        metadata    = np.array(
            [float(self.prev_action / 2.0), float(self.prev_conf), 0.0], dtype=np.float32
        )
        return np.concatenate([vis_feats, scaled_edge, metadata]).astype(np.float32)

    def step(self, action):
        row    = self.df.iloc[self.current_step]
        confs  = [row['n_conf'],  row['s_conf'],  row['l_conf']]
        lats   = [row['n_time'],  row['s_time'],  row['l_time']]
        counts = [row['n_count'], row['s_count'], row['l_count']]

        # ── Reward: ranking-normalised quality + efficiency ───────────────────
        # Step 1 — score each model on quality and efficiency
        #   quality    = confidence × √(object_count + 1)
        #   efficiency = quality / latency  (quality per unit time, rewards Nano on simple scenes)
        # Step 2 — blend: score = 0.84 × quality_norm + 0.16 × efficiency_norm
        #   where both norms are relative to the best model THIS step → range [0, 1]
        # Step 3 — rank-normalise the blended score across all three models:
        #   reward = (score[action] - min_score) / (max_score - min_score)
        #   → 1.0 = best possible choice this step, 0.0 = worst
        # This fills the full [0, 1] reward range every step, giving PPO strong
        # advantage signals and preventing value-function collapse to the mean.
        quality    = [confs[a] * np.sqrt(counts[a] + 1) for a in range(3)]
        efficiency = [quality[a] / (lats[a] + 1e-8)     for a in range(3)]

        q_max   = max(quality)    + 1e-8
        eff_max = max(efficiency) + 1e-8

        scores  = [self.w_quality * (quality[a] / q_max)
                   + self.w_efficiency * (efficiency[a] / eff_max)
                   for a in range(3)]

        s_min, s_max = min(scores), max(scores)
        spread = s_max - s_min

        if spread > 0.01:                          # models meaningfully differ
            reward = (scores[action] - s_min) / spread
        else:                                      # all models equivalent this step
            reward = 0.5

        tax    = self.switching_penalty if action != self.prev_action else 0.0
        reward -= tax

        self.prev_action = action
        self.prev_conf   = confs[action]
        self.current_step += 1

        steps_taken = self.current_step - self.episode_start
        done = steps_taken >= self.episode_length or self.current_step >= len(self.df) - 1
        obs  = self._get_obs() if not done else np.zeros(1028, dtype=np.float32)
        return obs, float(reward), done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random start: pick any position that leaves room for a full episode
        max_start = max(0, len(self.df) - self.episode_length - 1)
        self.episode_start = int(self.np_random.integers(0, max_start + 1))
        self.current_step  = self.episode_start
        self.prev_action   = 0
        self.prev_conf     = 0.5
        return self._get_obs(), {}
