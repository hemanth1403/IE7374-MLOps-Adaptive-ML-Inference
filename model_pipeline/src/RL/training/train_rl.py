import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from core.environment import AdaptiveInferenceEnv

def train():
    with open(os.path.join(_RL_ROOT, "params.yaml")) as f:
        P = yaml.safe_load(f)

    csv_path   = os.path.join(_RL_ROOT, P["paths"]["profile_csv"])
    models_dir = os.path.join(_RL_ROOT, "models", "PPO_v5")
    log_dir    = os.path.join(_RL_ROOT, P["paths"]["log_dir"])

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Initializing Environment (Aggressive 10.0 Alpha / 1031-dim)...")
    raw_env = AdaptiveInferenceEnv(csv_path=csv_path)
    env = Monitor(raw_env, log_dir)
    env = DummyVecEnv([lambda: env])

    # ent_coef=0.05 forces exploration across all three models.
    # alpha=1.5 in environment.py makes Large viable — it won't be crushed by latency penalty.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.05,  # Increased from 0.01 — forces exploration away from Small attractor
        device="cpu"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="adaptive_yolo_v7_short_ep"
    )

    print("Starting training. Goal: Balanced routing across Nano / Small / Large.")
    model.learn(total_timesteps=3000000, callback=checkpoint_callback, progress_bar=True)

    model.save(os.path.join(models_dir, "final_adaptive_model.zip"))
    print("Training Complete.")

if __name__ == "__main__":
    train()
