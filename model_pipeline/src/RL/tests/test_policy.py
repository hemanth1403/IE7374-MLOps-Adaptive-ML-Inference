import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import gymnasium as gym
from stable_baselines3 import PPO
from core.environment import AdaptiveInferenceEnv
import numpy as np

def verify_model():
    csv_path = os.path.join(_RL_ROOT, "model_performance_profile.csv")
    # Use latest available model: v3 > v2 > v1
    for version in ["PPO_v6", "PPO_v5", "PPO_v4", "PPO_v3", "PPO_v2", "PPO"]:
        candidate = os.path.join(_RL_ROOT, "models", version, "final_adaptive_model.zip")
        if os.path.exists(candidate):
            model_path = candidate
            break

    env = AdaptiveInferenceEnv(csv_path=csv_path)
    model = PPO.load(model_path)

    actions_taken = []
    rewards = []

    print("Testing model on 1000 samples...")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        actions_taken.append(action)
        rewards.append(reward)

        if done:
            obs, _ = env.reset()

    actions_taken = np.array(actions_taken)
    counts = np.bincount(actions_taken, minlength=3)

    print("\n--- TEST RESULTS ---")
    print(f"Action 0 (Nano) chosen:  {counts[0]} times ({counts[0]/10:.1f}%)")
    print(f"Action 1 (Small) chosen: {counts[1]} times ({counts[1]/10:.1f}%)")
    print(f"Action 2 (Large) chosen: {counts[2]} times ({counts[2]/10:.1f}%)")
    print(f"Average Reward: {np.mean(rewards):.4f}")

if __name__ == "__main__":
    verify_model()
