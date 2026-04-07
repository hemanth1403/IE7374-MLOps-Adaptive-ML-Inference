"""
evaluate_policy.py — standalone policy evaluation for the DVC pipeline.

Runs a 1000-step deterministic rollout on the trained PPO_v6 model,
writes results to metrics.json, and exits non-zero if routing is unbalanced
(any model used < 20% of the time).

Usage:
    python training/evaluate_policy.py
"""
import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import json
import yaml
import numpy as np
from stable_baselines3 import PPO
from core.environment import AdaptiveInferenceEnv

with open(os.path.join(_RL_ROOT, "params.yaml")) as f:
    P = yaml.safe_load(f)

CSV_PATH     = os.path.join(_RL_ROOT, P["paths"]["profile_csv"])
MODEL_PATH   = os.path.join(_RL_ROOT, P["paths"]["final_model"])
METRICS_PATH = os.path.join(_RL_ROOT, P["paths"]["metrics"])

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device="cpu")
    env   = AdaptiveInferenceEnv(csv_path=CSV_PATH)

    obs, _ = env.reset()
    acts, rewards = [], []

    print("Running 1000-step evaluation …")
    for _ in range(1000):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _, _ = env.step(act)
        acts.append(int(act))
        rewards.append(float(rew))
        if done:
            obs, _ = env.reset()

    counts   = np.bincount(acts, minlength=3)
    avg_rew  = float(np.mean(rewards))
    pct_nano  = round(counts[0] / 10, 1)
    pct_small = round(counts[1] / 10, 1)
    pct_large = round(counts[2] / 10, 1)

    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Nano:  {counts[0]:4d}  ({pct_nano:.1f}%)")
    print(f"Small: {counts[1]:4d}  ({pct_small:.1f}%)")
    print(f"Large: {counts[2]:4d}  ({pct_large:.1f}%)")
    print(f"Avg Reward: {avg_rew:.4f}")

    metrics = {
        "avg_reward": round(avg_rew, 4),
        "pct_nano":   pct_nano,
        "pct_small":  pct_small,
        "pct_large":  pct_large,
    }
    with open(METRICS_PATH, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"\nMetrics written → {METRICS_PATH}")

    # Quality gate: every model should be used at least 20% of the time
    if pct_nano < 20 or pct_large < 20:
        print(f"\nFAIL: Policy is unbalanced (Nano={pct_nano}%, Large={pct_large}%).")
        print("Retrain with: python training/pretrain_bc.py")
        sys.exit(1)

    print("PASS: Policy is balanced.")

if __name__ == "__main__":
    main()
