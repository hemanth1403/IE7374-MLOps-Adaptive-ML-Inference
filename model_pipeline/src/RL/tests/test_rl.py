import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import torch
import numpy as np
import tqdm
from core.environment import AdaptiveInferenceEnv
from core.agent import NeuralBanditAgent

def evaluate_agent(model_path=None, steps=500):
    if model_path is None:
        model_path = os.path.join(_RL_ROOT, "models", "rl_bandit_v1.pth")

    csv_path = os.path.join(_RL_ROOT, "model_performance_profile.csv")
    env = AdaptiveInferenceEnv(csv_path=csv_path)

    state_dim = env.observation_space.shape[0]
    agent = NeuralBanditAgent(input_dim=state_dim, epsilon=0.0)

    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    print(f"Loaded model from {model_path}. Starting evaluation...")

    state, _ = env.reset()
    results = {0: 0, 1: 0, 2: 0}
    total_reward = 0

    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            action = agent.select_action(state)

        next_state, reward, _, _, _ = env.step(action)

        results[action] += 1
        total_reward += reward
        state = next_state

    print(f"\n--- Inference Results over {steps} steps ---")
    print(f"Average Reward: {total_reward/steps:.4f}")
    print(f"Model Selection Breakdown:")
    print(f" - Nano  (0): {results[0]} times")
    print(f" - Small (1): {results[1]} times")
    print(f" - Large (2): {results[2]} times")

if __name__ == "__main__":
    evaluate_agent()
