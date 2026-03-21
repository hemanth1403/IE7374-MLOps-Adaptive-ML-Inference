import torch
import numpy as np
from environment import AdaptiveInferenceEnv
from agent import NeuralBanditAgent
from features import FeatureExtractor
import tqdm # For progress bars

def train_agent(episodes=100, steps_per_episode=500):
    # 1. Initialize Environment and Agent
    env = AdaptiveInferenceEnv(dataset_path="data/processed")
    agent = NeuralBanditAgent(input_dim=1027)
    extractor = FeatureExtractor()
    
    print(f"Starting training on {agent.device}...")

    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_episode):
            # 2. Agent selects an action (Nano, Small, or Large)
            action = agent.select_action(state)
            
            # 3. Environment executes action and returns reward
            # In training, this uses pre-computed mAP/Latency from your DVC logs
            next_state, reward, done, truncated, info = env.step(action)
            
            # 4. Update the Agent's Neural Network
            loss = agent.update(state, action, reward)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 5. Logging Progress
        if episode % 10 == 0:
            print(f"Episode {episode} | Avg Reward: {episode_reward/steps_per_episode:.4f} | Epsilon: {agent.epsilon:.2f}")
            # Decay exploration over time
            agent.epsilon = max(0.01, agent.epsilon * 0.95)

    # 6. Save the trained policy
    torch.save(agent.model.state_state_dict(), "models/rl_bandit_v1.pth")
    print("Training complete. Model saved to models/rl_bandit_v1.pth")

if __name__ == "__main__":
    # Ensure models directory exists
    import os
    os.makedirs("models", exist_ok=True)
    
    train_agent(episodes=200)






##########################

# Key Processing Steps in this File
# The Training Loop: It iterates through "episodes" (which could be different video sequences from your COCO dataset).

# Epsilon Decay: Notice the agent.epsilon * 0.95. This is a crucial RL technique. At the start, the agent explores randomly to see what happens when it picks Nano vs. Large. As training progresses, it becomes "smarter" and spends more time picking what it knows works (exploitation).

# Experience Replay (Optional): While this version updates "on-policy" (immediately), for more complex scenes, you could add a buffer here to store experiences and train in batches to stabilize the RTX 5060's GPU utilization.

# Model Saving: The .pth file is the final artifact. In your MLOps workflow, this is the file that you would run dvc add models/rl_bandit_v1.pth on.