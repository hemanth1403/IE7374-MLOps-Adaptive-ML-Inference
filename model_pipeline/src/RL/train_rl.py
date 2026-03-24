import torch
import numpy as np
from environment import AdaptiveInferenceEnv
from agent import NeuralBanditAgent
from tqdm import tqdm  # Ensure this is installed via pip
from pathlib import Path

def train_agent(episodes=500, steps_per_episode=1000):

    dataset_path = str(Path(__file__).resolve().parents[3] / "Data-Pipeline" / "data" / "processed")
    env = AdaptiveInferenceEnv(dataset_path=dataset_path, latency_weight=0.2)
    
    # Verify data loading
    if len(env.data) == 0:
        print("Error: No images found in train.txt. Check your paths!")
        return

    # agent = NeuralBanditAgent(input_dim=1027)
    state_dim = env.observation_space.shape[0] 
    agent = NeuralBanditAgent(input_dim=state_dim)
    
    print(f"Starting training on {agent.device}...")
    print(f"Dataset size: {len(env.data)} images ({len(env.data)//10} potential windows)")

    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        
        # Use tqdm for a live progress bar per episode
        pbar = tqdm(total=steps_per_episode, desc=f"Episode {episode+1}/{episodes}")
        
        for step in range(steps_per_episode):
            action = agent.select_action(state)
            if step % 50 == 0:
                print(f"Current Action choice: {action}") # 0=Nano, 1=Small, 2=Large
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update the agent
            loss = agent.update(state, action, reward)
            
            state = next_state
            episode_reward += reward
            
            pbar.update(1)
            pbar.set_postfix({"reward": f"{reward:.4f}", "epsilon": f"{agent.epsilon:.2f}"})
            
            if done:
                break
        
        pbar.close()
        
        # Decay exploration
        # agent.epsilon = max(0.01, agent.epsilon * 0.95)
        epsilon = max(0.01, 0.2 * (0.95 ** episode))

    # 6. Save the trained policy
    torch.save(agent.model.state_dict(), "models/rl_bandit_v1.pth")
    print("\nTraining complete. Model saved to models/rl_bandit_v1.pth")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_agent(episodes=50, steps_per_episode=200)