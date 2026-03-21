import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BanditNetwork(nn.Module):
    """
    A lightweight MLP that predicts the expected reward for each 
    of the 3 actions (Nano, Small, Large).
    """
    def __init__(self, input_dim=1027, hidden_dim=64):
        super(BanditNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # 3 outputs: Q-values for N, S, L
        )

    def forward(self, x):
        return self.network(x)

class NeuralBanditAgent:
    def __init__(self, input_dim=1027, learning_rate=1e-3, epsilon=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BanditNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Exploration rate
        self.epsilon = epsilon 

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3) # Explore
        
        self.model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_t)
            return torch.argmax(q_values).item() # Exploit best predicted model

    def update(self, state, action, reward):
        """
        Trains the network based on the observed reward.
        """
        self.model.train()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([reward]).to(self.device)
        
        # Predict reward for the chosen action
        predicted_rewards = self.model(state_t)
        chosen_reward_pred = predicted_rewards[0, action]
        
        # Backpropagate
        loss = self.criterion(chosen_reward_pred, reward_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    



#############################


# Why this design works for your project:
    # Hidden Dimension (64): 
    # We keep the hidden layer small. A massive network would take too long to compute. 
    # At 64 neurons, the inference time on your GPU will be measured in microseconds.

    # $\epsilon$-greedy Exploration: 
    # During training, the agent will occasionally pick a "wrong" model (like using Large for an empty room) just to confirm that the reward is indeed lower. 
    # This is how it learns the trade-off.

    # MSE Loss: 
    # Since the agent is trying to predict a continuous "Reward" value (Accuracy - Latency), Mean Squared Error is the standard loss function for this type of regression.

# The "Switching Penalty" Note:
# When you integrate this with buffer_manager.py, the state input already contains the prev_model_idx. 
# The agent will naturally learn that if it switches models too often, the "Stability" part of your reward function (which we can define in reward_functions.py) will go down.
