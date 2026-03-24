import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BanditNetwork(nn.Module):
    def __init__(self, input_dim=1028, hidden_dim=128): # Updated to 1028
        super(BanditNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1), # Prevents overfitting to simple patterns
            nn.Linear(hidden_dim, hidden_dim), # Added an extra layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) 
        )

    def forward(self, x):
        return self.network(x)

class NeuralBanditAgent:
    def __init__(self, input_dim=1028, learning_rate=1e-3, epsilon=0.1): # Updated to 1028
        self.device = torch.device("cpu")
        self.model = BanditNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon 

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3) 
        
        self.model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_t)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward):
        self.model.train()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Match shape: [1]
        reward_t = torch.FloatTensor([reward]).to(self.device)
        
        predicted_rewards = self.model(state_t) 
        # Extract chosen action and keep it as [1] shape
        chosen_reward_pred = predicted_rewards[:, action] 

        # Both are now [1], eliminating the warning
        loss = self.criterion(chosen_reward_pred, reward_t)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()