import numpy as np

class RewardCalculator:
    def __init__(self, alpha=0.5, beta=0.1, latency_budget=0.030):
        """
        alpha: Weight for latency penalty (higher = more aggressive saving)
        beta: Weight for switching penalty (higher = more stable/slower to change)
        latency_budget: Target latency in seconds (e.g., 30ms)
        """
        self.alpha = alpha
        self.beta = beta
        self.latency_budget = latency_budget

    def calculate(self, accuracy, latency, did_switch):
        """
        Calculates the final reward for a window.
        accuracy: mean confidence or mAP (0.0 to 1.0)
        latency: mean inference time in seconds
        did_switch: Boolean, True if the model changed from the last window
        """
        
        # 1. Base Accuracy Reward
        # We want to heavily penalize low accuracy to maintain the 95% goal
        acc_reward = accuracy if accuracy >= 0.90 else (accuracy * 0.5)
        
        # 2. Latency Penalty
        # If we are over budget, the penalty increases exponentially
        if latency > self.latency_budget:
            lat_penalty = self.alpha * (latency / self.latency_budget)**2
        else:
            lat_penalty = self.alpha * (latency / self.latency_budget)

        # 3. Switching Penalty (Stability)
        switch_penalty = self.beta if did_switch else 0.0

        total_reward = acc_reward - lat_penalty - switch_penalty
        
        return total_reward
    

########################################



# Strategic Tuning Guide
# As you move into the implementation phase, you'll need to "find the sweet spot" for your weights:
# 
# To prioritize the 95% Accuracy goal: 
# Increase the penalty for accuracy dropping below 0.90. 
# This forces the agent to use YOLOv8-Large as soon as the scene gets even slightly complex.
# 
# 
# To prioritize the 58% Latency reduction: 
# Increase $\alpha$. The agent will become "braver" and try to use YOLOv8-Nano even in moderately complex scenes to avoid the high latency cost.
# 
# 
# To fix "Control Chatter": 
# If you notice the system switching models every few seconds in your tests, increase $\beta$. 
# This makes the "cost" of switching higher than the gain in accuracy or speed.