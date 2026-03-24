import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from orchestrator import MLOpsOrchestrator
import numpy as np

class Evaluator:
    def __init__(self, test_video_path):
        self.video_path = test_video_path
        self.results = []

    def run_benchmark(self, mode="RL", window_size=10):
        """
        Runs the system and records Accuracy (Confidence) and Latency.
        mode: "RL", "Fixed-Nano", or "Fixed-Large"
        """
        orchestrator = MLOpsOrchestrator(window_size=window_size)
        
        # Override RL agent if we want a fixed baseline
        if mode == "Fixed-Nano":
            orchestrator.current_model_idx = 0
            # Disable switching logic for fixed baseline
            orchestrator.buffer.is_window_complete = lambda: False 
        elif mode == "Fixed-Large":
            orchestrator.current_model_idx = 2
            orchestrator.buffer.is_window_complete = lambda: False

        start_time = time.time()
        # This will run the orchestrator loop and we'll capture metrics
        # For evaluation, we assume orchestrator returns a list of (acc, lat)
        metrics = orchestrator.run_inference(self.video_path) 

        if not metrics:
            print(f"No metrics collected for mode: {mode}")
            return
        
        avg_acc = np.mean([m[0] for m in metrics])
        avg_lat = np.mean([m[1] for m in metrics])
        
        self.results.append({
            "Mode": mode,
            "Avg_Accuracy": avg_acc,
            "Avg_Latency_MS": avg_lat * 1000
        })

    def plot_results(self):
        df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(10, 6))
        for i, row in df.iterrows():
            plt.scatter(row['Avg_Latency_MS'], row['Avg_Accuracy'], s=100)
            plt.text(row['Avg_Latency_MS']+0.5, row['Avg_Accuracy'], row['Mode'])
        
        plt.title("Performance Trade-off: RL vs. Fixed Baselines")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Accuracy (Mean Confidence)")
        plt.grid(True)
        plt.savefig("performance_report.png")
        print("Report saved as performance_report.png")

if __name__ == "__main__":
    evaluator = Evaluator("test_data/val_sequence.mp4")
    
    print("Evaluating Fixed-Nano...")
    evaluator.run_benchmark("Fixed-Nano")
    
    print("Evaluating Fixed-Large...")
    evaluator.run_benchmark("Fixed-Large")
    
    print("Evaluating RL-Agent...")
    evaluator.run_benchmark("RL")
    
    evaluator.plot_results()