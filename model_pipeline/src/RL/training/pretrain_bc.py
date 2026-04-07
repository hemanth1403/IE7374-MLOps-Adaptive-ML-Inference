"""
Behavioral Cloning pre-trainer for the adaptive inference policy.

Generates (observation, optimal_action) pairs from the CSV dataset,
trains a supervised MLP classifier, and saves weights in a format
that can be loaded into a Stable Baselines3 PPO policy for fine-tuning.

Usage:
    python training/pretrain_bc.py
"""
import sys
import os
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from core.environment import AdaptiveInferenceEnv
from core.features import FeatureExtractor

# ─── Load params.yaml ─────────────────────────────────────────────────────────
_params_path = os.path.join(_RL_ROOT, "params.yaml")
with open(_params_path) as _f:
    _P = yaml.safe_load(_f)

SAMPLE_ROWS       = _P["bc"]["sample_rows"]
BC_EPOCHS         = _P["bc"]["epochs"]
BC_LR             = _P["bc"]["lr"]
BC_BATCH          = _P["bc"]["batch_size"]
HIDDEN_DIM        = _P["bc"]["hidden_dim"]
RL_FINETUNE_STEPS = _P["ppo"]["finetune_steps"]
CSV_PATH          = os.path.join(_RL_ROOT, _P["paths"]["profile_csv"])
MODELS_DIR        = os.path.join(_RL_ROOT, _P["paths"]["model_dir"])
LOG_DIR           = os.path.join(_RL_ROOT, _P["paths"]["log_dir"])
W_QUALITY         = _P["reward"]["w_quality"]
W_EFFICIENCY      = _P["reward"]["w_efficiency"]
METRICS_PATH      = os.path.join(_RL_ROOT, _P["paths"]["metrics"])
# ──────────────────────────────────────────────────────────────────────────────


def get_optimal_action(row):
    confs  = [row['n_conf'],  row['s_conf'],  row['l_conf']]
    lats   = [row['n_time'],  row['s_time'],  row['l_time']]
    counts = [row['n_count'], row['s_count'], row['l_count']]
    quality    = [confs[a] * np.sqrt(counts[a] + 1) for a in range(3)]
    efficiency = [quality[a] / (lats[a] + 1e-8)     for a in range(3)]
    q_max   = max(quality)    + 1e-8
    eff_max = max(efficiency) + 1e-8
    scores  = [W_QUALITY * (quality[a] / q_max) + W_EFFICIENCY * (efficiency[a] / eff_max) for a in range(3)]
    return int(np.argmax(scores))


def build_dataset(df, n_samples=SAMPLE_ROWS):
    extractor = FeatureExtractor()
    sampled   = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    obs_list    = []
    action_list = []
    missing     = 0

    print(f"Extracting observations from {len(sampled)} images …")
    for i, row in sampled.iterrows():
        if i % 2000 == 0:
            print(f"  {i}/{len(sampled)}")

        if os.path.exists(row['path']):
            frame    = cv2.imread(row['path'])
            vis_feats = extractor.get_visual_features(frame).flatten()   # 1024
            edge_val  = float(np.atleast_1d(extractor.get_edge_density(frame))[0])
        else:
            vis_feats = np.zeros(1024, dtype=np.float32)
            edge_val  = 0.0
            missing += 1

        scaled_edge = np.array([edge_val * 10.0], dtype=np.float32)
        metadata    = np.array([0.0, 0.5, 0.0], dtype=np.float32)   # neutral prior
        obs = np.concatenate([vis_feats, scaled_edge, metadata]).astype(np.float32)

        obs_list.append(obs)
        action_list.append(get_optimal_action(row))

    print(f"Done. Missing images: {missing}/{len(sampled)}")
    return np.stack(obs_list), np.array(action_list, dtype=np.int64)


class BCPolicy(nn.Module):
    """Mirrors the MlpPolicy architecture SB3 uses by default."""
    def __init__(self, obs_dim=1028, n_actions=3, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_bc(obs, actions):
    X = torch.FloatTensor(obs)
    y = torch.LongTensor(actions)

    dataset    = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BC_BATCH, shuffle=True)

    model   = BCPolicy()
    opt     = torch.optim.Adam(model.parameters(), lr=BC_LR)
    crit    = nn.CrossEntropyLoss()
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    print(f"\nTraining BC classifier for {BC_EPOCHS} epochs …")
    for epoch in range(BC_EPOCHS):
        model.train()
        total_loss = 0
        correct    = 0
        for xb, yb in dataloader:
            logits = model(xb)
            loss   = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(dim=1) == yb).sum().item()

        sched.step()
        if epoch % 5 == 0 or epoch == BC_EPOCHS - 1:
            acc = correct / len(dataset) * 100
            print(f"  Epoch {epoch+1:3d}/{BC_EPOCHS}  loss={total_loss/len(dataset):.4f}  acc={acc:.1f}%")

    # Action distribution on training set
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1).numpy()
    counts = np.bincount(preds, minlength=3)
    print(f"\nBC prediction distribution:")
    print(f"  Nano:  {counts[0]} ({counts[0]/len(preds)*100:.1f}%)")
    print(f"  Small: {counts[1]} ({counts[1]/len(preds)*100:.1f}%)")
    print(f"  Large: {counts[2]} ({counts[2]/len(preds)*100:.1f}%)")
    return model


def inject_bc_weights(bc_model: BCPolicy, ppo: PPO):
    """
    Copy BC MLP weights into the PPO MlpPolicy action_net + mlp_extractor.

    SB3 MlpPolicy layout (default net_arch=[256, 256]):
      policy.mlp_extractor.policy_net  : Linear(obs, 256) → Tanh → Linear(256, 256) → Tanh
      policy.action_net                : Linear(256, n_actions)

    Our BCPolicy layout:
      net[0] Linear(obs, 256) → Tanh
      net[2] Linear(256, 256) → Tanh
      net[4] Linear(256, n_actions)
    """
    src = bc_model.net
    tgt = ppo.policy

    with torch.no_grad():
        # Layer 0 → policy_net[0]
        tgt.mlp_extractor.policy_net[0].weight.copy_(src[0].weight)
        tgt.mlp_extractor.policy_net[0].bias.copy_(src[0].bias)
        # Layer 2 → policy_net[2]
        tgt.mlp_extractor.policy_net[2].weight.copy_(src[2].weight)
        tgt.mlp_extractor.policy_net[2].bias.copy_(src[2].bias)
        # Layer 4 → action_net
        tgt.action_net.weight.copy_(src[4].weight)
        tgt.action_net.bias.copy_(src[4].bias)

    print("BC weights successfully injected into PPO policy.")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    # ── Step 1: Build BC dataset ──────────────────────────────────────────────
    df  = pd.read_csv(CSV_PATH)
    obs, actions = build_dataset(df)

    label_counts = np.bincount(actions, minlength=3)
    print(f"\nGround-truth label distribution:")
    print(f"  Nano:  {label_counts[0]} ({label_counts[0]/len(actions)*100:.1f}%)")
    print(f"  Small: {label_counts[1]} ({label_counts[1]/len(actions)*100:.1f}%)")
    print(f"  Large: {label_counts[2]} ({label_counts[2]/len(actions)*100:.1f}%)")

    # ── Step 2: Train BC classifier ───────────────────────────────────────────
    bc_model = train_bc(obs, actions)

    # ── Step 3: Create PPO and inject weights ─────────────────────────────────
    print("\nBuilding PPO shell …")
    raw_env = AdaptiveInferenceEnv(csv_path=CSV_PATH)
    env     = Monitor(raw_env, LOG_DIR)
    env     = DummyVecEnv([lambda: env])

    ppo = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=_P["ppo"]["learning_rate"],
        n_steps=_P["ppo"]["n_steps"],
        batch_size=_P["ppo"]["batch_size"],
        ent_coef=_P["ppo"]["ent_coef"],
        gamma=0.99,
        device=_P["ppo"]["device"],
        policy_kwargs=dict(net_arch=[HIDDEN_DIM, HIDDEN_DIM]),
    )

    inject_bc_weights(bc_model, ppo)

    # Quick sanity check: run 500 deterministic steps
    env2    = AdaptiveInferenceEnv(csv_path=CSV_PATH)
    obs_env, _ = env2.reset()
    acts    = []
    for _ in range(500):
        act, _ = ppo.predict(obs_env, deterministic=True)
        obs_env, _, done, _, _ = env2.step(act)
        acts.append(int(act))
        if done:
            obs_env, _ = env2.reset()

    counts = np.bincount(acts, minlength=3)
    print(f"\nPost-BC-inject policy (500 steps, deterministic):")
    print(f"  Nano:  {counts[0]} ({counts[0]/5:.1f}%)")
    print(f"  Small: {counts[1]} ({counts[1]/5:.1f}%)")
    print(f"  Large: {counts[2]} ({counts[2]/5:.1f}%)")

    # ── Step 4: Fine-tune with PPO ────────────────────────────────────────────
    print(f"\nFine-tuning with PPO ({RL_FINETUNE_STEPS:,} steps) …")
    ppo.set_env(env)
    ppo.learn(total_timesteps=RL_FINETUNE_STEPS, progress_bar=True)

    out_path = os.path.join(MODELS_DIR, "final_adaptive_model.zip")
    ppo.save(out_path)
    print(f"\nSaved to {out_path}")

    # Final evaluation
    env3    = AdaptiveInferenceEnv(csv_path=CSV_PATH)
    obs3, _ = env3.reset()
    acts3   = []
    rewards3 = []
    for _ in range(1000):
        act, _ = ppo.predict(obs3, deterministic=True)
        obs3, rew, done, _, _ = env3.step(act)
        acts3.append(int(act))
        rewards3.append(rew)
        if done:
            obs3, _ = env3.reset()

    counts3 = np.bincount(acts3, minlength=3)
    avg_rew = float(np.mean(rewards3))
    print(f"\n=== FINAL EVALUATION (1000 steps) ===")
    print(f"  Nano:  {counts3[0]} ({counts3[0]/10:.1f}%)")
    print(f"  Small: {counts3[1]} ({counts3[1]/10:.1f}%)")
    print(f"  Large: {counts3[2]} ({counts3[2]/10:.1f}%)")
    print(f"  Avg Reward: {avg_rew:.4f}")

    # ── Write DVC metrics ─────────────────────────────────────────────────────
    import json
    metrics = {
        "avg_reward":   round(avg_rew, 4),
        "pct_nano":     round(counts3[0] / 10, 1),
        "pct_small":    round(counts3[1] / 10, 1),
        "pct_large":    round(counts3[2] / 10, 1),
        "bc_accuracy":  None,   # populated during train_bc if needed
    }
    with open(METRICS_PATH, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
