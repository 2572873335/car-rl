"""manual_bc_h7.py - H7: locate manual-BC's failure mechanism.

H6 established manual-BC is implementation-limited (0-2/10 regardless of demo
count) while d3rlpy's BC hits 10/10 from 50 demos.  H7 tests the leading
candidate: the LOSS FUNCTION.

  manual-BC (train_ot.pretrain) maximizes Gaussian log-likelihood over a
  stochastic policy head -> learns mean AND variance.
  d3rlpy BC on continuous actions defaults to deterministic MSE regression.

For bang-bang demonstrations (full-lane-change / full-commit actions), a
Gaussian mean can average BETWEEN modes and emit mushy mid-actions -> collisions.

Variants (each a single d=300 retrain; TIME-BOXED, no hyperparameter search):
  mse : replace Gaussian NLL with deterministic MSE on the policy mean
  lr  : keep NLL but lower Adam lr (only if mse fails)

Usage (main venv):
    uv run python manual_bc_h7.py --variant mse
    uv run python manual_bc_h7.py --variant lr
"""
import argparse
import os
import numpy as np
import torch
from stable_baselines3 import PPO

from overtake_env import OvertakeEnv
from manual_bc_sweep import load_subset, eval_manual_bc, CKPT_DIR

N_DEMOS = 300


def train_variant(obs, act, variant, epochs=10, seed=0):
    """manual-BC architecture identical to pretrain; only the loss (or lr)
    differs.  One variable at a time."""
    dev = "cpu"
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    model = PPO("MlpPolicy", OvertakeEnv(domain_randomize=True),
                policy_kwargs=dict(net_arch=[128, 128]),
                seed=seed, verbose=0, device=dev)
    lr = 1e-3 if variant == "mse" else 5e-4      # 'lr' variant halves lr
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(obs_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for b in range(0, n, 8192):
            idx = perm[b:b + 8192]
            if variant == "mse":
                # deterministic MSE on the policy mean (continuous BC default)
                mean = model.policy.get_distribution(obs_t[idx]).distribution.mean
                loss = torch.nn.functional.mse_loss(mean, act_t[idx])
            else:
                # original: Gaussian NLL
                dist = model.policy.get_distribution(obs_t[idx])
                loss = -dist.log_prob(act_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        print(f"  [{variant}] epoch {ep+1}/{epochs}  loss={tot/cnt:.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["mse", "lr"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)

    obs, act = load_subset(N_DEMOS)
    print(f"H7 variant={args.variant}: d={N_DEMOS}, {len(obs)} transitions")
    model = train_variant(obs, act, args.variant, epochs=args.epochs)
    path = os.path.join(CKPT_DIR, f"manualbc_d{N_DEMOS}_{args.variant}.zip")
    model.save(path)

    # deterministic eval (MSE-trained head has no useful stochasticity anyway)
    rows = eval_manual_bc(model)
    succ = sum(r["overtaken"] for r in rows)
    col = sum(r["reason"] == "collision" for r in rows)
    t_ots = [r["t_ot"] for r in rows if r["overtaken"]]
    print(f"H7RESULT variant={args.variant} d={N_DEMOS}: success={succ}/10  "
          f"collision={col}  t_ot={np.mean(t_ots) if t_ots else float('nan'):.1f}s  "
          f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
