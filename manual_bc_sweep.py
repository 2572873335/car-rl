"""manual_bc_sweep.py - H6 test: paper manual-BC over nested demo subsets.

WHY: the data-efficiency sweep exposed TWO different "BC"s:
  - paper manual-BC (train_ot.pretrain: MLP 128x128, Adam lr=1e-3, 10 epochs,
    batch 8192, max-log-likelihood on PPO policy head) -> 2/10 at d=300
  - d3rlpy BCConfig -> 10/10 already at d=50
Mixing them kills the causal reading of the data-efficiency curve.  Per review
decision (option 1), rerun manual-BC on the SAME nested subsets so the ONLY
variable is N.  H6 (pre-registered): manual-BC success rate does NOT improve
with demo count (max-min <= 2/10 across 50/100/200/300).

Implementation: byte-for-byte the pretrain hyperparams -- hyperparams are LOCKED,
only the data subset changes.  Trains with the MAIN venv (sb3+torch), reads
demos_v1.npz read-only, writes models to ckpt_offline/manualbc_d<N>.zip.

Usage (main venv):
    uv run python manual_bc_sweep.py --n-list 50 100 200 300
"""
import argparse
import os
import numpy as np
import torch
from stable_baselines3 import PPO

from overtake_env import OvertakeEnv   # only to construct the policy net, like pretrain

CKPT_DIR = "ckpt_offline"
DEMOS = "demos_v1.npz"
EVAL_SEEDS = list(range(2000, 2010))   # same eval protocol as everything else


def load_subset(max_episodes):
    """Nested subset: first N episodes of demos_v1.npz (same base seeds)."""
    d = np.load(DEMOS, allow_pickle=True)
    obs, act = d["observations"], d["actions"]
    ends = d["episode_ends"]
    last = int(ends[max_episodes - 1])
    return obs[: last + 1], act[: last + 1]


def train_manual_bc(obs, act, epochs=10, seed=0):
    """Exact pretrain loop: MLP 128x128, Adam lr 1e-3, batch 8192, MLE loss."""
    dev = "cpu"
    obs_t = torch.as_tensor(obs, dtype=torch.float32).to(dev)
    act_t = torch.as_tensor(act, dtype=torch.float32).to(dev)
    model = PPO("MlpPolicy", OvertakeEnv(domain_randomize=True),
                policy_kwargs=dict(net_arch=[128, 128]),
                seed=seed, verbose=0, device=dev)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    n = len(obs_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for b in range(0, n, 8192):
            idx = perm[b:b + 8192]
            dist = model.policy.get_distribution(obs_t[idx])
            loss = -dist.log_prob(act_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        print(f"  BC epoch {ep+1}/{epochs}  loss={tot/cnt:.4f}", flush=True)
    return model


def eval_manual_bc(model, seeds=EVAL_SEEDS):
    from overtake_env import OvertakeEnv
    rows = []
    for s in seeds:
        env = OvertakeEnv(domain_randomize=False)
        obs, _ = env.reset(seed=int(s))
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
        lg = env.log
        t = np.array(lg["t"]); delta = np.array(lg["delta"])
        ahead = t[delta > 0.05]
        rows.append(dict(seed=s, reason=env.term_reason,
                         overtaken=len(ahead) > 0,
                         t_ot=ahead[0] if len(ahead) else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="*", default=[50, 100, 200, 300])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)

    for n in args.n_list:
        obs, act = load_subset(n)
        print(f"manual-BC d={n}: {len(obs)} transitions (nested subset)", flush=True)
        model = train_manual_bc(obs, act, epochs=args.epochs)
        path = os.path.join(CKPT_DIR, f"manualbc_d{n}.zip")
        model.save(path)
        rows = eval_manual_bc(model)
        succ = sum(r["overtaken"] for r in rows)
        col = sum(r["reason"] == "collision" for r in rows)
        t_ots = [r["t_ot"] for r in rows if r["overtaken"]]
        print(f"MANUALBC d={n:3d}: success={succ}/10  collision={col}  "
              f"t_ot={np.mean(t_ots) if t_ots else float('nan'):.1f}s  -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
