"""export_demos.py - collect rule-based demonstrations for RQ2 (offline RL).

Writes a full transition dataset (5-tuple) that d3rlpy's MDPDataset can load
directly, plus per-episode metadata for auditing.

WHY THIS EXISTS (vs train_ot.py pretrain):
  pretrain() keeps only (obs, action) -- enough for behaviour cloning, but
  IQL/CQL also need reward and next_obs.  So this script re-runs the same
  collection loop and records the complete transition.

DESIGN NOTES
  * Zero new deps: numpy only.  d3rlpy pins gymnasium==1.0.0 while this
    project runs 1.3.0, so installing it here would DOWNGRADE the frozen
    environment.  Loading into d3rlpy happens later, in a separate venv.
  * Terminal-vs-timeout: env reports success/timeout both as `truncated`,
    so we classify by env.term_reason instead of the truncation flag:
        collision / offtrack / lost  -> true terminal (bad outcome)
        success                      -> true terminal (goal reached)
        timeout / failed             -> time-limit truncation
  * Seed base 30_000+i matches train_ot.py pretrain, so the dataset is
    directly comparable with the existing BC baseline (2/10).

Usage:
    uv run python export_demos.py                      # 300 demos -> demos_v1.npz
    uv run python export_demos.py --n-demos 300 --out demos_v1.npz
"""
import argparse
import hashlib
import os

import numpy as np

from overtake_env import OvertakeEnv, baseline_action_ot

DEFAULT_SEED_BASE = 30_000          # same family as train_ot.py --n-demos
TRUE_TERMINAL_REASONS = {"collision", "offtrack", "lost", "success"}
TIMEOUT_REASONS = {"timeout", "failed"}


def collect(n_demos, seed_base=DEFAULT_SEED_BASE, verbose=True):
    """Roll out the rule state machine, recording full transitions."""
    obs_buf, act_buf, rew_buf, nobs_buf = [], [], [], []
    done_buf, term_buf, timeout_buf = [], [], []
    ep_end_idx, ep_reason, ep_seed = [], [], []

    for i in range(n_demos):
        seed = seed_base + i
        env = OvertakeEnv(domain_randomize=True)
        obs, _ = env.reset(seed=seed)
        while True:
            a = baseline_action_ot(obs)
            next_obs, r, terminated, truncated, _ = env.step(a)
            obs_buf.append(obs)
            act_buf.append(a)
            rew_buf.append(r)
            nobs_buf.append(next_obs)
            done = bool(terminated or truncated)
            done_buf.append(done)
            term_buf.append(bool(terminated))
            timeout_buf.append(bool(truncated))
            obs = next_obs
            if done:
                break
        ep_end_idx.append(len(obs_buf) - 1)
        ep_reason.append(env.term_reason)
        ep_seed.append(seed)
        if verbose and (i + 1) % 50 == 0:
            print(f"  collected {i+1}/{n_demos} episodes, "
                  f"{len(obs_buf)} transitions", flush=True)

    data = {
        "observations": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(act_buf, dtype=np.float32),
        "rewards": np.asarray(rew_buf, dtype=np.float32),
        "next_observations": np.asarray(nobs_buf, dtype=np.float32),
        # d3rlpy MDPDataset fields
        "terminals": np.asarray(done_buf, dtype=bool),
        "timeouts": np.asarray(timeout_buf, dtype=bool),
        # per-episode audit trail
        "episode_ends": np.asarray(ep_end_idx, dtype=np.int64),
        "episode_reasons": np.asarray(ep_reason, dtype=object).astype("U16"),
        "episode_seeds": np.asarray(ep_seed, dtype=np.int64),
    }

    # d3rlpy wants episode_terminals: True only at a genuine MDP terminal
    # (not a time-limit truncation).
    ep_term = np.zeros(len(ep_end_idx), dtype=bool)
    for k, reason in enumerate(ep_reason):
        if reason in TRUE_TERMINAL_REASONS:
            ep_term[k] = True
    data["episode_terminals"] = np.zeros(len(done_buf), dtype=bool)
    for k, end in enumerate(ep_end_idx):
        data["episode_terminals"][end] = ep_term[k]
    return data


def summarize(data):
    n = len(data["rewards"])
    reasons = data["episode_reasons"]
    uniq, counts = np.unique(reasons, return_counts=True)
    print(f"\ndataset: {n} transitions, {len(data['episode_ends'])} episodes")
    print(f"  obs dim = {data['observations'].shape[1]}, "
          f"act dim = {data['actions'].shape[1]}")
    print(f"  rewards: mean={data['rewards'].mean():.3f} "
          f"min={data['rewards'].min():.1f} max={data['rewards'].max():.1f}")
    print("  episode outcomes:")
    for u, c in sorted(zip(uniq, counts), key=lambda x: -x[1]):
        print(f"    {str(u):10s} {c:4d}  ({100*c/len(reasons):.0f}%)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-demos", type=int, default=300)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--out", default="demos_v1.npz")
    args = ap.parse_args()

    print(f"collecting {args.n_demos} rule-based demos "
          f"(seed base {args.seed_base}) ...")
    data = collect(args.n_demos, args.seed_base)
    summarize(data)

    np.savez_compressed(args.out, **data)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}  ({size_mb:.2f} MB)")
    print(f"sha256 {sha256(args.out)}  {args.out}")
    print("\nadd the line above to DATA_MANAGEMENT.md section 8 (data assets).")


if __name__ == "__main__":
    main()
