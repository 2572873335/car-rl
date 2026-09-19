"""Parse a train_*.py stdout log -> real training curves (Phase 4 figures).

Usage:  uv run python make_curves.py <log> <out.png> [title]
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ("total_timesteps", "ep_rew_mean", "ep_len_mean")


def parse(path):
    """Return dict of series from sb3's tabular stdout blocks.

    Only blocks carrying ALL of KEYS are kept (early blocks may lack
    ep_rew_mean/ep_len_mean before the first episode finishes).
    """
    rows, cur = [], {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = re.match(r"^\|\s+([a-z_/]+)\s+\|\s+([-+0-9.eE]+)\s+\|$", line.rstrip())
            if not m:
                if line.strip().startswith("---") and cur:
                    rows.append(cur)
                    cur = {}
                continue
            key = m.group(1).split("/")[-1]
            if key in KEYS:
                try:
                    cur[key] = float(m.group(2))
                except ValueError:
                    pass
    if cur:
        rows.append(cur)
    rows = [r for r in rows if all(k in r for k in KEYS)]
    return {k: [r[k] for r in rows] for k in KEYS}


def main():
    log, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "training curves"
    d = parse(log)
    x = [t / 1e6 for t in d["total_timesteps"]]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, d["ep_rew_mean"], color="tab:blue", lw=1.1, label="ep_rew_mean")
    ax1.set_xlabel("timesteps [M]")
    ax1.set_ylabel("ep_rew_mean", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, d["ep_len_mean"], color="tab:orange", lw=1.1, label="ep_len_mean")
    ax2.set_ylabel("ep_len_mean", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"parsed {len(x)} points -> {out}")
    print(f"  final ep_rew_mean = {d['ep_rew_mean'][-1]:.1f}")
    print(f"  ep_len_mean: first={d['ep_len_mean'][0]:.0f} last={d['ep_len_mean'][-1]:.0f} "
          f"min={min(d['ep_len_mean']):.0f} max={max(d['ep_len_mean']):.0f}")


if __name__ == "__main__":
    main()
