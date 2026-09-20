"""plot_rq2_efficiency.py - Figure for report 4.8.2 (data-efficiency curves).

Reads the sweep numbers (hard-coded from the archived raw logs, each value
traceable to results/20260920_rq2_offline/data_efficiency_sweep.txt) and plots
success rate vs demo episodes for four methods.

Usage (main venv): uv run python plot_rq2_efficiency.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# N episodes on x-axis
N = [50, 100, 200, 300]
# success counts /10 (from data_efficiency_sweep.txt + manualbc_sweep.txt)
SERIES = {
    "manual-BC (paper impl.)": [0, 0, 0, 2],
    "d3rlpy-BC":               [10, 10, 10, 10],
    "IQL (offline RL)":        [10, 10, 10, 10],
    "TD3+BC (offline RL)":     [2, 10, 10, 2],
}


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "manual-BC (paper impl.)": dict(color="tab:red", marker="s", ls="--"),
        "d3rlpy-BC":               dict(color="tab:gray", marker="^", ls=":"),
        "IQL (offline RL)":        dict(color="tab:blue", marker="o"),
        "TD3+BC (offline RL)":     dict(color="tab:orange", marker="D"),
    }
    for name, ys in SERIES.items():
        ax.plot(N, ys, label=name, lw=1.6, ms=7, **styles[name])
    ax.axhline(10, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("number of demonstration episodes")
    ax.set_ylabel("overtake success [episodes / 10]")
    ax.set_title("RQ2 data efficiency: success vs demo count (nested subsets)")
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks(N)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = "results/20260920_rq2_offline/fig_rq2_data_efficiency.png"
    fig.savefig(out, dpi=120)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
