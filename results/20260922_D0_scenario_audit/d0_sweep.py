"""D0 scenario-fidelity sweep -- the formal executing artifact for
plan_D0_scenario_audit_v3.md (review2: Approve with amendments).

Design (all per the plan, conventions declared):
  * grid: 9 (d,v) cells, d in {0.20,0.50,1.00} x v in {0.30,0.55,1.00},
    behavior=constant, PLUS the (0.50,1.00) sinusoid/brake variants
    (plan section 2.5: variants are extra rows, not counted in the 9).
  * policies: RL (frozen ckpt), P+FF, P -- identical interface.
  * 100 seeds per cell (the aggregation unit D0.2 is judged on).
  * conventions (plan section 2.0): report settled AND whole; obs-cm AND true-cm.
    The judged quantity is settled + obs-cm.
  * iron rule 5: all termination reasons reported, not just success.
  * iron rule 2: frozen file untouched -- D_DES overridden at runtime through the
    wrapper, with try/finally and a post-run assertion (review2 B3).

Outputs into results/20260922_D0_scenario_audit/:
  raw.json        every per-seed settled value + term reason
  summary.csv     per (cell, policy) aggregate
  config.txt      run configuration snapshot
  fig_D0_envelope_v1.png   colour = RL settled obs-cm, reversal cells marked

Run:  uv run python results/20260922_D0_scenario_audit/d0_sweep.py
"""
import hashlib
import io
import json
import os
import sys
import time

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

OUT = f"{REPO}/results/20260922_D0_scenario_audit"
os.makedirs(OUT, exist_ok=True)

CKPT = f"{REPO}/ckpt/follow_stage2_final_v1.zip"
N_SEED = 100
SEED0 = 2000

# (d, v, behavior, is_core) -- is_core marks the 9-cell (d,v) grid
CELLS = [
    (0.20, 0.30, "constant", True),
    (0.20, 0.55, "constant", True),
    (0.20, 1.00, "constant", True),
    (0.50, 0.30, "constant", True),
    (0.50, 0.55, "constant", True),
    (0.50, 1.00, "constant", True),
    (0.50, 1.00, "sinusoid", False),
    (0.50, 1.00, "brake", False),
    (1.00, 0.30, "constant", True),
    (1.00, 0.55, "constant", True),
    (1.00, 1.00, "constant", True),
]


class DDESWrapper:
    """Runtime D_DES override; frozen file untouched.

    close() MUST be reached even on exception (review2 B3): a stale module
    constant silently re-points every later cell's setpoint.
    """

    def __init__(self, d_des, v_set, behavior):
        self.d_des, self.v_set, self.behavior = d_des, v_set, behavior

    def make(self):
        self.env = fe.FollowEnv(domain_randomize=False)
        return self.env

    def reset(self, seed):
        fe.D_DES = self.d_des
        obs, info = self.env.reset(seed=seed)
        self.env.v_set = self.v_set
        self.env.behavior = self.behavior
        return obs, info

    def step(self, a):
        return self.env.step(a)

    def close(self):
        fe.D_DES = 0.20


def rollout(policy, d_des, v_set, behavior, n_ep, seed0):
    """One cell x one policy. Returns per-seed settled/whole obs-cm + reasons."""
    w = DDESWrapper(d_des, v_set, behavior)
    w.make()
    settled, whole, reasons = [], [], {}
    try:
        for k in range(n_ep):
            obs, _ = w.reset(seed0 + k)
            while True:
                obs, r, term, trunc, _ = w.step(policy(w.env, obs))
                if term or trunc:
                    break
            e = np.abs(np.array(w.env.log["gap"]) - d_des)
            settled.append(float(e[int(len(e) * 0.2):].mean() * 200.0))  # obs-cm
            whole.append(float(e.mean() * 200.0))
            rr = w.env.term_reason
            reasons[rr] = reasons.get(rr, 0) + 1
    finally:
        w.close()
        assert fe.D_DES == 0.20, f"D_DES leaked: {fe.D_DES}"
    return settled, whole, reasons


def main():
    from stable_baselines3 import PPO
    model = PPO.load(CKPT)

    policies = {
        "RL":    lambda env, obs: model.predict(obs, deterministic=True)[0],
        "P+FF":  lambda env, obs: fe.baseline_action(obs, use_ff=True),
        "P":     lambda env, obs: fe.baseline_action(obs, use_ff=False),
    }

    h0 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()
    with io.open(f"{OUT}/config.txt", "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {CKPT}\n")
        f.write(f"n_seed: {N_SEED}  seed0: {SEED0}\n")
        f.write("convention: settled = mean over e[20%:]; unit obs-cm = 200*|e[m]|\n")
        f.write("frozen follow_env.py sha256: " + h0 + "\n")
        f.write("cells: " + json.dumps(CELLS) + "\n")

    raw, summary = [], []
    t_start = time.time()
    print("=" * 108, flush=True)
    print(f"D0 SWEEP  ({len(CELLS)} cells x {len(policies)} policies x {N_SEED} seeds)", flush=True)
    print(f"  settled obs-cm (judged) | whole obs-cm | 5-class term audit", flush=True)
    print("=" * 108, flush=True)
    print(f"  {'cell':>24s} {'policy':>6s} {'mean':>7s} {'std':>6s} {'+/-SE':>6s} "
          f"{'whole':>7s} {'reasons':>38s}", flush=True)

    for d, v, b, core in CELLS:
        cell = f"d={d:.2f} v={v:.2f} {b[:7]}"
        for name, pol in policies.items():
            settled, whole, reasons = rollout(pol, d, v, b, N_SEED, SEED0)
            arr = np.array(settled)
            se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            print(f"  {cell:>24s} {name:>6s} {arr.mean():>7.3f} {arr.std(ddof=1):>6.3f} "
                  f"{se:>6.3f} {np.mean(whole):>7.3f} {str(reasons):>38s}", flush=True)
            raw.append(dict(d=d, v=v, behavior=b, core=core, policy=name,
                            settled=settled, whole=whole, reasons=reasons))
            summary.append(dict(d=d, v=v, behavior=b, core=core, policy=name,
                                mean=float(arr.mean()), std=float(arr.std(ddof=1)),
                                se=float(se), whole_mean=float(np.mean(whole)),
                                reasons=reasons))
        print("-" * 108, flush=True)

    with io.open(f"{OUT}/raw.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=1)

    # paired winner per cell (settled)
    by_cell = {}
    for s in summary:
        by_cell.setdefault((s["d"], s["v"], s["behavior"]), {})[s["policy"]] = s
    with io.open(f"{OUT}/summary.csv", "w", encoding="utf-8") as f:
        f.write("d,v,behavior,core,policy,mean_settled_obscm,std,se,whole_mean,reasons\n")
        for s in summary:
            f.write(f"{s['d']},{s['v']},{s['behavior']},{int(s['core'])},{s['policy']},"
                    f"{s['mean']:.4f},{s['std']:.4f},{s['se']:.4f},{s['whole_mean']:.4f},"
                    f"\"{json.dumps(s['reasons'])}\"\n")

    print(flush=True)
    print("  WINNERS (settled obs-cm), core 9-cell grid marked *", flush=True)
    for (d, v, b), pol in by_cell.items():
        core = next(s["core"] for s in summary
                    if (s["d"], s["v"], s["behavior"]) == (d, v, b))
        rl, pf = pol["RL"]["mean"], pol["P+FF"]["mean"]
        win = "RL" if rl < pf else "P+FF"
        star = "*" if core else " "
        print(f"   {star} d={d:.2f} v={v:.2f} {b[:8]:>8s}  RL={rl:6.3f}  P+FF={pf:6.3f}  -> {win}",
              flush=True)

    h1 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()
    print(flush=True)
    print(f"  frozen follow_env.py sha256 intact: {h0 == h1}", flush=True)
    print(f"  elapsed: {time.time() - t_start:.0f}s", flush=True)

    # ---------------- envelope plot ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dv_cells = sorted({(s["d"], s["v"]) for s in summary if s["core"]})
        ds = sorted({c[0] for c in dv_cells})
        vs = sorted({c[1] for c in dv_cells})
        grid = np.full((len(vs), len(ds)), np.nan)
        rev = np.zeros((len(vs), len(ds)), dtype=bool)
        for i, v in enumerate(vs):
            for j, d in enumerate(ds):
                cell = by_cell.get((d, v, "constant"))
                if not cell:
                    continue
                grid[i, j] = cell["RL"]["mean"]
                rev[i, j] = cell["RL"]["mean"] >= cell["P+FF"]["mean"]

        fig, ax = plt.subplots(figsize=(7.5, 5))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(ds)), [f"{d:.2f}" for d in ds])
        ax.set_yticks(range(len(vs)), [f"{v:.2f}" for v in vs])
        ax.set_xlabel("target gap d_des [m]")
        ax.set_ylabel("leader speed v [m/s]")
        ax.set_title("D0 envelope: RL settled gap error (obs-cm)\n"
                     "X = cells where RL loses to P+FF")
        for i in range(len(vs)):
            for j in range(len(ds)):
                if np.isnan(grid[i, j]):
                    continue
                lbl = f"{grid[i, j]:.1f}" + ("\nX" if rev[i, j] else "")
                ax.text(j, i, lbl, ha="center", va="center",
                        color="w" if grid[i, j] < np.nanmean(grid) else "k",
                        fontsize=9, fontweight="bold" if rev[i, j] else "normal")
        fig.colorbar(im, ax=ax, label="RL settled mean|e| [obs-cm]")
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_D0_envelope_v1.png", dpi=140)
        print(f"  figure -> {OUT}/fig_D0_envelope_v1.png", flush=True)
    except Exception as e:
        print(f"  figure FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
