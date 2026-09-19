"""robustness_sweep.py - Phase A W1 (A1/A2): continuous robustness scan.

Evaluates RL (follow_stage2_final) vs rule P+FF under actuator degradation
epsilon in {0,10,20,30,40,50}%:  tau *= (1+eps)  (slower motor), a_max *= (1-eps)
(weaker accel).  Main metric = collision-free rate (R2); mean|e| reported only
on collision-free episodes, using the SAME settled definition as train_ppo.evaluate
(interior 80%, R4-2).  Follows the reviewed plan (review1, 2026-09-20).

Design (per plan):
  - Wrapper mutates ONLY env.follower.tau/a_max post-reset (R1); leader untouched.
  - eps=0 single-arg gate (R3) must reproduce the archived nominal result
    (RL 0.93 cm / P+FF 1.93 cm) before the full sweep runs.

Usage:
    uv run python robustness_sweep.py --eps-list 0 10 20 30 40 50
    uv run python robustness_sweep.py --only-0          # R3 pilot gate
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from follow_env import FollowEnv, baseline_action, D_DES

TAU0, AMAX0 = 0.12, 1.5          # nominal actuator params (FollowEnv, dr=False)
EPS_LIST = [0, 10, 20, 30, 40, 50]
# Two blocks of 10: 1000-1009 (paper protocol) + 2000-2009 (independent confirm).
# 20 seeds because 10 proved too few -- at eps=50% the first block showed 3/10
# collisions while a fresh block showed 6/10; merged is the honest estimate.
DEFAULT_SEEDS = list(range(1000, 1010)) + list(range(2000, 2010))
MODEL = "ckpt/follow_stage2_final_v1.zip"


class RobustnessWrapper:
    """Pin follower actuator params to degraded regime; leader stays nominal.
    Not a gym.Wrapper subclass -- plain facade so we control reset precisely
    and keep the frozen env file untouched (iron rule 2 / sha256)."""

    def __init__(self, eps):
        """eps: degradation FRACTION (0..0.5). 0.1 -> tau*1.1, a_max*0.9.
        A percent value (e.g. 10) is a units bug: a_max would go negative and
        the vehicle would lose its actuator limits entirely -- guard loudly."""
        if not (0.0 <= eps <= 0.5):
            raise ValueError(
                f"eps must be a fraction in [0, 0.5], got {eps!r} "
                f"(did you pass percent instead of fraction?)")
        self.eps = eps
        self.env = None

    def make(self):
        e = self.env = FollowEnv(domain_randomize=False)   # nominal params
        return e

    def reset(self, seed):
        obs, info = self.env.reset(seed=seed)              # nominal reset
        e = self.eps
        if e:
            # degrade ONLY the follower (R1); leader untouched
            self.env.follower.tau = TAU0 * (1.0 + e)
            self.env.follower.a_max = AMAX0 * (1.0 - e)
        return obs, info

    def step(self, action):
        # wrapper holds no terminal logic; delegate fully
        return self.env.step(action)


def settled_mean_abs_e(es):
    """Reuse train_ppo.evaluate's settled definition: interior 80%."""
    e = np.asarray(es, dtype=float)
    if len(e) == 0:
        return float("nan")
    return np.abs(e[int(len(e) * 0.2):]).mean() * 100.0   # cm


def roll_eval(policy, eps, seeds=None):
    """Evaluate one policy at one eps level over fixed seeds.
    Returns per-seed rows with full termination audit (iron rule 5)."""
    if seeds is None:
        seeds = DEFAULT_SEEDS
    rows = []
    for s in seeds:
        w = RobustnessWrapper(eps)
        env = w.make()
        obs, _ = w.reset(s)
        es = []
        while True:
            obs, _, term, trunc, _ = w.step(policy(obs))
            es.append(obs[0])
            if term or trunc:
                break
        rows.append(dict(
            seed=s, behavior=env.behavior, gap0=env.gap0,
            steps=len(es), reason=env.term_reason,
            mean_abs_e=settled_mean_abs_e(es),
        ))
    return rows


def summarize(rows):
    n = len(rows)
    col = sum(r["reason"] == "collision" for r in rows)
    lost = sum(r["reason"] == "lost" for r in rows)
    off = sum(r["reason"] == "offtrack" for r in rows)
    ok = [r for r in rows if r["reason"] == "timeout"]   # completed w/o termination
    cf_rate = (n - col) / n                              # collision-free
    mean_e = np.mean([r["mean_abs_e"] for r in ok]) if ok else float("nan")
    return dict(col=col, lost=lost, off=off, cf_rate=cf_rate,
                mean_e=mean_e, n_complete=len(ok))


def get_policies():
    from stable_baselines3 import PPO
    rl = PPO.load(MODEL)
    policies = {
        "RL(PPO)": lambda o: rl.predict(o, deterministic=True)[0],
        "rule P+FF": lambda o: baseline_action(o, use_ff=True),
    }
    return policies


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-list", type=int, nargs="*", default=EPS_LIST)
    ap.add_argument("--only-0", action="store_true", help="R3 pilot gate: eps=0 only")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--outdir", default="results/20260926_robustness_sweep")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="seed list; default = 2 blocks of 10 (see DEFAULT_SEEDS)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    MODEL = args.model
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS

    eps_list = [0] if args.only_0 else args.eps_list
    os.makedirs(args.outdir, exist_ok=True)

    policies = get_policies()

    # results[policy][eps_pct] = summary
    results = {}
    n = len(seeds)
    for pname, pol in policies.items():
        results[pname] = {}
        for eps_pct in eps_list:
            eps = eps_pct / 100.0          # percent -> fraction (0.1 = 10%)
            rows = roll_eval(pol, eps, seeds=seeds)
            s = summarize(rows)
            results[pname][eps_pct] = s
            print(f"{pname:10s} eps={eps_pct:3d}%  collision-free="
                  f"{s['cf_rate']*n:5.1f}/{n}  "
                  f"collision={s['col']} lost={s['lost']} offtrack={s['off']}  "
                  f"mean|e|(complete,cm)={s['mean_e']:6.2f}", flush=True)

    # ---- CSV ----
    csv_path = os.path.join(args.outdir, "robustness_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("policy,eps_pct,tau,a_max,n_seeds,collision_free,n_collision,"
                "n_lost,n_offtrack,mean_abs_e_cm,n_complete\n")
        for pname in results:
            for eps, s in results[pname].items():
                f.write(f"{pname},{eps},{TAU0*(1+eps/100):.4f},{AMAX0*(1-eps/100):.4f},"
                        f"{n},{round(s['cf_rate']*n)},{s['col']},{s['lost']},{s['off']},"
                        f"{s['mean_e']:.3f},{s['n_complete']}\n")
    print(f"\ncsv -> {csv_path}")

    if not args.no_plot:
        plot(results, eps_list, args.outdir, n_seeds=n)
    print("done.")


def plot(results, eps_list, outdir, n_seeds=10):
    x = [e / 100.0 for e in eps_list]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # main: collision-free rate
    for pname, d in results.items():
        y = [d[e]["cf_rate"] * n_seeds for e in eps_list]
        ax1.plot(x, y, marker="o", label=pname)
    ax1.axvline(1 / 3.0, color="k", ls="--", lw=1)
    ax1.text(1 / 3.0, 0.1, "  DR boundary\n  (a_max=1.0)",
             fontsize=7, va="bottom")
    ax1.set_ylabel(f"collision-free [episodes/{n_seeds}]")
    ax1.set_ylim(-0.3, n_seeds + 0.3)
    ax1.set_title("follow robustness vs actuator degradation (leader nominal)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # secondary: mean |e| on complete episodes only
    for pname, d in results.items():
        y = [d[e]["mean_e"] for e in eps_list]
        ax2.plot(x, y, marker="o", label=pname)
    ax2.axvline(1 / 3.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("eps (degradation)  [0.1 = tau*1.1, a_max*0.9]")
    ax2.set_ylabel("mean|e| [cm] (complete eps only)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(outdir, "fig_robustness_curve.png")
    fig.savefig(out, dpi=120)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
