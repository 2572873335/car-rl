"""Reviewer final numerics.

N1: Would the plan's W1 criterion PASS on the plan's OWN failed joint run?
    W1 clause: "对抗模式碰撞率不高于首 20%（相对 <= 1.2x）".
    Parse W5's run2 raw trend (bucket=100 episodes) and compute it.

N2: Fairness re-test of the rule machine with a CORRECT-gap adapter:
    do-nothing (car0) vs rule machine (car1), both adapters.

N3: Catastrophic-forgetting sanity: is 70/30 mixing even verifiable from
    the plan's W4 (only 10 seeds)?
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld, selfplay_to_frozen
from overtake_env import baseline_action_ot

RAW = "/home/zy/car_rl/code0919/results/20260920_phaseC_probe/probe_run2_joint_selfplay_trend_raw.txt"


def parse(path):
    rows = []
    for ln in open(path, errors="ignore"):
        if "Warning" in ln or "warn" in ln or "=" in ln or "steps" in ln:
            continue
        p = ln.split()
        if len(p) >= 3 and p[0].replace(",", "").isdigit() and p[2].endswith("%"):
            try:
                rows.append((int(p[0].replace(",", "")), float(p[1]),
                             float(p[2].rstrip("%"))))
            except ValueError:
                pass
    return rows


def N1():
    print("=" * 78)
    print("N1: does the plan's W1 PASS on the plan's OWN failed joint run?")
    print("=" * 78)
    rows = parse(RAW)
    steps = np.array([r[0] for r in rows])
    coll = np.array([r[2] for r in rows])
    smax = steps.max()
    first = coll[steps <= 0.2 * smax]
    last = coll[steps >= 0.8 * smax]
    print(f"  parsed {len(rows)} buckets, {steps.min():,} .. {steps.max():,} steps")
    print(f"  mean collision% over FIRST 20% of steps : {first.mean():.1f}%")
    print(f"  mean collision% over LAST  20% of steps : {last.mean():.1f}%")
    print(f"  ratio last/first                        : {last.mean()/first.mean():.2f}x")
    print(f"  W1 clause 'last <= 1.2 x first'         : "
          f"{'PASS' if last.mean() <= 1.2*first.mean() else 'FAIL'}")
    print("  NOTE: the plan labels this run 'P1 failed / not converging'.")
    # also the naive 'first bucket vs last bucket' reading
    print(f"  (naive reading: first bucket {coll[0]:.0f}% -> last bucket "
          f"{coll[-1]:.0f}%, ratio {coll[-1]/max(coll[0],1e-9):.2f}x)")


def N2():
    print()
    print("=" * 78)
    print("N2: do-nothing vs rule machine -- adapter fidelity (fairness check)")
    print("=" * 78)

    def hardcoded(o, L):
        return selfplay_to_frozen(o)

    def true_gap(o, L=5.484928):
        delta_n, v_self, v_other, e_lat_n, lane, clear_n = o
        delta = delta_n * 2.5
        gap = (-delta) % L
        return np.array([np.clip((gap - 0.2) / 0.5, -1.0, 6.0),
                         delta_n, v_self, e_lat_n, lane, v_other], np.float32)

    for label, mp in [("hardcoded gap=1.0", hardcoded), ("TRUE gap", true_gap)]:
        w = l = t = 0
        inner = 0
        for k in range(60):
            wd = NeutralWorld(*build_paths())
            obs = wd.reset(np.random.default_rng(20000 + k))
            ever = False
            while True:
                a0 = np.zeros(2)
                a1 = baseline_action_ot(mp(obs[1], wd.L))
                obs, _, done = wd.step(np.stack([a0, a1]))
                if wd.lane[1] == 1:
                    ever = True
                if done:
                    break
            if wd.delta(0) > 0.05:
                w += 1
            elif wd.delta(0) < -0.05:
                l += 1
            else:
                t += 1
            if ever:
                inner += 1
        print(f"  [{label:20s}] car0(do-nothing) W{w} L{l} T{t}"
              f"   car1 used inner {inner}/60")


def N3():
    print()
    print("=" * 78)
    print("N3: W4's 10-seed resolution (plan: task success >= 8/10)")
    print("=" * 78)
    print("  Binomial: if true p=0.90, P(>=8/10)=0.93 -> the 8/10 line almost")
    print("  never fires; if true p=0.70, P(>=8/10)=0.38.")
    for p in (0.7, 0.8, 0.9, 1.0):
        from math import comb
        pk = sum(comb(10, k) * p**k * (1-p)**(10-k) for k in range(8, 11))
        p10 = p**10
        print(f"    true p={p:.1f}: P(>=8/10)={pk:.2f}   P(==10/10)={p10:.2f}")
    print("  => a 10-seed 8/10 line cannot separate 'minor regression' from")
    print("     'no regression'.  Project's own H6 norm: boundary claims need")
    print("     30 seeds (ASSUMPTIONS.md H6: 10 seeds -> 30 seeds to resolve 2/10).")


if __name__ == "__main__":
    N1()
    N2()
    N3()
