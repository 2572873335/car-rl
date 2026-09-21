"""Simplest possible positive control: does the RULE MACHINE itself overtake
in criterion world v2?

The rule machine (baseline_action_ot) was hand-written to overtake a slow
leader -- exactly this world's task. If it succeeds here, it IS a valid
positive pole and no training is needed at all (the in-world PPO failed with
0% overtake -- the project's documented "exploration valley", which is why
train_ot.py needs a BC warm start).

Run it as the FOLLOWER against the scripted cruising leader.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _criterion_v2 import WorldV2, OVERTAKE_EPS
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot


def run_follower(pol, n_ep=40, seed0=123000):
    outer, inner = build_paths()
    deltas, times, tasks, colls, reasons = [], [], [], 0, {}
    for k in range(n_ep):
        w = WorldV2(outer, inner, prior="v2v")
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0 = pol(obs[0], w.L)
            obs, done = w.step(np.stack([a0, np.zeros(2)]))
            if done:
                break
        deltas.append(w.m_delta())
        times.append(w.m_time())
        tasks.append(w.m_task())
        reasons[w.reason] = reasons.get(w.reason, 0) + 1
        if w.reason == "collision":
            colls += 1
    return (np.array(deltas), np.array(times), np.array(tasks),
            colls / n_ep, reasons)


def main():
    # The rule machine speaks the FROZEN obs layout, so adapt our obs into it.
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))

    d, t, k, c, rs = run_follower(rule)
    fin = [x for x in t if np.isfinite(x)]
    print("=" * 72)
    print("RULE MACHINE as follower in criterion world v2 (40 episodes)")
    print("=" * 72)
    print(f"  mean_delta      = {d.mean():+.3f}   (uniq {len(set(np.round(d,6)))})")
    print(f"  overtake rate   = {len(fin)/len(t):.2f}")
    print(f"  mean t_overtake = {np.mean(fin) if fin else float('nan'):.2f} s")
    print(f"  task success    = {k.mean():.2f}")
    print(f"  collisions      = {c:.2f}")
    print(f"  reasons         = {rs}")
    print()
    if len(fin) / len(t) >= 0.5:
        print("  => The rule machine IS a valid positive pole for this world.")
        print("     A criterion can be validated without training anything.")
    else:
        print("  => The rule machine does not overtake here either; finding a")
        print("     positive pole needs the BC warm start (train_ot.py's recipe).")


if __name__ == "__main__":
    main()
