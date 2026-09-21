"""Fix the M1.3 metric: key on the DIVE, not the completion.

Audit finding: the counter dives at t=0.02 (inside the blocker's opening
transient) and completes at t*=6-10 s. My lookback window around t* therefore
missed the exploitation entirely and measured noise.

Correct operationalisation of "did the policy exploit the transient":
  find the follower's dive onset (its lane-change command toward inner while
  still behind), and test whether the BLOCKER was physically transitioning
  at that moment.

Measure the counter's rate under this definition -- that becomes the anchor.
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
SCRIPTS = f"{REPO}/results/20260920_phaseC_probe/scripts"
while "/tmp" in sys.path:
    sys.path.remove("/tmp")
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)

import numpy as np
from _blocker_v3 import BlockerWorld3
from _selfplay_design_probe import build_paths


def counter(w):
    g = w.gap_forward()
    d = w.delta_true()
    lane_f, lane_b = w.lane[0], w.lane[1]
    sw_f = w.switch[0]
    if d > 0.30:
        return np.array([0.0, -1.0 if lane_f == 1 else 0.0])
    if sw_f > 0:
        return np.array([0.8 if lane_f == 1 else 0.0, 0.0])
    if lane_f == 1:
        if lane_b == 0:
            return np.array([1.0, 0.0])
        return np.array([-0.4, -1.0])
    if lane_b == 1:
        return np.array([-0.7, 0.0] if g < 1.10 else [0.0, 0.0])
    if g < 1.40:
        return np.array([0.8, 1.0])
    return np.array([0.5, 0.0])


def audit(seed0, n_ep=30):
    """Return (dives_total, dives_inside_blocker_transient, completions)."""
    outer, inner = build_paths()
    dives = hits = comps = 0
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(seed0 + k))
        prev_lane = w.lane[0]
        was_ahead = False
        while True:
            # detect the follower's dive: inner while still behind
            a = counter(w)
            done = w.step(a)
            # after stepping, lane may have changed
            if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                dives += 1
                if w.switch[1] > 0.0:
                    hits += 1
            prev_lane = w.lane[0]
            if w.delta_true() > 0.30 and not was_ahead:
                comps += 1
                was_ahead = True
            if done:
                break
    return dives, hits, comps


def main():
    print("=" * 72)
    print("M1.3 re-anchored: dive-based definition")
    print("  dive = follower enters inner lane while still behind")
    print("  hit  = blocker physically transitioning at that moment")
    print("=" * 72)
    td = th = 0
    for s in (123000, 200000, 300000):
        d, h, c = audit(s)
        td += d
        th += h
        print(f"  seed0={s}: dives={d} hits={h} "
              f"({100*h/d if d else float('nan'):.0f}%)  completions={c}")
    print()
    print(f"  OVERALL: {th}/{td} = "
          f"{100*th/td if td else float('nan'):.0f}% of dives land inside a "
          f"blocker transient")
    if td:
        r = th / td
        print(f"  anchor: positive rate = {r:.2f}; 70% of it = {0.7*r:.2f}")
        print(f"  => proposed M1.3 threshold: >= {0.7*r:.2f}")


if __name__ == "__main__":
    main()
