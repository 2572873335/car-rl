"""Confirm the working counter config and its stability.

Finding: requiring the blocker SETTLED before diving gives 0.00, while diving
DURING the blocker's outer transition gives 0.70. That makes sense: while the
blocker's body is mid-move it is physically on neither lane centre, so it
cannot contest -- the window is the transient itself, not the settled cooldown.

Verify across independent seed sets with more episodes.
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


def run(seed0, n_ep=30, cruise_v=0.55, guard_hi=0.90):
    outer, inner = build_paths()
    comp, coll, ts = [], 0, []
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=cruise_v, guard_hi=guard_hi)
        w.reset(np.random.default_rng(seed0 + k))
        while True:
            if w.step(counter(w)):
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"))


def main():
    print("=" * 70)
    print("COUNTER STABILITY (cruise 0.55, guard 0.08-0.90, 30 eps)")
    print("=" * 70)
    for s in (123000, 200000, 300000, 400000, 500000):
        c, cl, t = run(s)
        print(f"  seed0={s}: complete={c:.2f} coll={cl:.0%} med_t={t:.2f}")


if __name__ == "__main__":
    main()
