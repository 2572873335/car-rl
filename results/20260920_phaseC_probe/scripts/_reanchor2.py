"""Re-anchor M1.3 under the CORRECTED definition.

Corrected hit definition:
    hit = the blocker is in the re-entry transient at the dive moment:
          lane_target == 1 AND lane == 0
(previous definition used switch[1] > 0, which also counted the outward move)

Report the counter's hit rate and the occupancy base rate for the same state,
so the threshold can be anchored as a lift.
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


def is_reentry(w):
    """Corrected target state: commanded inner, body still outer."""
    return (w.lane_target[1] == 1) and (w.lane[1] == 0)


def audit(seed0, n_ep=30):
    outer, inner = build_paths()
    dives = hits = steps = in_state = comps = 0
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(seed0 + k))
        prev_lane = w.lane[0]
        was_ahead = False
        while True:
            steps += 1
            if is_reentry(w):
                in_state += 1
            done = w.step(counter(w))
            if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                dives += 1
                if is_reentry(w):
                    hits += 1
            prev_lane = w.lane[0]
            if w.delta_true() > 0.30 and not was_ahead:
                comps += 1
                was_ahead = True
            if done:
                break
    return dives, hits, steps, in_state, comps


def main():
    print("=" * 76)
    print("M1.3 RE-ANCHOR (corrected definition)")
    print("  hit = blocker lane_target==1 AND lane==0 (re-entry transient)")
    print("=" * 76)
    td = th = ts = ti = tc = 0
    for s in (123000, 200000, 300000):
        d, h, st, ins, c = audit(s)
        td += d; th += h; ts += st; ti += ins; tc += c
        print(f"  seed0={s}: dives={d} hits={h} "
              f"({100*h/d if d else float('nan'):.0f}%)  completions={c}")
    base = ti / ts if ts else float("nan")
    hit = th / td if td else float("nan")
    print()
    print(f"  counter hit rate   = {hit:.3f}")
    print(f"  base rate (in re-entry state) = {base:.3f}")
    print(f"  lift = {hit/base if base else float('nan'):.2f}x")
    print()
    print(f"  => M1.3 threshold anchored at 1.5x base = {1.5*base:.3f}")
    print(f"  => completion rate (M1.1 anchor)        = {tc}/90 = {tc/90:.2f}")


if __name__ == "__main__":
    main()
