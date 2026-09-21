"""Corrected mechanism: the counter attacks when the blocker is committed OUTER,
not mid-transition. Verify by comparing dive-state distributions and testing
the claim directly.

Corrected hypothesis:
  the exploitable state is "blocker settled/committed on the OUTER lane while
  the follower is inside the guard band" -- because in that state the blocker
  must spend its own re-entry time to deny, and the follower can cross first.

Test: count dives by blocker state in the NORMAL condition too, and compare the
state distribution at dives against the state distribution overall (a lift
per state). If "blocker outer settled" has lift >> 1, that is the real signal.
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


def state_of(w):
    return (w.lane[1], w.lane_target[1],
            "moving" if w.switch[1] > 0 else "settled")


def audit(seed0, n_ep=30):
    outer, inner = build_paths()
    dive_states, all_states = {}, {}
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(seed0 + k))
        prev_lane = w.lane[0]
        while True:
            st = state_of(w)
            all_states[st] = all_states.get(st, 0) + 1
            done = w.step(counter(w))
            if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                dive_states[st] = dive_states.get(st, 0) + 1
            prev_lane = w.lane[0]
            if done:
                break
    return dive_states, all_states


def main():
    ds, als = {}, {}
    for s in (123000, 200000, 300000):
        a, b = audit(s)
        for k, v in a.items():
            ds[k] = ds.get(k, 0) + v
        for k, v in b.items():
            als[k] = als.get(k, 0) + v

    tot_dive = sum(ds.values()) or 1
    tot_all = sum(als.values()) or 1

    print("=" * 84)
    print("DIVE-STATE DISTRIBUTION vs OCCUPANCY (normal condition, 3 seed sets)")
    print("=" * 84)
    print(f"  {'blocker state':>34s} {'dives':>7s} {'dive%':>7s} "
          f"{'occ%':>7s} {'lift':>7s}")
    keys = set(list(ds.keys()) + list(als.keys()))
    for k in sorted(keys, key=lambda x: -ds.get(x, 0)):
        dp = ds.get(k, 0) / tot_dive
        op = als.get(k, 0) / tot_all
        lift = (dp / op) if op else float("nan")
        print(f"  {str(k):>34s} {ds.get(k,0):>7d} {100*dp:>6.1f}% "
              f"{100*op:>6.1f}% {lift:>7.2f}")
    print()
    print("  lift > 1 means the counter dives in that state more often than")
    print("  chance -- i.e. it is a state the counter TARGETS.")


if __name__ == "__main__":
    main()
