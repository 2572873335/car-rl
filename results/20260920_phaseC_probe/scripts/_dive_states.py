"""Why does the settled condition give hit rate 0.00 when the blocker still
guards (14% inner-lane time) and still has transitions (base 0.09)?

If the counter dives only when the blocker is settled-outer, that is a
DELIBERATE timing choice (wait for it to commit outward), not randomness --
and it would mean the counter is reading the blocker's state, just a different
signal than "mid-transition".

Instrument the settled condition: for each dive, record the blocker's
lane/lane_target/switch at that instant.
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


def settle(w):
    for _ in range(int(1.5 / 0.02)):
        w.step(np.zeros(2), blocker_a=w.blocker_action())
    w.switch[1] = 0.0
    w.lane_cd[1] = 0.0
    s_f = w.outer.nearest(w.cars[0].pos)[0]
    s_l = w.outer.nearest(w.cars[1].pos)[0]
    w.prog[0] = s_l - ((s_l - s_f) % w.L)
    w.prog[1] = s_l
    w.prev_s = [s_f, s_l]
    w.t, w.t_first_ahead, w.min_after_ahead = 0.0, np.nan, np.inf


def main():
    outer, inner = build_paths()
    print("=" * 78)
    print("At each DIVE, what is the blocker doing? (settled condition)")
    print("=" * 78)
    states = {}
    n_dive = 0
    for k in range(8):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(123000 + k))
        settle(w)
        prev_lane = w.lane[0]
        while True:
            done = w.step(counter(w))
            if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                n_dive += 1
                key = (w.lane[1], w.lane_target[1],
                       "moving" if w.switch[1] > 0 else "settled")
                states[key] = states.get(key, 0) + 1
            prev_lane = w.lane[0]
            if done:
                break
    print(f"  total dives = {n_dive}")
    for key, cnt in sorted(states.items(), key=lambda x: -x[1]):
        print(f"    blocker(lane={key[0]}, target={key[1]}, {key[2]}) "
              f"-> {cnt} dives")
    print()
    print("READING")
    print("  If dives cluster on a SPECIFIC blocker state, the counter is")
    print("  reading the blocker's state (a timing strategy), just not the")
    print("  mid-transition signal. If they spread evenly, it is not reading.")


if __name__ == "__main__":
    main()
