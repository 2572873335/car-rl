"""Verify the reviewer's cleanest zero-model: SHIFT INVARIANCE.

Claim (T4): shifting the counter's dive times by +0.5 / +1 / +2 s leaves the
M1.3 hit rate UNCHANGED -- i.e. the metric does not depend on WHEN the policy
dives at all. Combined with R2 (random diving scores higher than the counter),
that seals M1.3's fate.

Also verify T1: every dive is DECIDED when lane_b == 0 (the counter reads the
blocker's physical lane, not its transient).

Implementation: intercept the counter's lane dive by forcing a dive at a
time-shifted schedule rather than its own logic. Simplest faithful version:
run the counter but SUPPRESS its dive command until t >= shift, then allow it.
That tests whether the hits survive a delay.
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


def counter_with_delay(shift):
    """The real counter, but its dive commands are suppressed until t >= shift."""
    def pol(w):
        g = w.gap_forward()
        d = w.delta_true()
        lane_f, lane_b = w.lane[0], w.lane[1]
        sw_f = w.switch[0]
        allow_dive = (shift is None) or (w.t >= shift)

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
        # would dive here, but honour the delay
        if not allow_dive:
            return np.array([0.5, 0.0])          # keep station instead
        if g < 1.40:
            return np.array([0.8, 1.0])
        return np.array([0.5, 0.0])
    return pol


def run(pol, n_ep=30, seeds=(123000, 200000, 300000)):
    outer, inner = build_paths()
    dives = hits = comps = 0
    n = 0
    for s in seeds:
        for k in range(n_ep):
            n += 1
            w = BlockerWorld3(outer, inner, cruise_v=0.55)
            w.reset(np.random.default_rng(s + k))
            prev_lane = w.lane[0]
            was_ahead = False
            while True:
                done = w.step(pol(w))
                if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                    dives += 1
                    if w.lane_target[1] == 1 and w.lane[1] == 0:
                        hits += 1
                prev_lane = w.lane[0]
                if w.delta_true() > 0.30 and not was_ahead:
                    comps += 1
                    was_ahead = True
                if done:
                    break
    return dives, hits, comps, n


def main():
    print("=" * 76)
    print("SHIFT-INVARIANCE ZERO MODEL (does the metric depend on WHEN?)")
    print("=" * 76)
    print(f"  {'delay':>8s} {'dives':>7s} {'hits':>7s} {'hit%':>8s} "
          f"{'complete':>9s}")
    for shift in (None, 0.5, 1.0, 2.0):
        d, h, c, n = run(counter_with_delay(shift))
        hp = h / d if d else float("nan")
        lbl = "none" if shift is None else f"{shift:.1f}s"
        print(f"  {lbl:>8s} {d:>7d} {h:>7d} {hp:>7.1%} {c/n:>9.2f}")
    print()
    print("  If hit% barely moves with delay, the metric is timing-independent")
    print("  and therefore not measuring timing ability -- M1.3 is void.")


if __name__ == "__main__":
    main()
