"""INDEPENDENTLY VERIFY the M1 review's core claims against my own correction.

My correction (commit b76200e) claimed:
  (a) the old hit definition (switch[1] > 0, "any lateral move") wrongly counted
      the OUTWARD move
  (b) the true targeted state is the RE-ENTRY transient (lane_target==1 & lane==0)
  (c) lift = 6.45x

The reviewer says:
  R1-a: 0 of 202 dives occurred in the outward-move state, so old and new
        definitions classify all dives IDENTICALLY -- my correction changed the
        label, not the measurement; the lift changed only via the denominator.
  R2:   a blind diver (dive whenever lane_b==0, no timing) scores ~44.8%,
        indistinguishable from the counter's 44.6%; a random diver scores
        >10%, over my new threshold.
  R3:   restricting dives to the target state gives completion 0.00; restricting
        to non-target gives 0.77 (same as the counter). So hits are
        ANTI-correlated with winning.
  R1-c: all 90 "hits" happen at t <= 0.04 (the opening).

Verify each with fresh code. Do NOT take the reviewer on trust.
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


def classify(w):
    """(lane, target, moving) of the blocker."""
    return (w.lane[1], w.lane_target[1],
            "moving" if w.switch[1] > 0 else "settled")


def is_target(w):
    """My corrected definition: re-entry transient."""
    return (w.lane_target[1] == 1) and (w.lane[1] == 0)


def audit(pol, n_ep=30, seeds=(123000, 200000, 300000)):
    outer, inner = build_paths()
    dives = []          # dicts with state, time, is_target
    steps = in_target = 0
    comps = 0
    for s in seeds:
        for k in range(n_ep):
            w = BlockerWorld3(outer, inner, cruise_v=0.55)
            w.reset(np.random.default_rng(s + k))
            prev_lane = w.lane[0]
            was_ahead = False
            while True:
                steps += 1
                if is_target(w):
                    in_target += 1
                done = w.step(pol(w))
                if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                    dives.append({
                        "t": w.t,
                        "state": classify(w),
                        "target": is_target(w),
                        "old_def": w.switch[1] > 0.0,
                    })
                prev_lane = w.lane[0]
                if w.delta_true() > 0.30 and not was_ahead:
                    comps += 1
                    was_ahead = True
                if done:
                    break
    return dives, comps, n_ep * len(seeds), steps, in_target


def main():
    print("=" * 84)
    print("CLAIM R1-a: does the old definition ever catch the OUTWARD move?")
    print("=" * 84)
    dives, comps, n_eps, steps, intg = audit(counter)

    outward = [d for d in dives if d["state"] == (0, 0, "moving")]
    print(f"  total dives = {len(dives)}")
    print(f"  dives in OUTWARD-move state (lane0,target0,moving) = {len(outward)}")
    old = sum(1 for d in dives if d["old_def"])
    new = sum(1 for d in dives if d["target"])
    print(f"  old def (switch>0) hits = {old}/{len(dives)} = {old/len(dives):.3f}")
    print(f"  new def (target)   hits = {new}/{len(dives)} = {new/len(dives):.3f}")
    print(f"  => definitions identical on every dive? {old == new}")
    print(f"  => R1-a {'CONFIRMED (my correction was a relabel)' if old == new and len(outward) == 0 else 'NOT confirmed'}")

    print()
    print("=" * 84)
    print("CLAIM R1-c: are all hits at t <= 0.04 (opening)?")
    print("=" * 84)
    early = [d for d in dives if d["t"] <= 0.04]
    late_target = [d for d in dives if d["target"] and d["t"] > 0.04]
    print(f"  dives at t <= 0.04                    = {len(early)}")
    print(f"  TARGET-state hits at t > 0.04         = {len(late_target)}")
    print(f"  => all target-hits are opening? {len(late_target) == 0}")

    print()
    print("=" * 84)
    print("BASE RATE under the corrected definition")
    print("=" * 84)
    print(f"  time in target state = {intg}/{steps} = {intg/steps:.4f}")


if __name__ == "__main__":
    main()
