"""Reconcile the t <= 0.04 discrepancy.

My audit records dive time as w.t AFTER the step that completes the lane
entry. The reviewer claims all 90 hits are at t <= 0.04. Both could be true if
the dive is DETECTED at the same instant the blocker's opening transient is
active, but the recorded wall time differs by measurement convention.

The mechanism is structural (reviewer's strongest claim): reset() places the
follower 100% inside the guard band behind, so the blocker ALWAYS commands the
inner lane at t=0, and the counter dives because it sees lane_b==0. Verify:
  (a) is the follower always in the guard band at reset? gap0 ~ U(0.4,0.9),
      guard_hi = 0.90 -> gap_forward == gap0 in (0.4, 0.9) subset of (0.08,0.9)?
  (b) does the blocker command inner at the very first step in every episode?
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

outer, inner = build_paths()
GUARD_LO, GUARD_HI = 0.08, 0.90

print("=" * 78)
print("STRUCTURAL CHECK: is the opening 'hit' mechanically guaranteed?")
print("=" * 78)

in_band = 0
first_cmd = {1: 0, 0: 0}
N = 90
for s in (123000, 200000, 300000):
    for k in range(30):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(s + k))
        # (a) follower's gap at reset vs guard band
        g = w.gap_forward()
        if GUARD_LO < g < GUARD_HI:
            in_band += 1
        # (b) what does the blocker command on the very first step?
        ba = w.blocker_action()
        cmd = "inner" if ba[1] > 0.3 else ("outer" if ba[1] < -0.3 else "hold")
        first_cmd[1 if cmd == "inner" else 0] += 1

print(f"  (a) follower inside guard band at reset: {in_band}/{N} = {in_band/N:.0%}")
print(f"  (b) blocker commands INNER at first step: {first_cmd[1]}/{N} = "
      f"{first_cmd[1]/N:.0%}")

print()
print("=" * 78)
print("SO: the counter's first dive is into the blocker's opening command.")
print("The dive TIME recorded by the step loop is the FIRST inner-entry,")
print("which can happen at any w.t if the car then returns to outer and")
print("re-enters later. Distinguish 'first dive of episode' vs 'any dive'.")
print("=" * 78)

# First-dive analysis: is the FIRST dive of each episode inside a target hit
# and at what time?
first_times = []
first_hit = 0
for s in (123000, 200000, 300000):
    for k in range(30):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(s + k))
        prev_lane = w.lane[0]
        first_t = None
        first_is_target = None
        while True:
            done = w.step(counter(w))
            if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                if first_t is None:
                    first_t = w.t
                    first_is_target = (w.lane_target[1] == 1 and w.lane[1] == 0)
            prev_lane = w.lane[0]
            if done:
                break
        if first_t is not None:
            first_times.append(first_t)
            if first_is_target:
                first_hit += 1

import numpy as np
ft = np.array(first_times)
print(f"  episodes with a first dive: {len(ft)}/90")
print(f"  first dive times: min={ft.min():.2f} median={np.median(ft):.2f} "
      f"max={ft.max():.2f}")
print(f"  fraction of first dives in target state: {first_hit}/{len(ft)} = "
      f"{first_hit/len(ft):.2f}")
