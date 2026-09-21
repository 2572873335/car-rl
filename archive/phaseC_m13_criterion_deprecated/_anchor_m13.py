"""Anchor M1.3's threshold: what fraction of the COUNTER's overtakes happen
inside the blocker's lateral-transient window?

My plan set ">= 0.60" by guesswork, which violates the project rule that a
threshold needs a positive-pole basis (§10.4). The counter-script is the
positive pole, so measure ITS hit rate and anchor the threshold to that.

Definition (fixed in the plan, unchanged here):
  for each successful overtake at t*, check whether the blocker was physically
  transitioning (switch[1] > 0) at any point in [t* - 0.8, t*].
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

WINDOW = 0.80


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


def run_one(w, pol):
    """Roll out; return list of overtake times and the blocker's transient log."""
    trans_log = []          # (t, switch_active)
    overtakes = []
    was_ahead = False
    while True:
        trans_log.append((w.t, w.switch[1] > 0.0))
        d = w.delta_true()
        if d > 0.30 and not was_ahead:
            overtakes.append(w.t)
            was_ahead = True
        if w.step(pol(w)):
            break
    return overtakes, trans_log


def main():
    outer, inner = build_paths()
    hits = total = 0
    per_seed = []
    for seed0 in (123000, 200000, 300000):
        h = t = 0
        for k in range(30):
            w = BlockerWorld3(outer, inner, cruise_v=0.55)
            w.reset(np.random.default_rng(seed0 + k))
            overts, log = run_one(w, counter)
            for t_star in overts:
                t += 1
                in_window = any(active for (tt, active) in log
                                if t_star - WINDOW <= tt <= t_star)
                if in_window:
                    h += 1
        hits += h
        total += t
        per_seed.append((seed0, h, t, (h / t if t else float("nan"))))
        print(f"  seed0={seed0}: {h}/{t} overtakes inside a blocker transient "
              f"({100*h/t if t else float('nan'):.0f}%)")

    print()
    print("=" * 70)
    print("ANCHOR for M1.3")
    print("=" * 70)
    print(f"  counter-script overall: {hits}/{total} "
          f"({100*hits/total if total else float('nan'):.0f}%)")
    if total:
        rate = hits / total
        print(f"  observed rate = {rate:.2f}")
        print(f"  => a threshold anchored at 70% of the positive would be "
              f"{0.7*rate:.2f}")
        print(f"  => my plan's guessed 0.60 is "
              f"{'CONSISTENT' if 0.7*rate <= 0.60 <= rate else 'OFF'} "
              f"relative to the positive band [{0.7*rate:.2f}, {rate:.2f}]")


if __name__ == "__main__":
    main()
