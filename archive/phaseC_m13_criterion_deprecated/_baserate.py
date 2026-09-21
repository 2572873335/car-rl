"""Is the counter's 45% dive-hit rate ABOVE CHANCE?

Without the base rate, 45% means nothing. Measure what fraction of ALL time
the blocker spends transitioning during an episode. If the counter's dives hit
the window at a rate clearly above that base rate, the window is being
exploited; if the rates are similar, the counter is diving at random with
respect to the blocker's state and F23's claim would need qualification.
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
    outer, inner = build_paths()
    dives = hits = steps = trans_steps = 0
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(seed0 + k))
        prev_lane = w.lane[0]
        while True:
            # sample the blocker's state BEFORE the step
            steps += 1
            if w.switch[1] > 0.0:
                trans_steps += 1
            a = counter(w)
            done = w.step(a)
            if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                dives += 1
                if w.switch[1] > 0.0:
                    hits += 1
            prev_lane = w.lane[0]
            if done:
                break
    return dives, hits, steps, trans_steps


def main():
    print("=" * 74)
    print("BASE-RATE TEST: is 45% above chance?")
    print("=" * 74)
    td = th = ts = tt = 0
    for s in (123000, 200000, 300000):
        d, h, st, tr = audit(s)
        td += d; th += h; ts += st; tt += tr
        print(f"  seed0={s}: dives={d} hits={h} | blocker transitioning "
              f"{tr}/{st} steps = {100*tr/st:.1f}% of time")
    print()
    base = tt / ts
    hit = th / td
    print(f"  base rate (blocker transitioning) = {base:.3f} "
          f"({100*base:.1f}%)")
    print(f"  counter dive-hit rate              = {hit:.3f} ({100*hit:.1f}%)")
    print(f"  lift = {hit/base if base else float('nan'):.2f}x")
    print()
    if hit > base * 1.5:
        print("  => dives hit the window well ABOVE chance: the counter IS")
        print("     timing the transient (F23 stands).")
        print(f"  => M1.3 could be anchored as: policy dive-hit rate must")
        print(f"     exceed the base rate by >= 1.5x, i.e. >= {1.5*base:.2f}")
    else:
        print("  => dives are only marginally above chance. The counter wins")
        print("     largely by OTHER means, and F23's window claim must be")
        print("     qualified: the transient helps (settle-wait scored 0.00)")
        print("     but is not the sole mechanism.")


if __name__ == "__main__":
    main()
