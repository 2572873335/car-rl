"""Follow-up: WHY does the settled start make the counter BETTER (0.87) while
removing all transient hits?

Hypothesis: with the opening transient suppressed, the blocker never enters
the guard band at all -- the counter wins by simply out-dragging a blocker
that has already released to the outer lane. In other words the settled start
changes the task: the blocker becomes passive.

Check: in the settled condition, does the blocker ever occupy the inner lane?
If it essentially never guards, then "completion 0.87" is a different (easier)
task, and neither number supports "the counter reads the window".

Also re-examine the normal condition: is the counter's 0.77 driven by the
transient, or by the same out-drag mechanism plus incidental overlap?
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


def audit(seed0, n_ep=30, suppress_opening=False):
    outer, inner = build_paths()
    inner_steps = 0
    steps = 0
    comps = 0
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(seed0 + k))
        if suppress_opening:
            for _ in range(int(1.5 / 0.02)):
                w.step(np.zeros(2), blocker_a=w.blocker_action())
            w.switch[1] = 0.0
            w.lane_cd[1] = 0.0
            s_f = w.outer.nearest(w.cars[0].pos)[0]
            s_l = w.outer.nearest(w.cars[1].pos)[0]
            w.prog[0] = s_l - ((s_l - s_f) % w.L)
            w.prog[1] = s_l
            w.prev_s = [s_f, s_l]
            w.t = 0.0
            w.t_first_ahead = np.nan
            w.min_after_ahead = np.inf
        was_ahead = False
        while True:
            steps += 1
            if w.lane[1] == 1:
                inner_steps += 1
            done = w.step(counter(w))
            if w.delta_true() > 0.30 and not was_ahead:
                comps += 1
                was_ahead = True
            if done:
                break
    return comps / n_ep, inner_steps / steps


def main():
    print("=" * 74)
    print("Does the BLOCKER still guard in the settled condition?")
    print("=" * 74)
    for label, sup in [("normal start  ", False), ("settled start ", True)]:
        cd = ig = 0.0
        for s in (123000, 200000, 300000):
            c, i = audit(s, suppress_opening=sup)
            cd += c / 3
            ig += i / 3
        print(f"  {label}: completion={cd:.2f}   blocker time on INNER lane "
              f"= {ig:.2f}")
    print()
    print("READING")
    print("  If 'blocker time on inner' collapses in the settled condition,")
    print("  the blocker stopped guarding -- the task changed, so 0.87 is not")
    print("  comparable, and neither result shows the counter READING a window.")


if __name__ == "__main__":
    main()
