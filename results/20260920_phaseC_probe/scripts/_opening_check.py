"""Self-check the sharpest objection to F23/M1.3: is the counter exploiting a
GENUINE window, or merely a special OPENING transient?

The concern: the audit showed the counter dives at t = 0.02 s, which is the
blocker's initial lane change (it begins on the outer lane and may move inner
at t=0 depending on the very first gap sample). If the counter's wins come
from that opening transient only, then "it reads the window" is overstated --
it might just be exploiting a peculiarity of the reset.

Test: disable the opening advantage by giving the blocker a settled start
(start it already in the state it would hold, with switch[1] == 0 and cooldown
clear), then re-measure both the completion rate and the dive-hit rate.

If completion stays high and hit rate stays well above base, the window is real
throughout the episode. If both collapse, F23 needs the qualification.
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
    """If suppress_opening, clear the blocker's initial transient before the
    first step so the counter cannot ride the t=0 lane change."""
    outer, inner = build_paths()
    dives = hits = comps = steps = tsteps = 0
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=0.55)
        w.reset(np.random.default_rng(seed0 + k))
        if suppress_opening:
            # let the blocker settle: run it with a neutral follower for 1.5 s
            for _ in range(int(1.5 / 0.02)):
                w.step(np.zeros(2), blocker_a=w.blocker_action())
            w.switch[1] = 0.0
            w.lane_cd[1] = 0.0
            # reset the follower's progress baseline so delta is fair
            s_f = w.outer.nearest(w.cars[0].pos)[0]
            s_l = w.outer.nearest(w.cars[1].pos)[0]
            w.prog[0] = s_l - ((s_l - s_f) % w.L)
            w.prog[1] = s_l
            w.prev_s = [s_f, s_l]
            w.t = 0.0
            w.t_first_ahead = np.nan
            w.min_after_ahead = np.inf

        prev_lane = w.lane[0]
        was_ahead = False
        while True:
            steps += 1
            if w.switch[1] > 0.0:
                tsteps += 1
            done = w.step(counter(w))
            if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                dives += 1
                if w.switch[1] > 0.0:
                    hits += 1
            prev_lane = w.lane[0]
            if w.delta_true() > 0.30 and not was_ahead:
                comps += 1
                was_ahead = True
            if done:
                break
    return dives, hits, comps, n_ep, steps, tsteps


def main():
    for label, sup in [("NORMAL start (opening transient present)", False),
                       ("SETTLED start (opening advantage removed)", True)]:
        td = th = tc = ts = tt = 0
        n = 0
        for s in (123000, 200000, 300000):
            d, h, c, nep, st, tr = audit(s, suppress_opening=sup)
            td += d; th += h; tc += c; ts += st; tt += tr; n += nep
        print("=" * 76)
        print(f"{label}")
        print("=" * 76)
        print(f"  completion rate = {tc/n:.2f}  ({tc}/{n})")
        print(f"  dives = {td}, hits = {th} -> hit rate "
              f"{th/td if td else float('nan'):.2f}")
        print(f"  base rate (blocker transitioning) = "
              f"{tt/ts if ts else float('nan'):.2f}")
        print()

    print("READING")
    print("  If the settled start keeps completion high and hit rate well above")
    print("  base, then the window is a general property (F23 stands as stated).")
    print("  If both collapse, the counter was mainly riding the opening")
    print("  transient and F23 must be qualified accordingly.")


if __name__ == "__main__":
    main()
