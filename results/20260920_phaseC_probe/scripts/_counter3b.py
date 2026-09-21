"""Harden the counter: the target band now yields 0.70 but with 30% collisions.

Diagnose and reduce collisions, then test stability across seeds. Goal per the
ruling: a STABLE winning counter at the common band (0.08-0.90), so a threshold
can be anchored to its score.

Likely collision source: the counter dives while the blocker is also moving,
so they meet laterally. The fix is to dive only when the blocker has finished
its own transition (switch_b == 0) AND its body is settled outer -- then the
blocker must spend ~0.6-0.8 s returning before it can contest again.
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


def make_counter(dive_gap=1.40, abort_gap=1.10, push_speed=1.0,
                 wait_speed=0.5, require_settled=True):
    def counter(w):
        g = w.gap_forward()
        d = w.delta_true()
        lane_f = w.lane[0]
        lane_b = w.lane[1]
        sw_f = w.switch[0]
        sw_b = w.switch[1]

        if d > 0.30:
            return np.array([0.0, -1.0 if lane_f == 1 else 0.0])
        if sw_f > 0:                       # finish own transition
            return np.array([0.8 if lane_f == 1 else 0.0, 0.0])

        if lane_f == 1:
            if lane_b == 0 and (sw_b == 0 or not require_settled):
                return np.array([push_speed, 0.0])
            return np.array([-0.4, -1.0])

        # outer
        if lane_b == 1:
            return (np.array([-0.7, 0.0]) if g < abort_gap
                    else np.array([0.0, 0.0]))
        # blocker physically outer
        settled = (sw_b == 0) if require_settled else True
        if settled and g < dive_gap:
            return np.array([0.8, 1.0])
        if g > dive_gap:
            return np.array([wait_speed, 0.0])
        return np.array([0.0, 0.0])
    return counter


def run(counter, seed0, n_ep=20, cruise_v=0.55, guard_hi=0.90, guard_lo=0.08):
    outer, inner = build_paths()
    comp, coll, ts = [], 0, []
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=cruise_v,
                          guard_hi=guard_hi, guard_lo=guard_lo)
        w.reset(np.random.default_rng(seed0 + k))
        while True:
            if w.step(counter(w)):
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"))


def main():
    print("=" * 84)
    print("COUNTER HARDENING: does requiring the blocker SETTLED cut collisions?")
    print("=" * 84)
    for label, rs in [("require_settled=True ", True),
                      ("require_settled=False", False)]:
        for dg in (1.10, 1.40, 1.80):
            c = make_counter(dive_gap=dg, require_settled=rs)
            r = run(c, 123000, n_ep=20)
            print(f"  {label} dive_gap={dg:.2f} -> complete={r[0]:.2f} "
                  f"coll={r[1]:.0%} med_t={r[2]:.2f}")

    print()
    print("=" * 84)
    print("STABILITY across seed sets (best config)")
    print("=" * 84)
    best = make_counter(dive_gap=1.40, require_settled=True)
    for s in (123000, 200000, 300000, 400000):
        c, cl, t = run(best, s, n_ep=20)
        print(f"  seed0={s} -> complete={c:.2f} coll={cl:.0%} med_t={t:.2f}")


if __name__ == "__main__":
    main()
