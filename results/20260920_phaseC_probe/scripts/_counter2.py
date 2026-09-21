"""Counter-script v2: dive only inside the blocker's TRUE commitment window.

Bug in v1: it dived whenever the blocker had cooldown, INCLUDING while the
blocker was still mid-transition (switch_timer > 0). Both cars then swapped
lanes at the same along-track point and the 2D distance fell below the
collision radius -- a self-inflicted crash, not a blocker win.

Correct reading of the mechanic: the blocker is committed for 1.5 s AFTER its
lane change completes. The window to attack is therefore when the blocker is
settled in the OUTER lane with most of its cooldown still running: it cannot
re-enter the inner lane in time to deny.

Also require the blocker to be settled (switch_timer == 0) before initiating,
so the two never swap simultaneously.
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
from _blocker_v2 import BlockerWorld2
from _selfplay_design_probe import build_paths

COOLDOWN = 1.5


def counter_action(w):
    g = w.gap_forward()
    d = w.delta_true()
    lane_f, lane_b = w.lane[0], w.lane[1]
    cd_b = w.lane_cd[1]
    sw_f, sw_b = w.switch[0], w.switch[1]

    # already ahead: return to outer and hold
    if d > 0.30:
        return np.array([0.0, -1.0 if lane_f == 1 else 0.0])

    # mid-transition: finish the manoeuvre, never reverse
    if sw_f > 0:
        return np.array([0.8 if lane_f == 1 else 0.0, 0.0])

    if lane_f == 1:
        # in inner: push hard while the blocker is not defending the inner lane
        if lane_b == 0 and sw_b == 0:
            return np.array([1.0, 0.0])
        return np.array([-0.4, -1.0])          # blocker defends: abort to outer

    # in outer
    if lane_b == 1:
        # blocker defends inner: hold station outside the guard band
        if g < 1.10:
            return np.array([-0.7, 0.0])
        return np.array([0.0, 0.0])

    # blocker is OUTER and settled: exploit the commitment window
    if sw_b == 0 and cd_b > 0.6 and g < 1.30:
        return np.array([0.8, 1.0])            # dive now
    # else close gently and wait
    if g > 1.30:
        return np.array([0.5, 0.0])
    return np.array([0.0, 0.0])


def run_counter(seed0, n_ep=20, cruise_v=0.55, guard_hi=0.90):
    outer, inner = build_paths()
    comp, coll, ts = [], 0, []
    for k in range(n_ep):
        w = BlockerWorld2(outer, inner, cruise_v=cruise_v, guard_hi=guard_hi)
        w.reset(np.random.default_rng(seed0 + k))
        while True:
            if w.step(counter_action(w)):
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"))


def main():
    print("=" * 74)
    print("COUNTER v2 (attack only inside the true commitment window)")
    print("=" * 74)
    print(f"  {'cruise':>7s} {'guard_hi':>9s} {'complete':>9s} {'coll%':>7s} "
          f"{'med_t':>7s}")
    for cv in (0.40, 0.55, 0.70):
        for gh in (0.70, 0.90, 1.20, 1.60):
            c, cl, t = run_counter(123000, n_ep=20, cruise_v=cv, guard_hi=gh)
            print(f"  {cv:>7.2f} {gh:>9.2f} {c:>9.2f} {cl:>6.0%} {t:>7.2f}")


if __name__ == "__main__":
    main()
