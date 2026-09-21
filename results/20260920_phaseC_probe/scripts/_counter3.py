"""Counter v3: exploit the now-PHYSICAL commitment window.

With blocker v3, the blocker's occupancy is honest: after it releases to the
outer lane it needs ~0.6-0.8 s of body travel before it can physically occupy
the inner lane again. Its re-entry is therefore blocked for
    1.5 s (cooldown) ... but occupancy only counts on arrival,
so the true denial-free window is longer than the paper cooldown by the
transient time.

The counter attacks when the blocker's body is genuinely in the OUTER lane and
mid-transition (so it cannot be on the inner path), then pushes through.

World access is via the v3 object; the same quantities must later be exposed as
observation slots for a learned policy.
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


def counter_action(w):
    g = w.gap_forward()
    d = w.delta_true()
    lane_f = w.lane[0]
    lane_b = w.lane[1]
    tgt_b = w.lane_target[1]
    cd_b = w.lane_cd[1]
    sw_f = w.switch[0]

    # already ahead: return to outer and hold
    if d > 0.30:
        return np.array([0.0, -1.0 if lane_f == 1 else 0.0])

    # mid-transition: finish it, never reverse
    if sw_f > 0:
        return np.array([0.8 if lane_f == 1 else 0.0, 0.0])

    if lane_f == 1:
        # in inner: push while the blocker's BODY is not on the inner path
        if lane_b == 0:
            return np.array([1.0, 0.0])
        return np.array([-0.4, -1.0])

    # outer
    # blocker physically outer AND heading/released outward -> window is open
    blocker_out_or_leaving = (lane_b == 0)
    if blocker_out_or_leaving and g < 1.40:
        return np.array([0.8, 1.0])         # dive now
    if lane_b == 1:
        # blocker holds inner: stay outside the band, wait
        if g < 1.10:
            return np.array([-0.7, 0.0])
        return np.array([0.0, 0.0])
    if g > 1.40:
        return np.array([0.5, 0.0])
    return np.array([0.0, 0.0])


def run_counter(seed0, n_ep=20, cruise_v=0.55, guard_hi=0.90, guard_lo=0.08):
    outer, inner = build_paths()
    comp, coll, ts = [], 0, []
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=cruise_v,
                          guard_hi=guard_hi, guard_lo=guard_lo)
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
    print("=" * 78)
    print("COUNTER v3 vs blocker v3 -- with PHYSICAL lane occupancy")
    print("  (the ruling's test: common guard band 0.08-0.90)")
    print("=" * 78)
    print(f"  {'cruise':>7s} {'guard_hi':>9s} {'complete':>9s} {'coll%':>7s} "
          f"{'med_t':>7s}")
    for cv in (0.40, 0.55, 0.70):
        for gh in (0.70, 0.90, 1.20, 1.60):
            c, cl, t = run_counter(123000, n_ep=20, cruise_v=cv, guard_hi=gh)
            mark = "  <-- target band" if (cv == 0.55 and gh == 0.90) else ""
            print(f"  {cv:>7.2f} {gh:>9.2f} {c:>9.2f} {cl:>6.0%} {t:>7.2f}{mark}")


if __name__ == "__main__":
    main()
