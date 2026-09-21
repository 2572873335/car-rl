"""The COUNTER-SCRIPT: proof the blocker task is solvable, and that solving it
requires reading the opponent's state (not geometry).

Key mechanic: the blocker defends the inner lane while the follower is within
[guard_lo, guard_hi] behind. Its lane changes are subject to a 1.5 s cooldown
(SWITCH_COOLDOWN), i.e. once it switches it is COMMITTED for 1.5 s and cannot
switch back.

Exploit (the "commitment window"):
  1. back off until gap > guard_hi  -> blocker releases to OUTER
  2. wait until the blocker has just switched to outer (its cooldown starts)
  3. immediately dive inner and accelerate through the window: the blocker
     cannot re-enter the inner lane to deny until its cooldown expires
  4. once clearly ahead, return outer and hold

Steps 2-3 require observing the blocker's LANE and COOLDOWN STATE, i.e.
opponent-state reading. Geometry alone (always dive, or dive when close)
cannot time this window -- which is exactly what the blocker test showed.

The counter reads world state directly (oracle-style). For a learned policy the
same information must be exposed as observation slots; that is noted for the
plan, not needed to prove solvability.
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

GUARD_HI = 0.90
COOLDOWN = 1.5


def counter_action(w):
    """Return a0 for the follower given full world access (oracle counter)."""
    g = w.gap_forward()
    d = w.delta_true()
    lane_f = w.lane[0]
    lane_b = w.lane[1]
    cd_b = w.lane_cd[1]          # blocker's remaining commitment

    if d > 0.30 and lane_f == 1:
        # passed: return to outer and hold
        return np.array([0.0, -1.0])

    if lane_f == 1:
        # in the inner lane: push through the window, else hold
        blocker_out = (lane_b == 0)
        if blocker_out:
            return np.array([1.0, 0.0])      # accelerate through
        return np.array([0.0, -1.0])          # blocker holds inner: abort

    # currently outer
    if lane_b == 1:
        # blocker is defending: back off until it releases
        return np.array([-0.6, 0.0])
    # blocker is OUTER. If it has just committed (cooldown running), dive now.
    if cd_b > 0.15 and g < 1.20:
        return np.array([0.6, 1.0])          # dive into the window
    # otherwise close a little and wait for the window
    return np.array([0.2, 0.0])


def run_counter(seed0, n_ep=20, cruise_v=0.55, guard_hi=GUARD_HI, verbose=False):
    outer, inner = build_paths()
    comp, coll, ts = [], 0, []
    for k in range(n_ep):
        w = BlockerWorld2(outer, inner, cruise_v=cruise_v, guard_hi=guard_hi)
        w.reset(np.random.default_rng(seed0 + k))
        while True:
            done = w.step(counter_action(w))
            if done:
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
        if verbose and k == 0:
            print(f"    ep0: reason={w.reason} complete={w.complete()} "
                  f"t_first={w.t_first_ahead}")
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"))


def main():
    print("=" * 76)
    print("COUNTER-SCRIPT vs the blocker (positive control for the blocker task)")
    print("=" * 76)
    for cv in (0.45, 0.55, 0.70):
        for gh in (0.70, 0.90, 1.20):
            c, cl, t = run_counter(123000, n_ep=20, cruise_v=cv, guard_hi=gh)
            print(f"  cruise={cv:.2f} guard_hi={gh:.2f} -> "
                  f"complete={c:.2f} coll={cl:.0%} med_t={t:.2f}")

    print()
    print("Compare: geometry/reference policies under the SAME blocker")
    from stable_baselines3 import PPO
    from _ckpt_as_opponent import frozen_layout
    from overtake_env import baseline_action_ot
    from _blocker_v2 import run

    ckpt = PPO.load(f"{REPO}/ckpt_ot/overtake_final_v1.zip")
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))

    def geom(o, L):
        d = o[0] * 2.5
        return np.array([0.0, 1.0 if d < 0.30 else -1.0])

    for name, fn in [("geom 2-line", geom), ("rule machine", rule),
                     ("ckpt (+)", ck)]:
        c, cl, t, s, a = run(fn, 123000, n_ep=20, cruise_v=0.55)
        print(f"  {name:>14s} -> complete={c:.2f} coll={cl:.0%} med_t={t:.2f}")


if __name__ == "__main__":
    main()
