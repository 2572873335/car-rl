"""Measure the ACTUAL lateral transient in the current blocker world.

The ruling's mechanism argument: the deadlock's every cycle rests on
"lane change is zero-duration" -- the blocker re-occupies the inner lane the
instant its cooldown expires. The fix should make the commitment window
physical: 1.5 s minus the real lateral travel time.

Before implementing, MEASURE what the transient actually is. The car follows
a path via PurePursuit, so when `lane` flips its target path changes and the
body turns toward it -- a physical move, but possibly fast enough to be
effectively instant for blocking purposes.

Measure:
  - lateral offset from the old path over time after a lane command
  - time to reach the new lane's centre (|e_lat| < 0.02 on the new path)
  - whether the blocker's LOGICAL lane (the variable) leads the physical move
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

outer, inner = build_paths()


def main():
    w = BlockerWorld2(outer, inner, cruise_v=0.55, guard_hi=0.90)
    w.reset(np.random.default_rng(123000))

    # hold everything still except force a lane command for car 1 at t=0
    print("=" * 78)
    print("LATERAL TRANSIENT MEASUREMENT (car 1 commanded outer -> inner)")
    print("=" * 78)
    print(f"  outer lane pt vs inner lane pt: lateral separation 0.15 m")
    print()
    print(f"  {'t':>6s} {'lane_var':>9s} {'switch_t':>9s} "
          f"{'e_lat_outer':>12s} {'e_lat_inner':>12s} {'lateral_disp':>13s}")

    # force blocker to inner: call step with an explicit blocker action
    forced = np.array([0.0, 1.0])
    t_reached = None
    for k in range(200):
        if k % 5 == 0:
            e_out = outer.nearest(w.cars[1].pos)[1]
            e_in = inner.nearest(w.cars[1].pos)[1]
            # signed lateral distance from the outer path, in metres
            disp = abs(e_out)
            print(f"  {w.t:>6.2f} {w.lane[1]:>9d} {w.switch[1]:>9.2f} "
                  f"{e_out:>12.4f} {e_in:>12.4f} {disp:>13.4f}")
        if t_reached is None and abs(inner.nearest(w.cars[1].pos)[1]) < 0.02:
            t_reached = w.t
        done = w.step(np.zeros(2), blocker_a=forced)
        if done:
            break

    print()
    print(f"  time for car 1 to physically reach the inner lane centre: "
          f"{t_reached if t_reached is not None else float('nan'):.2f} s")
    print(f"  SWITCH_COOLDOWN = 1.5 s;  switch_timer (speed cap) = 0.8 s")
    print()
    print("READING")
    print("  If the physical move takes ~0.s, the logical lane variable leads")
    print("  the body by that much -- and the blocker 'occupies' the inner lane")
    print("  while still physically on the outer path. That gap is what makes")
    print("  the commitment window look longer on paper than it is in physics.")


if __name__ == "__main__":
    main()
