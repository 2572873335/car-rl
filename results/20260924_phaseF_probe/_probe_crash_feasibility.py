"""Corrected feasibility probe for the crash-adversary proposal.

The first version inverted its own verdict test (it printed YES when the lanes
are farther apart than the collision radius, which is the case where a
lane-centred adversary CANNOT reach). Redone carefully.

Key quantities:
  lane separation (centrelines)  = 0.15 m
  COLLISION_2D (frozen)          = 0.12 m
  OFFTRACK (frozen)              = 0.40 m   (lateral budget per car)

So at exact centreline positions the cars are 0.15 apart and do NOT collide
(3 cm clear). Collision becomes reachable only if the two cars' combined lateral
deviation toward each other exceeds 0.03 m -- which is trivially inside the
0.40 m off-track budget. This probe quantifies that and checks what the frozen
env permits.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
from car_following_sim import LoopPath, A_START
import overtake_env as oe

sep = 0.15          # env comment: lane separation
coll = oe.COLLISION_2D
offtrack = oe.OFFTRACK

print("=" * 86)
print("CRASH-ADVERSARY FEASIBILITY (corrected)")
print("=" * 86)
print(f"  lane separation (centrelines) : {sep:.3f} m")
print(f"  COLLISION_2D                  : {coll:.3f} m")
print(f"  OFFTRACK (per-car lateral cap): {offtrack:.3f} m")

# ---- 1. centreline case -------------------------------------------------
clear = sep - coll
print(f"\n[1] both cars centred on their own lanes:")
print(f"    separation - collision radius = {clear:+.3f} m")
print(f"    => collision? {sep < coll}   (so a cleanly-lane-centred pair does NOT collide)")

# ---- 2. how much lateral deviation is needed ----------------------------
need = coll - sep          # negative -> need this much closing
print(f"\n[2] closing needed for collision: {abs(need):.3f} m of COMBINED lateral")
print(f"    deviation toward each other (e.g. 0.015 m each, or 0.03 m by one side)")
print(f"    as a fraction of the off-track budget: {abs(need)/offtrack:.1%}")
print(f"    => reachable? {abs(need) < offtrack}  (trivially: "
      f"{abs(need):.3f} << {offtrack:.3f})")

# ---- 3. what does the frozen env actually let the leader do? -----------
import inspect
step_src = inspect.getsource(oe.OvertakeEnv.step)
pinned = "omega_cmd(self.leader, self.outer)" in step_src
print(f"\n[3] leader's steering in the frozen env:")
print(f"    leader follows `self.outer` (hardcoded): {pinned}")
print(f"    leader CANNOT leave the outer lane, so it cannot pursue laterally")

obs_src = inspect.getsource(oe.OvertakeEnv._obs)
sees_opp = any(k in obs_src for k in ("other", "opp", "adversar", "opponent"))
print(f"\n[4] defender observation:")
print(f"    contains opponent lane / lateral position: {sees_opp}")
print(f"    (slots: gap_ref, delta, v_self, e_lat, lane_SELF, v_leader)")

print("\n" + "=" * 86)
print("CONCLUSIONS")
print("=" * 86)
print(f" * Geometry ALLOWS a crash: only {abs(need):.3f} m of combined lateral")
print(f"   approach is needed, far inside the {offtrack:.2f} m off-track budget.")
print(f"   So the adversary idea is not geometrically blocked.")
print(f" * But the frozen env blocks BOTH requirements:")
print(f"     (a) the leader is pinned to `self.outer` -- it cannot steer into the")
print(f"         follower, so 'chase and crash' is impossible as written;")
print(f"     (b) the defender's observation has no opponent-lateral slot, so it")
print(f"         could not evade even if attacked.")
print(f" * => a crash adversary REQUIRES an env change. Editing the frozen file")
print(f"   invalidates the published 10/10 overtake result (same env).")
print(f"   Recommend a SEPARATE env module (adversary_env.py) reusing the frozen")
print(f"   primitives, leaving overtake_env.py byte-identical (iron rule 2).")
