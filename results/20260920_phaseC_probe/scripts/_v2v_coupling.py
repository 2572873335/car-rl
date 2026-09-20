"""Does the V2V prior make speed-based overtaking IMPOSSIBLE in peer self-play?

Analytical concern with v_cmd = v_opponent + 0.8*a in a PEER race:

  In the frozen env the leader's speed is EXOGENOUS (scripted, 0.15-0.30 m/s),
  so v_l is a fixed external signal -> agent can beat it by commanding a=1.

  In peer self-play BOTH agents reference EACH OTHER:
      v0_cmd = v1 + 0.8*a0
      v1_cmd = v0 + 0.8*a1
  If both play the same action, their commands are identical and the speed
  difference is preserved (locked). Any speed lead one gains is copied by the
  other, because the other's REFERENCE moves up with it. So speed alone may
  no longer be a competitive axis -- the only way to gain ground would be the
  shorter inner lane.

If true, that fundamentally changes the game W6 would be training, and makes
"V2V prior" a poor choice for a peer race (it was designed for a
leader-follower task with an exogenous reference).

Test it directly: two cars, scripted actions, measure whether a persistent
speed asymmetry can be created at all.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _v2v_transfer_test import WorldV2V
from _selfplay_design_probe import build_paths, World

VS = 1.3
GAIN = 0.8


def rollout(world, policy0, policy1, steps=1500):
    w = world
    w.reset(np.random.default_rng(1234))
    vs = []
    for k in range(steps):
        a = np.stack([policy0(k, w), policy1(k, w)])
        w.step(a)
        vs.append((w.cars[0].v, w.cars[1].v))
        if w.done:
            break
    return np.array(vs)


def main():
    outer, inner = build_paths()
    print("=" * 78)
    print("Speed control under the two priors: can one car outrun the other?")
    print("=" * 78)

    for label, WorldCls in [("SELF-relative (W5 world)", World),
                            ("V2V / opponent-relative (W6 plan)", WorldV2V)]:
        print()
        print(f"--- {label} ---")
        # car 0 full throttle, car 1 brakes (max asymmetry)
        w = WorldCls(outer, inner)
        vs = rollout(w, lambda k, w_: np.array([1.0, 0.0]),
                     lambda k, w_: np.array([-1.0, 0.0]))
        print(f"  a0=+1 (full), a1=-1 (brake):")
        print(f"    mean v0={vs[:,0].mean():.3f}  mean v1={vs[:,1].mean():.3f}"
              f"  v0-v1 mean={(vs[:,0]-vs[:,1]).mean():+.3f}")
        print(f"    final v0={vs[-1,0]:.3f}  v1={vs[-1,1]:.3f}")

        # both full throttle -- does the pair accelerate together (lock)?
        w2 = WorldCls(outer, inner)
        vs2 = rollout(w2, lambda k, w_: np.array([1.0, 0.0]),
                      lambda k, w_: np.array([1.0, 0.0]))
        print(f"  BOTH a=+1:")
        print(f"    mean v0={vs2[:,0].mean():.3f}  mean v1={vs2[:,1].mean():.3f}"
              f"  v0-v1 mean={(vs2[:,0]-vs2[:,1]).mean():+.3f}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    w = WorldV2V(outer, inner)
    vs = rollout(w, lambda k, w_: np.array([1.0, 0.0]),
                 lambda k, w_: np.array([-1.0, 0.0]))
    gap = (vs[:, 0] - vs[:, 1]).mean()
    print(f"  Under V2V, max-asymmetry speed gap attained: {gap:+.3f} m/s")
    if abs(gap) < 0.15:
        print("  => The speed axis is SEVERELY COMPRESSED by the V2V coupling:")
        print("     with a full-throttle vs braking split you still barely")
        print("     separate.  Overtaking becomes almost purely a LANE-CHANGE")
        print("     problem.  That is a different game from W5's -- and it makes")
        print("     'V2V prior for a peer race' a questionable design choice.")
    else:
        print("  => Speed asymmetry survives; the coupling is not fatal.")


if __name__ == "__main__":
    main()
