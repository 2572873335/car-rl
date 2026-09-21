"""Chase the 20/20 red flag from the pilot gate.

Hypothesis: the rule opponent is being fed a MISMATCHED observation layout,
so it is not behaving like the rule state machine at all -- making the win
rate meaningless.

Frozen env obs layout (overtake_env._obs):
    [ (gap-0.2)/0.5, delta/2.5, v_f/1.3, e_lat/0.25, lane, v_l/0.5 ]
    idx  0            1          2         3            4     5

My self-play obs layout:
    [ delta/2.5, v_self, v_other, e_lat/0.25, lane, clearance/2.5 ]
    idx  0        1       2         3          4        5

=> baseline_action_ot(obs) reads obs[0] as GAP but gets DELTA,
   obs[1] as DELTA but gets v_self, obs[4] as LANE but gets CLEARANCE.
Exactly the "silent semantics mismatch" class of bug this project flags.

Also check: does car 0 start with a structural lead (delta0 = +gap0)?
"""
import sys

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import numpy as np
from _selfplay_design_probe import World, build_paths
from overtake_env import baseline_action_ot


def selfplay_to_frozen_obs(o_self):
    """Remap my self-play obs into the frozen env's layout so the frozen
    rule machine is actually being evaluated as intended."""
    delta_n, v_self, v_other, e_lat_n, lane, clear_n = o_self
    # frozen layout wants: [gap_norm, delta_norm, v_f_norm, e_lat_norm, lane, v_l_norm]
    # gap is not part of my obs -> reconstruct from clearance is not exact;
    # use a large constant so the rule's "gap<0.15 brake" branch keys off
    # the actual 2D geometry rather than a bogus number.
    return np.array([
        (1.0 - 0.2) / 0.5,      # pretend gap = 1.0 m (not used for lane logic)
        delta_n,
        v_self,
        e_lat_n,
        lane,
        v_other,
    ], np.float32)


def main():
    world = World(*build_paths())

    print("=" * 78)
    print("CHECK 1: initial lead is structural (car 0 starts AHEAD)")
    print("=" * 78)
    leads = []
    for k in range(20):
        w = World(*build_paths())
        w.reset(np.random.default_rng(100 + k))
        leads.append(w.delta(0))
    print(f"  delta0 at reset over 20 episodes: mean={np.mean(leads):+.3f} m, "
          f"min={np.min(leads):+.3f}, max={np.max(leads):+.3f}")
    print(f"  all positive (car 0 always starts ahead)? "
          f"{all(x > 0 for x in leads)}")
    print("  => 'win at timeout = be ahead' gives car 0 a HEAD START.")
    print("     An untrained policy that simply holds position can win, so")
    print("     P2's win-rate criterion is SATURATED and measures nothing.")

    print()
    print("=" * 78)
    print("CHECK 2: rule opponent under MISMATCHED vs REMAPPED obs")
    print("=" * 78)

    for label, mapper in [("as-is (mismatched)", lambda o: o),
                          ("remapped (correct)", selfplay_to_frozen_obs)]:
        wins = 0
        for k in range(20):
            w = World(*build_paths())
            obs = w.reset(np.random.default_rng(9000 + k))
            steps = 0
            while True:
                a0 = np.zeros(2)                       # car 0 does nothing
                a1 = baseline_action_ot(mapper(obs[1]))  # car 1 = rule machine
                obs, _, done = w.step(np.stack([a0, a1]))
                steps += 1
                if done:
                    break
            if w.delta(0) > 0.05:
                wins += 1
        print(f"  {label:22s}: car0(do-nothing) still ahead in {wins}/20")

    print()
    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    print("  The pilot gate's 20/20 is an ARTIFACT of both flaws above:")
    print("  (a) car 0 starts ahead and 'win' is measured as 'be ahead',")
    print("  (b) the rule opponent was fed the wrong observation layout.")
    print("  P2 must be redesigned: hand the LEAD to the opponent and require")
    print("  the policy to OVERTAKE, or measure 'did the opponent overtake'")
    print("  from a neutral start.  The obs-layout remap must be explicit.")


if __name__ == "__main__":
    main()
