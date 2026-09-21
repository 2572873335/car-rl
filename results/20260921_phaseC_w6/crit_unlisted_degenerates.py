"""Pre-empt the reviewer's most valuable check: do UNLISTED mediocre
strategies pass the new criterion?

The plan's degenerate table covers zero-action, full-throttle and random.
But those are all obviously broken. The dangerous cases are PLAUSIBLE but
skill-less behaviours -- a policy that looks reasonable and still cannot
overtake. If any of these passes, the criterion is not ready.

Candidates (all deliberately "sensible-looking" but without overtaking skill):
  A. constant cruise  : hold a fixed lag, never attempt to pass
  B. lane-only        : always dive to the inner lane, never regulate speed
  C. speed-only       : full throttle but never change lane
  D. mirror-follower  : match the leader's speed exactly (perfect following)
  E. hesitant passer  : dive toward the inner lane but abort partway
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _rolefixed_probe import RoleWorld
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

GAP = 0.5
THR = 1.0


def outcomes(policy_fn, opponent_fn, n_ep=40, seed0=99000):
    out = []
    for k in range(n_ep):
        w = RoleWorld(*build_paths(), prior="v2v")
        obs = w.reset(np.random.default_rng(seed0 + k), gap=GAP)
        while True:
            a0, a1 = policy_fn(obs[0], w.L), opponent_fn(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        out.append((w.delta(0), w.reason))
    return out


def wr(out, thr=THR):
    return sum(1 for d, _ in out if d > thr) / len(out)


def main():
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))

    # obs layout: [delta_n, v_self_n, v_other_n, e_lat_n, lane, clear_n]
    def cruise(o, L):
        """Hold station: match opponent speed (a=0 in V2V = same speed)."""
        return np.array([0.0, 0.0])

    def lane_only(o, L):
        """Always dive inside, never regulate speed."""
        return np.array([0.0, 1.0])

    def speed_only(o, L):
        """Full throttle, never change lane."""
        return np.array([1.0, 0.0])

    def mirror(o, L):
        """Perfectly match the leader's speed (ideal follower, no overtake)."""
        v_self = o[1] * 1.3
        v_other = o[2] * 1.3
        # V2V prior: v = v_other + 0.8a -> a = (v_self - v_other)/0.8
        a = float(np.clip((v_self - v_other) / 0.8, -1, 1))
        return np.array([a, 0.0])

    def hesitant(o, L):
        """Try to pass, then back off: oscillate the lane command."""
        d = o[0]
        return np.array([0.3, 1.0 if d < 0 else -1.0])

    cands = [("A constant-cruise", cruise),
             ("B lane-only", lane_only),
             ("C speed-only", speed_only),
             ("D mirror-follower", mirror),
             ("E hesitant-passer", hesitant)]

    print("=" * 78)
    print("UNLISTED PLAUSIBLE-BUT-SKILL-LESS POLICIES vs the new criterion")
    print(f"  follower starts +{GAP} behind; win = delta > {THR}; "
          f"opponent = rule machine")
    print("=" * 78)
    print(f"  {'policy':>22s} {'win rate':>9s} {'mean_delta':>11s} {'reasons'}")
    print("-" * 78)
    flagged = []
    for name, fn in cands:
        o = outcomes(fn, rule)
        w = wr(o)
        md = float(np.mean([d for d, _ in o]))
        rs = {}
        for _, r in o:
            rs[r] = rs.get(r, 0) + 1
        mark = ""
        if w >= 0.2:
            mark = "  <-- WOULD PASS (criterion not ready!)"
            flagged.append(name)
        print(f"  {name:>22s} {w:>9.2f} {md:>+11.3f} {rs}{mark}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if flagged:
        print(f"  {len(flagged)} unlisted strategy(ies) scored >= 0.2:")
        for f_ in flagged:
            print(f"    - {f_}")
        print("  => The criterion does NOT yet exclude all skill-less behaviour.")
        print("     Threshold or design must be tightened before use.")
    else:
        print("  No unlisted plausible-but-skill-less policy scored >= 0.2.")
        print("  => The criterion survives this probe. (Still subject to the")
        print("     reviewer's independent degenerate set.)")


if __name__ == "__main__":
    main()
