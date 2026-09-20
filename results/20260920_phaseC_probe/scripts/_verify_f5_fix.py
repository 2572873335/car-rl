"""Verify the F5 FIX: does randomising the initial lead de-saturate the
win-rate criterion?

Root cause found: car 0 always started ~0.43 m ahead, so "win = be ahead at
timeout" was won by a do-nothing policy 20/20.  Proposed fix: randomise which
car starts ahead.

A fix I recommend must be TESTED, not assumed (iron rule 1 applies to my own
recommendations too).  Success criterion for the fix:
  - a do-nothing policy should NOT win ~100% any more
  - the win rate should become sensitive to policy quality
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import World, build_paths
from overtake_env import baseline_action_ot


class NeutralWorld(World):
    """F5 FIX: randomise which car starts ahead, symmetric magnitude."""

    def reset(self, rng):
        self.rng = rng
        s_A = self.outer.nearest(A_START)[0]
        gap0 = float(rng.uniform(0.15, 0.60))
        # FIX: random sign -> neither car has a structural head start
        sign = 1.0 if rng.random() < 0.5 else -1.0
        self.gap0 = gap0 * sign
        self.cars, self.steer, self.lane = [], [], []
        self.lane_cd = [0.0, 0.0]
        self.switch = [0.0, 0.0]
        for i in range(2):
            # offsets 0 and -gap0 => car1 leads when sign<0
            s = (s_A - i * self.gap0) % self.L
            p, t = self.outer.point_at(s), self.outer.tan[0]
            self.cars.append(Car(*p, float(np.arctan2(t[1], t[0]))))
            self.steer.append(PurePursuit())
            self.lane.append(0)
        self.t, self.done = 0.0, False
        return self.obs()


from car_following_sim import Car, PurePursuit, A_START  # noqa: E402


def selfplay_to_frozen(o):
    """Explicit obs-layout adapter (F5 fix #2): my layout -> frozen layout."""
    delta_n, v_self, v_other, e_lat_n, lane, _ = o
    return np.array([(1.0 - 0.2) / 0.5, delta_n, v_self, e_lat_n, lane, v_other],
                    np.float32)


def win_rate(world_cls, policy_fn, opp_fn, n_ep=60, seed0=3000):
    wins = ties = losses = 0
    for k in range(n_ep):
        w = world_cls(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0 = policy_fn(obs[0])
            a1 = opp_fn(obs[1])
            obs, _, done = w.step(np.stack([a0, a1]))
            if done:
                break
        d = w.delta(0)
        if d > 0.05:
            wins += 1
        elif d < -0.05:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def main():
    rnd = np.random.default_rng(11)

    policies = {
        "do-nothing": lambda o: np.zeros(2),
        "random":     lambda o: rnd.uniform(-1, 1, 2),
        "full-thr":   lambda o: np.array([1.0, 0.0]),
    }
    rule_opp = lambda o: baseline_action_ot(selfplay_to_frozen(o))

    print("=" * 78)
    print("F5 FIX VERIFICATION -- win rate of car 0 vs rule opponent (60 eps)")
    print("=" * 78)
    print(f"{'policy':12s} {'OLD (biased start)':>22s} {'NEW (neutral start)':>22s}")
    print("-" * 78)
    for name, pol in policies.items():
        wo, lo, to = win_rate(World, pol, rule_opp)
        wn, ln, tn = win_rate(NeutralWorld, pol, rule_opp)
        print(f"{name:12s} {f'W{wo} L{lo} T{to}':>22s} {f'W{wn} L{ln} T{tn}':>22s}")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    wn, ln, tn = win_rate(NeutralWorld, lambda o: np.zeros(2), rule_opp)
    sat_old = win_rate(World, lambda o: np.zeros(2), rule_opp)[0]
    print(f"  do-nothing: OLD won {sat_old}/60 -> NEW wins {wn}/60")
    if wn < 55:
        print("  => FIX WORKS: the criterion is no longer won by doing nothing,")
        print("     so P2 now measures real overtaking ability.")
    else:
        print("  => FIX INSUFFICIENT: still saturated; the rule opponent may be")
        print("     too weak.  P2 needs a stronger opponent (e.g. a trained")
        print("     snapshot) rather than the hand-coded machine.")

    print()
    print("  NOTE: if do-nothing still scores high, that itself is informative:")
    print("  the frozen rule machine, used as a per-agent policy, may not be a")
    print("  competent ADVERSARY in this overtaking game (it was tuned as a")
    print("  task solver, not as a defender).  W6's league design must account")
    print("  for that -- a self-play league is not the same as 'vs the script'.")


if __name__ == "__main__":
    main()
