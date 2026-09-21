"""Route B feasibility: tune the criterion world's leader speed into the rule
machine's competent range, so the rule machine becomes a LEGITIMATE positive.

Why this might be the right answer: the criterion's purpose (per W6) is to
measure whether a LEARNED policy can overtake. A positive control must be a
policy that demonstrably can overtake in this world. The rule machine is hand
written for exactly that, but its thresholds were tuned for a SLOW leader
(0.15-0.30 m/s per overtake_env.py). In world v2 the leader cruises 0.40-0.80,
so the rule machine rams it.

Sweep the leader cruise band and find where the rule machine succeeds. Then
check: does the FULL criterion still exclude loiter at that setting? (If the
lane geometry still lets loiter win, Route B does not help.)

Risk to record either way: if the band must be lowered a lot, the world
approaches the original frozen task and the criterion loses its point.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _criterion_v3 import WorldV3
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

ZERO2 = np.zeros(2)


class WorldTuned(WorldV3):
    """Leader cruise band is configurable, so we can sweep it."""

    def __init__(self, outer, inner, prior="v2v", lead_lo=0.40, lead_hi=0.80):
        super().__init__(outer, inner, prior)
        self.lead_lo, self.lead_hi = lead_lo, lead_hi

    def reset(self, rng):
        obs = super().reset(rng)
        # override the leader speed sampled by the parent
        self.lead_v = float(rng.uniform(self.lead_lo, self.lead_hi))
        return obs


def run(pol, lead_lo, lead_hi, n_ep=40, seed0=123000):
    outer, inner = build_paths()
    fulls, ot, colls, deltas = [], [], 0, []
    for k in range(n_ep):
        w = WorldTuned(outer, inner, prior="v2v",
                       lead_lo=lead_lo, lead_hi=lead_hi)
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0 = pol(obs[0], w.L)
            obs, done = w.step(np.stack([a0, ZERO2]), leader_action=ZERO2)
            if done:
                break
        fulls.append(w.m_full())
        ot.append(0.0 if np.isnan(w.t_first_ahead) else 1.0)
        deltas.append(w.m_delta())
        if w.reason == "collision":
            colls += 1
    return (np.mean(fulls), np.mean(ot), colls / n_ep, np.mean(deltas))


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]

    print("=" * 88)
    print("ROUTE B SWEEP: does the rule machine become a valid positive when the")
    print("leader cruises slower?  (and does FULL still exclude loiter?)")
    print("=" * 88)
    print(f"  {'leader band':>14s} | {'RULE: FULL':>10s} {'ot':>5s} {'coll':>6s} "
          f"| {'LOITER: FULL':>13s} {'ot':>5s} | {'CKPT: FULL':>11s}")
    print("-" * 88)

    for lo, hi in [(0.15, 0.30), (0.20, 0.35), (0.25, 0.45),
                   (0.30, 0.55), (0.40, 0.80)]:
        rf, ro, rc, _ = run(rule, lo, hi)
        lf, lo_, lc, _ = run(loiter, lo, hi)
        cf, co, cc, _ = run(ck, lo, hi)
        tag = " <-- frozen env's band" if (lo, hi) == (0.15, 0.30) else \
              (" <-- world v2 default" if (lo, hi) == (0.40, 0.80) else "")
        print(f"  {f'{lo:.2f}-{hi:.2f}':>14s} | {rf:>10.2f} {ro:>5.2f} {rc:>5.0%} "
              f"| {lf:>13.2f} {lo_:>5.2f} | {cf:>11.2f}{tag}")

    print()
    print("=" * 88)
    print("READING")
    print("=" * 88)
    print("  Want a band where RULE machine has FULL > 0 (valid positive) AND")
    print("  LOITER still has FULL = 0 (criterion still excludes geometry-buying).")
    print("  If no band satisfies both, Route B fails and Route A (BC warm start)")
    print("  is required.")


if __name__ == "__main__":
    main()
