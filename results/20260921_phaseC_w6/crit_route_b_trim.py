"""Route B trimmed: just the decisive cases, fewer episodes.
Can the rule machine be a valid positive in ANY leader band (with the FIXED
adapter)? Only the frozen band (0.15-0.30) and the v2 default (0.40-0.80)
matter. 20 episodes each is enough to see FULL vs collision.
"""
import sys

# F18/F19 BOOTSTRAP: resolve helper modules from the REPO, never /tmp.
# (/tmp copies caused a stale-module contamination and made these scripts
#  unrunnable from a clean checkout.)
import os as _os, sys as _sys
_RESULTS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_REPO = _os.path.dirname(_RESULTS)
_SCRIPTS = _os.path.join(_RESULTS, '20260920_phaseC_probe', 'scripts')
for _p in ('/tmp', '/home/zy/car_rl/code0919'):
    while _p in _sys.path:
        _sys.path.remove(_p)
for _p in (_SCRIPTS, _REPO):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np

from _criterion_v3 import WorldV3
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

ZERO2 = np.zeros(2)


class WorldTuned(WorldV3):
    def __init__(self, outer, inner, prior="v2v", lead_lo=0.40, lead_hi=0.80):
        super().__init__(outer, inner, prior)
        self.lead_lo, self.lead_hi = lead_lo, lead_hi

    def reset(self, rng):
        obs = super().reset(rng)
        self.lead_v = float(rng.uniform(self.lead_lo, self.lead_hi))
        return obs


def run(pol, lo, hi, n_ep=20, seed0=123000):
    outer, inner = build_paths()
    fulls, ot, colls, deltas = [], [], 0, []
    for k in range(n_ep):
        w = WorldTuned(outer, inner, prior="v2v", lead_lo=lo, lead_hi=hi)
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
    return np.mean(fulls), np.mean(ot), colls / n_ep, np.mean(deltas)


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]

    print("=" * 78)
    print("ROUTE B (FIXED adapter): can ANY leader band give a valid positive?")
    print("=" * 78)
    print(f"  {'band':>10s} {'policy':>16s} {'FULL':>7s} {'ot':>6s} "
          f"{'coll':>7s} {'mean_d':>9s}")
    print("-" * 78)
    for lo, hi in [(0.15, 0.30), (0.30, 0.55), (0.40, 0.80)]:
        for name, fn in [("rule", rule), ("loiter", loiter), ("ckpt", ck)]:
            f, o, c, d = run(fn, lo, hi)
            print(f"  {f'{lo:.2f}-{hi:.2f}':>10s} {name:>16s} {f:>7.2f} "
                  f"{o:>6.2f} {c:>6.0%} {d:>+9.3f}")
    print()
    print("  Want: some (band, policy=rule) with FULL > 0 and low collision,")
    print("  AND loiter still FULL = 0 at that band.")


if __name__ == "__main__":
    main()
