"""Trimmed: the decisive robustness question only.

F2 (follower always starts +0.5 behind) at thr=1.0, positive control vs a
STRONG leader (the checkpoint itself), because a criterion that only shows the
positive control beating a weak rule machine may just be measuring opponent
weakness.
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

from _rolefixed_probe import RoleWorld
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot


def outcomes(policy_fn, opponent_fn, n_ep=20, seed0=97000, gap=0.5):
    out = []
    for k in range(n_ep):
        w = RoleWorld(*build_paths(), prior="v2v")
        obs = w.reset(np.random.default_rng(seed0 + k), gap=gap)
        while True:
            a0, a1 = policy_fn(obs[0], w.L), opponent_fn(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        out.append((w.delta(0), w.reason))
    return out


def wr(out, thr=1.0):
    return sum(1 for d, _ in out if d > thr) / len(out)


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    ckpf = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    zero = lambda o, L: np.zeros(2)

    print("=" * 72)
    print("F2 (always +0.5 behind), thr=1.0")
    print(f"  {'policy':>16s} {'vs rule':>10s} {'vs ckpt':>10s} "
          f"{'mean_d(rule)':>13s}")
    print("-" * 72)
    for n, fn in [("zero-action", zero), ("FROZEN CKPT(+)", ckpf)]:
        a = outcomes(fn, rule)
        b = outcomes(fn, ckpf)
        md = np.mean([d for d, _ in a])
        print(f"  {n:>16s} {wr(a):>10.2f} {wr(b):>10.2f} {md:>+13.3f}")

    print()
    print("  reasons vs rule (positive control):",
          {r: sum(1 for _, x in outcomes(ckpf, rule) if x == r)
           for r in set(x for _, x in outcomes(ckpf, rule))})
    print("  reasons vs ckpt (positive control):",
          {r: sum(1 for _, x in outcomes(ckpf, ckpf) if x == r)
           for r in set(x for _, x in outcomes(ckpf, ckpf))})


if __name__ == "__main__":
    main()
