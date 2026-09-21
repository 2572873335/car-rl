"""Complete degenerate-reference table (the user's mandated four items) plus
a threshold-tightness check.

Required pass conditions (user's spec):
  - zero-action  -> min win rate ~ 0
  - full-throttle -> no wins (crashes)
  - random       -> ~ 0
  - POSITIVE: the existing PPO overtake policy (the 1.4 s one) as follower
    must be significantly > 0   <-- without this, all-zero is indistinguishable

Also: the positive control's mean_delta was +2.36 m against a 0.3 m threshold.
Check whether 0.3 discriminates or is so loose that mediocre policies pass.
Sweep the threshold over the positive control and the degenerate set.
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

GAPS = [0.5, -0.5]          # fixed behind / fixed ahead (user's design)


def ep_outcomes(policy_fn, opponent_fn, prior="v2v", n_ep=40, seed0=95000):
    """Return list of (follower_delta, reason) over fixed-gap starts."""
    out = []
    for k in range(n_ep):
        gap = GAPS[k % len(GAPS)]
        w = RoleWorld(*build_paths(), prior=prior)
        obs = w.reset(np.random.default_rng(seed0 + k), gap=gap)
        while True:
            a0, a1 = policy_fn(obs[0], w.L), opponent_fn(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        out.append((w.delta(0), w.reason, gap))
    return out


def win_rate_at(outcomes, thr):
    """Follower wins if it is ahead by > thr at the end."""
    return sum(1 for d, _, _ in outcomes if d > thr) / len(outcomes)


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    ckpf = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rng = np.random.default_rng(23)

    policies = [
        ("zero-action",   lambda o, L: np.zeros(2)),
        ("full-throttle", lambda o, L: np.array([1.0, 0.0])),
        ("random",        lambda o, L: rng.uniform(-1, 1, 2)),
        ("FROZEN CKPT(+)", ckpf),
    ]

    print("=" * 80)
    print("DEGENERATE-REFERENCE TABLE  (opponent = rule machine, V2V prior,")
    print("  follower starts behind by 0.5 or -0.5 on alternating episodes)")
    print("=" * 80)
    print(f"  {'policy':>16s} {'win@0.3':>9s} {'win@1.0':>9s} {'mean_d':>9s} "
          f"{'collision':>10s} {'reasons'}")
    print("-" * 80)

    cache = {}
    for name, fn in policies:
        out = ep_outcomes(fn, rule)
        cache[name] = out
        w03 = win_rate_at(out, 0.3)
        w10 = win_rate_at(out, 1.0)
        md = float(np.mean([d for d, _, _ in out]))
        coll = sum(1 for _, r, _ in out if r == "collision") / len(out)
        rs = {}
        for _, r, _ in out:
            rs[r] = rs.get(r, 0) + 1
        print(f"  {name:>16s} {w03:>9.2f} {w10:>9.2f} {md:>+9.3f} "
              f"{100*coll:>9.1f}% {rs}")

    print()
    print("=" * 80)
    print("THRESHOLD SWEEP: at what margin does the positive control separate?")
    print("=" * 80)
    print(f"  {'threshold':>10s} " + " ".join(f"{n:>16s}" for n, _ in policies))
    for thr in [0.3, 0.5, 1.0, 1.5, 2.0, 2.5]:
        row = " ".join(f"{win_rate_at(cache[n], thr):>16.2f}" for n, _ in policies)
        print(f"  {thr:>10.1f} {row}")

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    sep03 = win_rate_at(cache["FROZEN CKPT(+)"], 0.3)
    deg_max = max(win_rate_at(cache[n], 0.3)
                  for n in ["zero-action", "full-throttle", "random"])
    print(f"  At thr=0.3: positive={sep03:.2f}, worst degenerate={deg_max:.2f}")
    if sep03 > 0.5 and deg_max < 0.2:
        print("  => CRITERION DISCRIMINATES. Positive control passes, all three")
        print("     degenerate references fail. The four mandated checks hold.")
    else:
        print("  => Criterion does NOT separate; needs redesign before use.")
    print()
    print("  Note on the leader column (measured earlier): 'as leader' was 1.00")
    print("  for EVERY policy including zero-action, because as leader a win is")
    print("  'the follower failed to get ahead' -- satisfied trivially by")
    print("  timeout/crash. So min(follower, leader) == follower column; the")
    print("  leader role adds no discrimination and must be documented as such.")


if __name__ == "__main__":
    main()
