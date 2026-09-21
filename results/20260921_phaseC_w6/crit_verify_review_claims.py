"""Independently verify the criterion review's two decisive claims.

Claim 1 (B1): the crit_* scripts import _ckpt_as_opponent.frozen_layout, which
  STILL contains the retracted gap bug -> the positive control's 1.00 is an
  artifact, and with the fix it collapses to 0.00.
Claim 2 (B2): RoleWorld.reset never uses rng -> deterministic policies give
  win rate in {0,1} only; "30 episodes x 3 seeds" is one rollout repeated.

Both are mechanically checkable. Do not take either on trust.
"""
import sys
import inspect
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

print("=" * 78)
print("CLAIM 1: is the gap bug still present in the adapter the crit_* use?")
print("=" * 78)

import _ckpt_as_opponent
src = inspect.getsource(_ckpt_as_opponent.frozen_layout)
print("  _ckpt_as_opponent.frozen_layout source:")
for ln in src.splitlines():
    print("   ", ln)

BUGGY = "L / 2.0 - delta" in src or "L/2.0 - delta" in src or "L / 2 - delta" in src
print()
print(f"  contains the retracted '(L/2 - delta)' form?  {BUGGY}")
if BUGGY:
    print("  => CLAIM 1 CONFIRMED: the crit scripts feed the BUGGY adapter.")
else:
    print("  => claim 1 not reproduced.")

print()
print("=" * 78)
print("CLAIM 1b: does the positive control collapse under the FIXED adapter?")
print("=" * 78)

from _rolefixed_probe import RoleWorld
from _selfplay_design_probe import build_paths
from overtake_env import baseline_action_ot

L_REF = 5.4849


def fixed_layout(o, L):
    """Corrected adapter: gap_ref = (-delta) % L."""
    delta = o[0] * 2.5
    gap_ref = (-delta) % L
    return np.array([np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
                     o[0], o[1], o[3], o[4], o[2]], np.float32)


def run(pol, opp, n_ep=40, gap=0.5, thr=1.0):
    wins = coll = 0
    deltas = []
    for k in range(n_ep):
        w = RoleWorld(*build_paths(), prior="v2v")
        obs = w.reset(np.random.default_rng(98000 + k), gap=gap)
        while True:
            a0, a1 = pol(obs[0], w.L), opp(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        d = w.delta(0)
        deltas.append(d)
        if d > thr:
            wins += 1
        if w.reason == "collision":
            coll += 1
    return wins / n_ep, coll / n_ep, len(set(np.round(deltas, 6)))


from stable_baselines3 import PPO
ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")

combos = [
    ("ckpt[BUGGY] vs rule[BUGGY]  (what the plan used)",
     lambda o, L: ckpt.predict(_ckpt_as_opponent.frozen_layout(o, L),
                               deterministic=True)[0],
     lambda o, L: baseline_action_ot(_ckpt_as_opponent.frozen_layout(o, L))),
    ("ckpt[BUGGY] vs rule[FIXED]",
     lambda o, L: ckpt.predict(_ckpt_as_opponent.frozen_layout(o, L),
                               deterministic=True)[0],
     lambda o, L: baseline_action_ot(fixed_layout(o, L))),
    ("ckpt[FIXED]  vs rule[BUGGY]",
     lambda o, L: ckpt.predict(fixed_layout(o, L), deterministic=True)[0],
     lambda o, L: baseline_action_ot(_ckpt_as_opponent.frozen_layout(o, L))),
    ("ckpt[FIXED]  vs rule[FIXED]  (correct config)",
     lambda o, L: ckpt.predict(fixed_layout(o, L), deterministic=True)[0],
     lambda o, L: baseline_action_ot(fixed_layout(o, L))),
]

print(f"  {'configuration':>46s} {'win@1.0':>8s} {'coll%':>7s} {'uniq_d':>7s}")
print("-" * 78)
for label, pol, opp in combos:
    w, c, u = run(pol, opp)
    print(f"  {label:>46s} {w:>8.2f} {100*c:>6.1f}% {u:>7d}")

print()
print("=" * 78)
print("CLAIM 2: is the world deterministic (no rng in reset)?")
print("=" * 78)
w1 = RoleWorld(*build_paths(), prior="v2v")
w1.reset(np.random.default_rng(1), gap=0.5)
p1 = [c.pos.copy() for c in w1.cars]
w2 = RoleWorld(*build_paths(), prior="v2v")
w2.reset(np.random.default_rng(999999), gap=0.5)
p2 = [c.pos.copy() for c in w2.cars]
same = all(np.allclose(a, b) for a, b in zip(p1, p2))
print(f"  rng(1) vs rng(999999) initial positions identical? {same}")
if same:
    print("  => CLAIM 2 CONFIRMED: reset ignores rng; episodes are identical.")
print()
print("  unique final deltas for a DETERMINISTIC policy over 40 'episodes':")
for label, pol in [
    ("do-nothing", lambda o, L: np.zeros(2)),
    ("ckpt[FIXED]", lambda o, L: ckpt.predict(fixed_layout(o, L), deterministic=True)[0]),
]:
    _, _, u = run(pol, lambda o, L: baseline_action_ot(fixed_layout(o, L)))
    print(f"    {label:>14s}: unique = {u}  (1 => all 40 episodes identical)")
