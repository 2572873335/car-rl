"""Verify B3: does a two-line, skill-less "loiter" strategy PASS the criterion,
and does it beat the plan's own positive control at thr=2.5?

Reviewer's claim: strategy [brake if close else 0, always inner lane] scores
final_delta ~ +2.55, passing thresholds up to 2.5, while the plan's positive
control (+2.36) drops out at 2.5.

Also verify I4: delta is bounded by wrapL at +-L/2, so any "sit on the inner
lane" policy's delta climbs to ~+L/2 and pins -- meaning the "clean plateau"
is a wrap-saturation band, not a skill band.
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

import inspect
print("=" * 78)
print("I4 CHECK: is delta bounded by wrapL at +-L/2?")
print("=" * 78)

from _selfplay_design_probe import wrapL
L = build_paths()[0].length
print(f"  L = {L:.4f}  ->  L/2 = {L/2:.4f}")
print(f"  delta(i) = -wrapL(s_other - s_self, L)  => range (-{L/2:.3f}, +{L/2:.3f}]")


def run(pol, opp, n_ep=40, gap=0.5):
    deltas, colls, inners = [], 0, []
    for k in range(n_ep):
        w = RoleWorld(*build_paths(), prior="v2v")
        obs = w.reset(np.random.default_rng(98000 + k), gap=gap)
        inner_steps = 0
        while True:
            a0, a1 = pol(obs[0], w.L), opp(obs[1], w.L)
            if w.lane[0] == 1:
                inner_steps += 1
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        deltas.append(w.delta(0))
        if w.reason == "collision":
            colls += 1
        inners.append(inner_steps * 0.02)
    return np.array(deltas), colls / n_ep, np.mean(inners)


def wr(deltas, thr):
    return float(np.mean(deltas > thr))


from stable_baselines3 import PPO
ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")

# the reviewer's loiter: brake (-1) when the other car is close, else 0; always inner lane
def make_loiter(brk):
    def loiter(o, L):
        clear = o[5]           # clearance slot
        return np.array([-brk if clear < 0.3 else 0.0, 1.0])
    return loiter

print()
print("=" * 78)
print("B3 CHECK: does 'loiter' (two lines, no overtaking skill) pass?")
print("=" * 78)
print(f"  {'policy':>26s} {'final_delta':>12s} {'inner_s':>9s} {'coll%':>7s} "
      f"{'pass@1.0':>9s} {'pass@2.5':>9s}")
print("-" * 78)

cands = [
    ("loiter(brk=0.25)", make_loiter(0.25)),
    ("loiter(brk=0.5)", make_loiter(0.5)),
    ("plan positive ctrl", lambda o, L: ckpt.predict(frozen_layout(o, L),
                                                    deterministic=True)[0]),
]
res = {}
for name, fn in cands:
    d, c, inner = run(fn, lambda o, L: baseline_action_ot(frozen_layout(o, L)))
    res[name] = d
    print(f"  {name:>26s} {d.mean():>+12.3f} {inner:>9.1f} {100*c:>6.1f}% "
          f"{wr(d,1.0):>9.2f} {wr(d,2.5):>9.2f}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
lo = res.get("loiter(brk=0.25)")
po = res.get("plan positive ctrl")
if lo is not None and po is not None:
    if lo.mean() >= po.mean():
        print(f"  LOITER (mean {lo.mean():+.3f}) >= positive control "
              f"(mean {po.mean():+.3f})")
        print("  => B3 CONFIRMED: a two-line skill-less policy scores at least")
        print("     as well as the intended positive control. The criterion")
        print("     ranks a degenerate behaviour first.")
    if abs(lo.mean()) > L / 4:
        print(f"  AND loiter's delta ({lo.mean():+.3f}) sits near the wrap")
        print(f"     bound L/2 = {L/2:.3f} => it is riding the inner-lane")
        print(f"     geometry, not demonstrating overtaking skill (I4 holds).")
