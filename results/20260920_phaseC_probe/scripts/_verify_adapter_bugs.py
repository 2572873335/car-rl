"""INDEPENDENTLY VERIFY the reviewer's two claimed adapter bugs.

Neither is taken on trust: the reviewer is a subagent, and these conclusions
are already committed to a public repo. Verify from source + fresh runs.

Bug 1: _verify_f5_fix.py::selfplay_to_frozen hardcodes gap = 1.0 m, which may
       disable the rule machine's dive/brake branches (gap<0.45 / gap<0.15).
Bug 2: _ckpt_as_opponent.py::frozen_layout computes gap_ref = (L/2 - delta) % L
       while the frozen env defines gap_ref = (s_other - s_self) % L.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

import inspect
from overtake_env import baseline_action_ot
from _selfplay_design_probe import build_paths, wrapL
from _verify_f5_fix import NeutralWorld as World

print("=" * 78)
print("PART 1: what does baseline_action_ot actually require?")
print("=" * 78)
src = inspect.getsource(baseline_action_ot)
for ln in src.splitlines():
    if "gap" in ln or "a_lane" in ln or "a_speed" in ln:
        print("   ", ln.strip())

outer, inner = build_paths()
L = outer.length
print(f"\n   L = {L:.4f} m ; L/2 = {L/2:.4f} m")


print()
print("=" * 78)
print("PART 2: the gap formula. Construct a known state and compare.")
print("=" * 78)
print("   Frozen env (overtake_env._measure): gap_ref = (s_leader - s_follower) % L")
print("   Self-play obs slot0 = delta_self/2.5, where delta_self = -wrapL(s_other-s_self)")
print()

for behind in [0.30, 0.20, 0.10]:
    delta = -behind          # I'm BEHIND by `behind` metres -> delta_self negative
    correct = (-delta) % L
    mine = (L / 2.0 - delta) % L
    print(f"   opponent ahead {behind:.2f} m -> delta={delta:+.2f}")
    print(f"      CORRECT  ( -delta ) % L   = {correct:.4f} m  -> norm {(correct-0.2)/0.5:+.2f}")
    print(f"      MINE     (L/2-delta)%L   = {mine:.4f} m  -> norm {(mine-0.2)/0.5:+.2f}")
    print(f"      off by                   = {abs(mine-correct):.4f} m")
print()
print("   => if MINE reads ~3 m when truly 0.3 m, the rule machine sees")
print("      'nobody ahead' and its dive branch (gap<0.45) never fires.")


print()
print("=" * 78)
print("PART 3: does the FAKED gap disable the rule machine? (rule vs rule)")
print("=" * 78)


def frozen_layout_true(o, L):
    """CORRECTED: gap_ref = (-delta) % L."""
    delta = o[0] * 2.5
    gap_ref = (-delta) % L
    return np.array([np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
                     o[0], o[1], o[3], o[4], o[2]], np.float32)


def frozen_layout_fake(o, L):
    """What I originally used in _verify_f5_fix: gap pinned at 1.0 m."""
    delta_n, v_self, v_other, e_lat_n, lane, _ = o
    return np.array([(1.0 - 0.2) / 0.5, delta_n, v_self, e_lat_n, lane, v_other],
                    np.float32)


def rule_race(adapter, n_ep=40, seed0=4000):
    ever_inner = 0
    lead_changes = 0
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        prev = 0 if w.delta(0) > 0 else 1
        inner_used = False
        while True:
            a = np.stack([baseline_action_ot(adapter(obs[0], w.L)),
                          baseline_action_ot(adapter(obs[1], w.L))])
            obs, _, done = w.step(a)
            if max(w.lane) == 1:
                inner_used = True
            now = 0 if w.delta(0) > 0 else 1
            if now != prev:
                lead_changes += 1
                prev = now
            if done:
                break
        if inner_used:
            ever_inner += 1
    return ever_inner, lead_changes


for label, adapter in [("FAKE gap=1.0 (what I committed)", frozen_layout_fake),
                       ("CORRECT gap=(-delta)%L", frozen_layout_true)]:
    ei, lc = rule_race(adapter)
    print(f"   {label:34s}: inner-lane {ei}/40   lead changes {lc}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
print("   If FAKE shows ~0/40 and CORRECT shows many/40, the reviewer is right:")
print("   my 'the rule machine is not an adversary' conclusion is an ADAPTER")
print("   ARTIFACT, and it was committed to the repo in findings/ASSUMPTIONS.")
