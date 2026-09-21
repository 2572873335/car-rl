"""Verify R4: does the rule machine complete the manoeuvre but score FULL=0
because of a lap-wrap bug in held_after_return?

Observed in the R1 run: rule machine ot=1.00, collisions 0%, t_overtake 2.58 s
-- yet FULL = 0.00. The frozen env guards this exact case
(overtake_env.py:229-231: ahead_flag is never reset because delta wraps at
+-L/2 once the follower laps the leader). WorldV3.held_after_return has no such
guard, so when delta wraps past -L/2 it records "fell behind" incorrectly.

Instrument one episode: track delta over time and flag whether
held_after_return was flipped by a wrap rather than a genuine loss of the lead.
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
SCRIPTS = f"{REPO}/results/20260920_phaseC_probe/scripts"
while "/tmp" in sys.path:
    sys.path.remove("/tmp")
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)

import numpy as np
from _criterion_v3 import WorldV3
from _selfplay_design_probe import build_paths, wrapL
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

ZERO2 = np.zeros(2)


class WT(WorldV3):
    def __init__(self, outer, inner):
        super().__init__(outer, inner, prior="v2v")

    def reset(self, rng):
        obs = super().reset(rng)
        self.lead_v = 0.25          # in the rule machine's competent band
        return obs


def main():
    outer, inner = build_paths()
    L = outer.length
    w = WT(outer, inner)
    obs = w.reset(np.random.default_rng(123000))

    print("=" * 78)
    print("R4: rule machine, instrumented (look for a wrap-induced flip)")
    print("=" * 78)
    print(f"  L={L:.4f}  L/2={L/2:.4f}  (delta wraps here)")
    print()
    print(f"  {'t':>6s} {'delta':>9s} {'lane':>5s} {'first_ahead':>12s} "
          f"{'returned':>9s} {'held':>6s}")

    prev_held = None
    flipped_at = None
    for k in range(1500):
        a0 = baseline_action_ot(frozen_layout(obs[0], w.L))
        obs, done = w.step(np.stack([a0, ZERO2]), leader_action=ZERO2)
        if k % 100 == 0 or (prev_held is not None and w.held_after_return != prev_held):
            fa = ("nan" if np.isnan(w.t_first_ahead)
                  else f"{w.t_first_ahead:.2f}")
            print(f"  {w.t:>6.2f} {w.delta(0):>+9.3f} {w.lane[0]:>5d} {fa:>12s} "
                  f"{str(w.returned_outer):>9s} {str(w.held_after_return):>6s}")
            if prev_held is not None and w.held_after_return != prev_held:
                flipped_at = (w.t, w.delta(0))
        prev_held = w.held_after_return
        if done:
            print(f"  -> {w.reason} at t={w.t:.2f}")
            break

    print()
    print(f"  FULL = {w.m_full():.2f}")
    print(f"  t_first_ahead = {w.t_first_ahead}")
    print(f"  returned_outer = {w.returned_outer}")
    print(f"  held_after_return = {w.held_after_return}")
    if flipped_at:
        t, d = flipped_at
        print(f"  held flipped at t={t:.2f} with delta={d:+.3f}")
        near_wrap = abs(abs(d) - L / 2) < 0.6
        print(f"  |delta| near wrap bound L/2? {near_wrap}")
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not np.isnan(w.t_first_ahead) and w.returned_outer and not w.held_after_return:
        print("  R4 CONFIRMED: the machine completed overtake + return,")
        print("  yet held_after_return = False. If the flip happened near the")
        print("  wrap bound, it is a lap-wrap artefact -- the frozen env guards")
        print("  this (overtake_env.py:229-231); WorldV3 does not.")
    else:
        print("  R4 not reproduced in this episode.")


if __name__ == "__main__":
    main()
