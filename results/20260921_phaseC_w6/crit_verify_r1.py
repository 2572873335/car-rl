"""Independently verify review2's R1: does a two-line GEOMETRY policy score
FULL = 1.00, indistinguishable from the positive control?

Strategy under test (reviewer's, allegedly skill-free):
    a_speed = 0 always; lane = inner if delta < 0.30 else outer

If this matches the positive control's FULL on the same seeds, then FULL does
not measure overtaking skill -- it measures "did you ever move to the inner
lane and come back", which is free for anyone who is already ahead.

Also checks R2 (FULL vs collision decoupling) and R4 (rule machine scored 0
despite completing the manoeuvre -- suspected lap-wrap bug in
held_after_return).
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
from stable_baselines3 import PPO
from _criterion_v3 import WorldV3
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

ZERO2 = np.zeros(2)


class WT(WorldV3):
    def __init__(self, outer, inner, lo, hi):
        super().__init__(outer, inner, prior="v2v")
        self.lead_lo, self.lead_hi = lo, hi

    def reset(self, rng):
        obs = super().reset(rng)
        self.lead_v = float(rng.uniform(self.lead_lo, self.lead_hi))
        return obs


def run(pol, seed0, n_ep=30, lo=0.15, hi=0.30):
    outer, inner = build_paths()
    full, ot, coll, ts = [], [], 0, []
    for k in range(n_ep):
        w = WT(outer, inner, lo, hi)
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            obs, done = w.step(np.stack([pol(obs[0], w.L), ZERO2]),
                               leader_action=ZERO2)
            if done:
                break
        full.append(w.m_full())
        if np.isnan(w.t_first_ahead):
            ot.append(0.0)
        else:
            ot.append(1.0)
            ts.append(w.t_first_ahead)
        if w.reason == "collision":
            coll += 1
    return (float(np.mean(full)), float(np.mean(ot)), coll / n_ep,
            float(np.mean(ts)) if ts else float("nan"))


def main():
    ckpt = PPO.load(f"{REPO}/ckpt_ot/overtake_final_v1.zip")
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))

    def geom(o, L):
        d = o[0] * 2.5
        return np.array([0.0, 1.0 if d < 0.30 else -1.0])

    def geom_slow(o, L):
        d = o[0] * 2.5
        return np.array([-0.3, 1.0 if d < 0.30 else -1.0])

    print("=" * 82)
    print("R1 VERIFICATION: two-line geometry policy vs positive control")
    print("=" * 82)
    print(f"  {'seed0':>8s} {'policy':>14s} {'FULL':>7s} {'ot':>5s} "
          f"{'coll':>7s} {'t_ot':>8s}")
    print("-" * 82)
    for seed0 in [123000, 200000, 300000]:
        for name, fn in [("ckpt(+)", ck), ("geom(2-line)", geom),
                         ("geom_slow", geom_slow), ("rule machine", rule)]:
            f, o, c, t = run(fn, seed0)
            print(f"  {seed0:>8d} {name:>14s} {f:>7.2f} {o:>5.2f} "
                  f"{c:>6.0%} {t:>8.2f}")

    print()
    print("=" * 82)
    print("VERDICT")
    print("=" * 82)
    fc, _, cc, _ = run(ck, 123000)
    fg, _, cg, _ = run(geom, 123000)
    print(f"  positive ckpt : FULL={fc:.2f} coll={cc:.0%}")
    print(f"  two-line geom : FULL={fg:.2f} coll={cg:.0%}")
    if abs(fc - fg) < 1e-9:
        print()
        print("  => R1 CONFIRMED: a skill-free two-line policy scores FULL")
        print("     IDENTICALLY to the intended positive control.")
        print("     FULL does NOT measure overtaking skill; it measures")
        print("     'entered the inner lane and came back', which is free.")
    else:
        print()
        print("  => R1 NOT reproduced (geom differs from positive).")


if __name__ == "__main__":
    main()
