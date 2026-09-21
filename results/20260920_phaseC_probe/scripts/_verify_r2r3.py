"""Verify the DECISIVE causal claims: R2 (blind/random divers match the
counter) and R3 (hits are ANTI-correlated with winning).

R3 is the killer: if suppressing the target state (never dive when it holds)
gives completion 0.77 -- same as the counter -- while diving ONLY in that state
gives 0.00, then M1.3 does not track what it claims.

R2: a blind diver (dive whenever lane_b == 0, no timing) and a random diver
should score around the counter's hit rate under the corrected definition.
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
from _blocker_v3 import BlockerWorld3
from _selfplay_design_probe import build_paths

GUARD_HI = 0.90


def make_counter(mode, rng=None):
    """mode: 'counter' | 'blind' | 'random' | 'only_target' | 'never_target'"""

    def pol(w):
        g = w.gap_forward()
        d = w.delta_true()
        lane_f, lane_b = w.lane[0], w.lane[1]
        sw_f = w.switch[0]
        target = (w.lane_target[1] == 1) and (w.lane[1] == 0)

        # ---- gating hooks for the causal tests ----
        if mode == "only_target" and not target:
            return _drive(w, "hold")
        if mode == "never_target" and target:
            return _drive(w, "hold")

        if mode == "random":
            if d > 0.30:
                return np.array([0.0, -1.0 if lane_f == 1 else 0.0])
            if sw_f > 0:
                return np.array([0.8 if lane_f == 1 else 0.0, 0.0])
            if rng.random() < 0.15:
                return np.array([0.0, 1.0 if lane_f == 0 else -1.0])
            return np.array([0.5, 0.0])

        if mode == "blind":
            # dive whenever the blocker is NOT on the inner lane; no timing
            if d > 0.30:
                return np.array([0.0, -1.0 if lane_f == 1 else 0.0])
            if sw_f > 0:
                return np.array([0.8 if lane_f == 1 else 0.0, 0.0])
            if lane_f == 1:
                if lane_b == 0:
                    return np.array([1.0, 0.0])
                return np.array([-0.4, -1.0])
            if lane_b == 0:
                return np.array([0.8, 1.0])
            return np.array([0.5, 0.0])

        # default: the real counter
        if d > 0.30:
            return np.array([0.0, -1.0 if lane_f == 1 else 0.0])
        if sw_f > 0:
            return np.array([0.8 if lane_f == 1 else 0.0, 0.0])
        if lane_f == 1:
            if lane_b == 0:
                return np.array([1.0, 0.0])
            return np.array([-0.4, -1.0])
        if lane_b == 1:
            return np.array([-0.7, 0.0] if g < 1.10 else [0.0, 0.0])
        if g < 1.40:
            return np.array([0.8, 1.0])
        return np.array([0.5, 0.0])

    return pol


def _drive(w, which):
    """Neutral driving (hold station) for the gated tests."""
    d = w.delta_true()
    lane_f = w.lane[0]
    sw_f = w.switch[0]
    if d > 0.30:
        return np.array([0.0, -1.0 if lane_f == 1 else 0.0])
    if sw_f > 0:
        return np.array([0.8 if lane_f == 1 else 0.0, 0.0])
    if lane_f == 1:
        return np.array([0.5, 0.0])
    return np.array([0.5, 0.0])


def run(pol, n_ep=30, seeds=(123000, 200000, 300000)):
    outer, inner = build_paths()
    dives = hits = comps = colls = 0
    n = 0
    for s in seeds:
        for k in range(n_ep):
            n += 1
            w = BlockerWorld3(outer, inner, cruise_v=0.55)
            w.reset(np.random.default_rng(s + k))
            prev_lane = w.lane[0]
            was_ahead = False
            while True:
                done = w.step(pol(w))
                if w.lane[0] == 1 and prev_lane == 0 and w.delta_true() < 0.0:
                    dives += 1
                    if w.lane_target[1] == 1 and w.lane[1] == 0:
                        hits += 1
                prev_lane = w.lane[0]
                if not was_ahead and (not np.isnan(w.t_first_ahead)):
                    comps += 1
                    was_ahead = True
                if done:
                    if w.reason == "collision":
                        colls += 1
                    break
    return dives, hits, comps, colls, n


def main():
    rng = np.random.default_rng(7)
    print("=" * 78)
    print("R2/R3 VERIFICATION")
    print("=" * 78)
    print(f"  {'policy':>16s} {'dives':>6s} {'hits':>6s} {'hit%':>7s} "
          f"{'complete':>9s} {'coll%':>7s}")
    for name, mode in [("counter (real)", "counter"), ("blind diver", "blind"),
                       ("random diver", "random"),
                       ("ONLY in target", "only_target"),
                       ("NEVER in target", "never_target")]:
        pol = make_counter(mode, rng)
        d, h, c, cl, n = run(pol)
        hp = h / d if d else float("nan")
        print(f"  {name:>16s} {d:>6d} {h:>6d} {hp:>6.1%} "
              f"{c/n:>9.2f} {cl/n:>6.0%}")


if __name__ == "__main__":
    main()
