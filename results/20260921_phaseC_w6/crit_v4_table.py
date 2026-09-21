"""Criterion v4, self-contained trimmed run.

Inlines the run helper so there is no extra module to place. Tests whether the
wrap-safe COMPLETE measure + collision gate separates the positive control from
the two-line geometry policy.

Critical control: zero-action MUST score COMPLETE=0.00 after the progress-init
fix. If it does not, the measure is still broken.
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
from _criterion_v4 import WorldV4          # world is fine; only `run` clashed
from _selfplay_design_probe import build_paths

ZERO2 = np.zeros(2)


def run(pol, seed0, n_ep=20, lo=0.15, hi=0.30):
    outer, inner = build_paths()
    comp, coll, ts = [], 0, []
    for k in range(n_ep):
        w = WorldV4(outer, inner, lo, hi)
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            obs, done = w.step(np.stack([pol(obs[0], w.L), ZERO2]),
                               leader_action=ZERO2)
            if done:
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"))


def main():
    from stable_baselines3 import PPO
    from _ckpt_as_opponent import frozen_layout
    from overtake_env import baseline_action_ot

    ckpt = PPO.load(f"{REPO}/ckpt_ot/overtake_final_v1.zip")
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    rng = np.random.default_rng(5)

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    def geom(o, L):
        d = o[0] * 2.5
        return np.array([0.0, 1.0 if d < 0.30 else -1.0])

    def geom_slow(o, L):
        d = o[0] * 2.5
        return np.array([-0.3, 1.0 if d < 0.30 else -1.0])

    pols = [
        ("zero-action  ", lambda o, L: np.zeros(2)),
        ("full-throttle", lambda o, L: np.array([1.0, 0.0])),
        ("loiter       ", loiter),
        ("geom 2-line  ", geom),
        ("geom_slow    ", geom_slow),
        ("rule machine ", rule),
        ("ckpt (+)     ", ck),
    ]

    print("=" * 72)
    print("CRITERION v4 (seed0=123000, 20 eps) -- progress init FIXED")
    print("=" * 72)
    print(f"  {'policy':>14s} {'COMPLETE':>9s} {'coll%':>7s} {'med_t_ot':>9s} "
          f"{'accept':>7s}")
    print("-" * 72)
    for name, fn in pols:
        c, cl, t = run(fn, 123000, n_ep=20)
        acc = "PASS" if (c >= 0.8 and cl <= 0.10) else "fail"
        print(f"  {name:>14s} {c:>9.2f} {cl:>6.0%} {t:>9.2f} {acc:>7s}")

    print()
    c0, _, _ = run(lambda o, L: np.zeros(2), 123000, n_ep=20)
    print(f"CONTROL: zero-action COMPLETE = {c0:.2f} ->",
          "OK (measure sane)" if c0 < 0.05 else "STILL BROKEN")


if __name__ == "__main__":
    main()
