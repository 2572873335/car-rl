"""Criterion v4 + TIME BOUND: does a time limit on the overtake separate the
positive control from the geometry policies?

Data so far (seed 123000, 20 eps):
    ckpt (+)      complete 1.00   coll  0%   med_t_ot  1.64 s
    rule machine  complete 1.00   coll  0%   med_t_ot  2.63 s
    geom 2-line   complete 1.00   coll  0%   med_t_ot 20.01 s
    loiter        complete 1.00   coll  0%   med_t_ot 20.01 s

Completion alone cannot separate them; TIME appears to. The contest task is
"overtake promptly", and the frozen env's own baseline does it in 2.0 s.

Test a bound at several thresholds across seeds, with extra skill-less
candidates added (fast geometry) to try to break it.
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
from _criterion_v4 import WorldV4
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
    rng = np.random.default_rng(7)

    def geom(o, L):
        d = o[0] * 2.5
        return np.array([0.0, 1.0 if d < 0.30 else -1.0])

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    # extra attempts to break the time bound: fast geometry variants
    def geom_dive_early(o, L):
        """dive as soon as slightly behind, full throttle to close"""
        d = o[0] * 2.5
        return np.array([1.0, 1.0 if d < 0.60 else -1.0])

    def geom_smart(o, L):
        """dive only when close, then throttle"""
        d = o[0] * 2.5
        return np.array([1.0 if d < 0.45 else -0.5,
                         1.0 if d < 0.45 else -1.0])

    POLS = [
        ("ckpt (+)      ", ck),
        ("rule machine  ", rule),
        ("geom 2-line   ", geom),
        ("loiter        ", loiter),
        ("geom_dive_early", geom_dive_early),
        ("geom_smart    ", geom_smart),
        ("zero-action   ", lambda o, L: np.zeros(2)),
        ("full-throttle ", lambda o, L: np.array([1.0, 0.0])),
        ("random        ", lambda o, L: rng.uniform(-1, 1, 2)),
    ]

    print("=" * 84)
    print("v4 + TIME BOUND: completion AND collisions<=10% AND t_overtake<=T")
    print("=" * 84)
    hdr = f"  {'policy':>16s} "
    for T in (3.0, 5.0, 8.0):
        hdr += f"{'T='+str(T):>7s} "
    hdr += f"{'coll%':>7s} {'med_t':>7s}"
    print(hdr)
    print("-" * 84)
    for name, fn in POLS:
        c, cl, t = run(fn, 123000)
        row = f"  {name:>16s} "
        for T in (3.0, 5.0, 8.0):
            ok = (c >= 0.8 and cl <= 0.10 and (not np.isnan(t)) and t <= T)
            row += f"{'PASS' if ok else 'fail':>7s} "
        row += f"{cl:>6.0%} {t:>7.2f}"
        print(row)

    print()
    print("Look for a T where ckpt+rule PASS while every geom/loiter/zero FAIL.")


if __name__ == "__main__":
    main()
