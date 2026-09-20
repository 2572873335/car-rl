"""Step 0: do the M1 criteria actually DISCRIMINATE?

Both the W5 probe and the W6 v1 plan failed the same way: the pass/fail
criteria were never validated, so degenerate policies satisfied them
(win-rate band: do-nothing 0.52, random 0.52).

This script applies the same treatment to the M1 criteria I just proposed:
  M1.1 stability : final-20% collision std <= 15%
  M1.2 absolute  : final-20% collision rate <= 10%

A criterion is only usable if known-BAD behaviour FAILS it. Test against
three reference policies whose quality we already know, plus the recorded
W5/V2V training series (known-unstable).
"""
import re
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World

W5_JOINT = "/home/zy/car_rl/code0919/results/20260920_phaseC_probe/probe_run2_joint_selfplay_trend_raw.txt"
V2V_JOINT = "/home/zy/car_rl/code0919/results/20260920_phaseC_probe/probe_run4_v2v_prior_transfer_raw.txt"


def final_window_from_series(rows):
    """rows = list of (steps, coll_frac). Criteria use the last 20% of steps."""
    if not rows:
        return None, None
    hi = rows[-1][0]
    lo = hi * 0.8
    sel = [c for s, c in rows if s >= lo]
    return float(np.mean(sel)), float(np.std(sel))


def load_w5(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.match(r"\s*([\d,]+)\s+(-?[\d.]+)\s+([\d.]+)%", ln)
            if m:
                rows.append((int(m.group(1).replace(",", "")),
                             float(m.group(3)) / 100.0))
    return rows


def load_v2v(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.search(r"\[jointV2V\]\s+steps=\s*([\d,]+)\s+ret=\s*(-?[\d.]+)\s+coll=\s*([\d.]+)%", ln)
            if m:
                rows.append((int(m.group(1).replace(",", "")),
                             float(m.group(3)) / 100.0))
    return rows


def policy_collision_rate(fn, n_ep=60, seed0=51000):
    """Collision rate of (fn as both cars) in the W5-style world."""
    coll = 0
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            obs, _, done = w.step(np.stack([fn(obs[0]), fn(obs[1])]))
            if done:
                break
        if w.reason == "collision":
            coll += 1
    return coll / n_ep


def main():
    STD_MAX, MEAN_MAX = 0.15, 0.10

    print("=" * 76)
    print("M1 CRITERIA DISCRIMINATION CHECK")
    print(f"  M1.1 stability: final-20% collision std <= {STD_MAX:.0%}")
    print(f"  M1.2 absolute : final-20% collision mean <= {MEAN_MAX:.0%}")
    print("=" * 76)

    print()
    print("A. Against RECORDED training series (known-bad should FAIL)")
    print(f"  {'series':>28s} {'final mean':>11s} {'final std':>10s} "
          f"{'M1.1':>6s} {'M1.2':>6s}")
    for label, rows in [("W5 joint (self-relative)", load_w5(W5_JOINT)),
                        ("V2V joint (opponent-relative)", load_v2v(V2V_JOINT))]:
        m, s = final_window_from_series(rows)
        if m is None:
            print(f"  {label:>28s}   (no data)")
            continue
        p1 = "PASS" if s <= STD_MAX else "FAIL"
        p2 = "PASS" if m <= MEAN_MAX else "FAIL"
        print(f"  {label:>28s} {100*m:>10.1f}% {100*s:>9.1f}% {p1:>6s} {p2:>6s}")
    print("  => both known-unstable series should FAIL at least M1.2.")

    print()
    print("B. Against DEGENERATE policies (must FAIL M1.2)")
    rng = np.random.default_rng(9)
    for label, fn in [("zero action (both cars)", lambda o: np.zeros(2)),
                      ("random (both cars)", lambda o: rng.uniform(-1, 1, 2)),
                      ("full throttle (both)", lambda o: np.array([1.0, 0.0]))]:
        r = policy_collision_rate(fn)
        p2 = "PASS" if r <= MEAN_MAX else "FAIL"
        print(f"  {label:>28s}  collision rate = {100*r:>6.1f}%   M1.2: {p2}")

    print()
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    print("  Criteria are USABLE only if every degenerate/known-bad case FAILS.")
    print("  Compare with the REJECTED v1 criterion (win rate in [35%,65%]),")
    print("  where do-nothing scored 0.52 = PASS -- i.e. no discrimination.")
    print()
    print("  Unlike a win-rate band, an ABSOLUTE collision-rate cap cannot be")
    print("  satisfied by inaction: standing still is not a low-collision")
    print("  strategy here because episodes are scored on relative outcome and")
    print("  the do-nothing/random pairs collide or fail to complete.")
    print()
    print("  NOTE: if any degenerate policy PASSES M1.2, the threshold 10% is")
    print("  too loose and must be tightened BEFORE training (per plan S3).")


if __name__ == "__main__":
    main()
