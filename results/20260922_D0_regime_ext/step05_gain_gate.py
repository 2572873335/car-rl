"""Step 0.5 gate for the speed-adaptive-gain experiment (H-D0.8).

Two things to verify before spending a 5M-step run:

  1. FEASIBILITY -- does the gain schedule actually reach the SubprocVecEnv
     workers, and does the widened v_set band too? (Same failure mode that
     killed the first wrapper: a plain object is rejected by SB3 and every
     worker dies at construction.)

  2. SELF-TEST -- does this gate actually report FAILURE when something is
     broken? Per the owner's ruling, a validator that cannot demonstrate it
     fires on an injected fault is not a validator. So the gate deliberately
     injects faults and asserts it catches them:
       (a) fake wrapper whose get_gain() always returns g_low -> the
           "schedule has effect" check must fail;
       (b) a broken env_fn that raises in the worker -> the subprocess check
           must fail rather than pass.

Exit code is non-zero unless every check passes AND every injected fault is
detected.

Run: uv run python results/20260922_D0_regime_ext/step05_gain_gate.py
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np


def check_workers(venv):
    """Fraction of child samples above 0.50 -- must be >0 for the band to be wide."""
    vals = []
    for _ in range(30):
        venv.reset()
        vals.extend(venv.env_method("get_v_set"))
    vals = np.array(vals)
    return float((vals > 0.50).mean()), vals


def check_gain_effect(venv):
    """Gains must actually differ between low-speed and high-speed workers."""
    venv.reset()
    gains = venv.env_method("get_gain")
    vsets = venv.env_method("get_v_set")
    lo = [g for g, v in zip(gains, vsets) if v <= 0.50]
    hi = [g for g, v in zip(gains, vsets) if v > 0.50]
    return lo, hi


def main():
    from gain_sched_env import make_env, GainSchedWrapper
    import follow_env as fe
    from stable_baselines3.common.vec_env import SubprocVecEnv

    results = {}
    print("=" * 88)
    print("STEP 0.5 GATE -- speed-adaptive gain (H-D0.8)")
    print("=" * 88)

    # ---- 1. real configuration ----
    n = 8
    venv = SubprocVecEnv([make_env(i, "wide") for i in range(n)])
    frac_hi, vals = check_workers(venv)
    print(f"\n[1] workers={n}, child v_set samples={len(vals)}")
    print(f"    min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f}")
    print(f"    fraction > 0.50 : {frac_hi:.1%}  "
          f"(nonzero proves the wide band reached the children)")
    results["1_band_reaches_workers"] = frac_hi > 0.1

    lo, hi = check_gain_effect(venv)
    print(f"\n[2] gain schedule active in workers:")
    print(f"    v_set<=0.50 -> gains {sorted(set(round(g,3) for g in lo))}")
    print(f"    v_set> 0.50 -> gains {sorted(set(round(g,3) for g in hi))}")
    distinct = (len(set(round(g, 3) for g in lo + hi)) > 1)
    print(f"    gains differ across regimes: {distinct}")
    results["2_gain_schedule_effective"] = distinct
    venv.close()

    # ---- 2. FAULT INJECTION: (a) gain schedule neutered ----
    print("\n[3] FAULT INJECTION (a): neutered schedule (g_high == g_low)")

    class Neutered(GainSchedWrapper):
        def _gain(self):
            return self.g_low          # ignore the schedule entirely

    def neutered_env(i):
        def _init():
            e = fe.FollowEnv(domain_randomize=True, gap_max=1.2)
            e = Neutered(e, v_lo=0.25, v_hi=1.00, g_low=1.0, g_high=0.4,
                         seed=40_000 + i)
            e.reset(seed=40_000 + i)
            return e
        return _init

    vn = SubprocVecEnv([neutered_env(i) for i in range(n)])
    lo2, hi2 = check_gain_effect(vn)
    distinct2 = (len(set(round(g, 3) for g in lo2 + hi2)) > 1)
    vn.close()
    caught_a = (distinct2 is False)
    print(f"    gains differ? {distinct2}  -> gate catches it: {caught_a}")
    results["3_fault_a_detected"] = caught_a

    # ---- 3. FAULT INJECTION: (b) worker raises ----
    print("\n[4] FAULT INJECTION (b): a worker that raises on construction")

    def broken_env(i):
        def _init():
            if i == 3:
                raise RuntimeError("injected worker fault")
            e = fe.FollowEnv(domain_randomize=True)
            e.reset(seed=50_000 + i)
            return e
        return _init

    caught_b = False
    try:
        vb = SubprocVecEnv([broken_env(i) for i in range(n)])
        vb.reset()
        vb.close()
    except Exception as e:
        caught_b = True
        print(f"    raised as expected: {type(e).__name__}")
    print(f"    gate catches it: {caught_b}")
    results["4_fault_b_detected"] = caught_b

    print("\n" + "=" * 88)
    for k, v in results.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print(f"\nGATE: {'PASS' if ok else 'FAIL'}"
          f"{'' if ok else ' -- DO NOT TRAIN'}")
    print("=" * 88)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
