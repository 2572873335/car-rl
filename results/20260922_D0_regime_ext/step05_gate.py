"""Step 0.5 gate (owner-required, plan section 8.1.1): does the widened v_set
actually reach the SubprocVecEnv WORKER processes?

This is the failure mode that would waste a 5M-step run: the override holds in
the parent but the children construct their own FollowEnv and sample
U(0.25,0.50) as usual, so training silently uses the old distribution while the
logs look fine.

Checks:
  1. single env: does the wrapper's v_set draw land in [0.25, 1.00]?
  2. SubprocVecEnv with the same make_env: do the CHILD samples land there too,
     and is the realized distribution actually wider than U(0.25,0.50)?
  3. do the samples stay <= 1.0 (the leader clip) -- i.e. is the range we
     inject actually realizable?

The verdict is conjunctive: ALL checks must pass. An earlier version keyed the
verdict on check 1 alone and printed PASS while check 2 was crashing -- a gate
that passes when a required check failed is worse than no gate.

Run:  uv run python results/20260922_D0_regime_ext/step05_gate.py
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np


def main():
    from regime_ext_env import RegimeExtWrapper, make_env
    import follow_env as fe

    results = {}
    print("=" * 82)
    print("STEP 0.5 GATE: does the widened v_set reach the workers?")
    print("=" * 82)

    # ---- 1. single env ----
    env = fe.FollowEnv(domain_randomize=True)
    w = RegimeExtWrapper(env, v_lo=0.25, v_hi=1.00, seed=0)
    w.reset(seed=1)
    singles = []
    for k in range(200):
        w.reset(seed=100 + k)
        singles.append(w.get_v_set())
    singles = np.array(singles)
    frac_hi = float((singles > 0.50).mean())
    print(f"\n[1] single env, 200 resets:")
    print(f"    v_set min={singles.min():.3f} max={singles.max():.3f} "
          f"mean={singles.mean():.3f}")
    print(f"    inside [0.25,1.00]: {((singles >= 0.25) & (singles <= 1.00)).mean():.1%}")
    print(f"    fraction > 0.50 (impossible under the frozen default): {frac_hi:.1%}")
    results["1_single_env"] = frac_hi > 0.1

    # ---- 2. SubprocVecEnv, same construction as training ----
    from stable_baselines3.common.vec_env import SubprocVecEnv
    n = 8
    try:
        venv = SubprocVecEnv([make_env(i) for i in range(n)])
        venv.reset()
        child_vals = []
        for _ in range(40):
            venv.reset()
            child_vals.extend(venv.env_method("get_v_set"))
        child_vals = np.array(child_vals)
        frac_child = float((child_vals > 0.50).mean())
        print(f"\n[2] SubprocVecEnv, {n} workers, {len(child_vals)} child samples:")
        print(f"    v_set min={child_vals.min():.3f} max={child_vals.max():.3f} "
              f"mean={child_vals.mean():.3f}")
        print(f"    inside [0.25,1.00]: "
              f"{((child_vals >= 0.25) & (child_vals <= 1.00)).mean():.1%}")
        print(f"    fraction > 0.50: {frac_child:.1%}  "
              f"<-- nonzero proves the override reached the children")
        results["2_subproc"] = frac_child > 0.1
        venv.close()
    except Exception as e:
        print(f"\n[2] SubprocVecEnv FAILED: {type(e).__name__}: {e}")
        results["2_subproc"] = False

    # ---- 3. realizable? ----
    peaks = np.minimum(singles * 1.0, 1.0)
    print(f"\n[3] realizability: leader peak max={peaks.max():.3f} "
          f"(clip=1.0, V_MAX={fe.V_MAX})")
    print(f"    any leader peak > V_MAX? {(peaks > fe.V_MAX).any()}")
    results["3_realizable"] = bool(not (peaks > fe.V_MAX).any())

    print("\n" + "=" * 82)
    for k, v in results.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    ok = all(results.values())
    print(f"\nGATE: {'PASS' if ok else 'FAIL'} -- "
          f"{'widened band is in effect' if ok else 'DO NOT TRAIN: a required check failed'}")
    print("=" * 82)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
