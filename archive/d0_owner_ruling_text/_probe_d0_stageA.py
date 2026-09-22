"""Probe D0's load-bearing technical claim WITHOUT touching frozen files.

Claim: we can evaluate the frozen RL checkpoint at d_des = 0.50/1.00 m
("zero-shot, measurement-only") by overriding the module constant D_DES at
runtime, without editing follow_env.py (so iron rule 2 is preserved).

Test:
  1. Does overriding follow_env.D_DES at runtime actually change the measured
     gap error vs the 0.20 baseline? (If it does, the experiment is meaningful.)
  2. Is the frozen checkpoint's observation encoding consistent across d_des?
     The policy was trained at D_DES=0.20; its obs is e = (gap - D_DES)/0.5.
     At D_DES=1.0, the same gap now maps to a different e -- is that a genuine
     re-point of the task, or does the policy just see nonsense it was never
     trained on?
  3. The physics: at 1.0 m/s, is 0.20 m spacing even reachable, or is a crash
     inevitable (validating plan section 2.3)?

Run the frozen checkpoint via the rule-baseline eval harness if available, or a
minimal rollout, at d_des in {0.20, 0.50, 1.00} and v_set in {0.3, 0.55, 1.0}.
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np


def eval_at(d_des, n_ep=10, seed0=2000):
    """Override D_DES at runtime, run the frozen RL checkpoint, report metrics."""
    import follow_env as fe
    from stable_baselines3 import PPO

    # runtime override -- no file edit
    fe.D_DES = d_des
    # v_set lives in the env's reset as r.uniform(0.25, 0.50); we can't easily
    # change it without a file edit, so we approximate by NOT relying on it and
    # instead measuring the achieved gap under the fixed controller.

    env = fe.FollowEnv(domain_randomize=False)
    model = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")

    gaps = []
    for k in range(n_ep):
        obs, _ = env.reset(seed=seed0 + k)
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        # final gap is in env.log
        gaps.append(env.log["gap"][-1] if env.log["gap"] else np.nan)
    fe.D_DES = 0.20  # restore
    return np.mean([g for g in gaps if g == g])


def main():
    print("=" * 78)
    print("D0 STAGE-A PROBE: runtime D_DES override, frozen checkpoint")
    print("=" * 78)
    for d in [0.20, 0.50, 1.00]:
        try:
            m = eval_at(d)
            print(f"  d_des={d:.2f} m -> terminal gap mean = {m:.3f} m")
        except Exception as e:
            print(f"  d_des={d:.2f} m -> ERROR: {type(e).__name__}: {e}")

    print()
    print("READING")
    print("  If terminal gap does NOT track d_des (i.e. stays ~0.20), then")
    print("  runtime override does not re-point the task and Stage A is void.")
    print("  If it tracks, Stage A may be meaningful -- but distinguish")
    print("  'genuine re-point' from 'the policy sees out-of-distribution obs'.")


if __name__ == "__main__":
    main()
