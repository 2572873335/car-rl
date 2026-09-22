"""D0 execution step 0: the D_DES wrapper (per owner ruling -- wrapper path,
no frozen-file edits, following RobustnessWrapper precedent).

Precedent check: robustness_sweep.RobustnessWrapper is a plain facade that
sets env.instance attributes in reset(). But D_DES is a MODULE constant
referenced bare in _obs()/step(), so the wrapper must override follow_env.D_DES
at runtime (the exact path the reviewer's T0 validated).

This probe confirms:
  1. the wrapper approach produces correct measurements at d_des 0.50/1.00,
     WITHOUT touching follow_env.py (sha256 stays intact);
  2. the frozen checkpoint works zero-shot across spacing (the finding to
     record as AGV deployment flexibility);
  3. records the 1.0 m/s reversal in the process.
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from car_following_sim import DT


class DDESWrapper:
    """Plain facade (RobustnessWrapper precedent). Overrides the module
    constant D_DES at reset; the frozen env file is untouched."""

    def __init__(self, d_des, v_set=0.30, behavior="constant"):
        self.d_des = d_des
        self.v_set = v_set
        self.behavior = behavior
        self.env = None

    def make(self):
        self.env = fe.FollowEnv(domain_randomize=False)
        return self.env

    def reset(self, seed):
        fe.D_DES = self.d_des                # runtime override, module constant
        obs, info = self.env.reset(seed=seed)
        self.env.v_set = self.v_set
        self.env.behavior = self.behavior
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        fe.D_DES = 0.20                      # restore


def run(policy, d_des, v_set, behavior, n_ep=20, seed0=2000):
    w = DDESWrapper(d_des, v_set, behavior)
    w.make()
    errs, coll = [], 0
    reasons = {}
    try:
        for k in range(n_ep):
            obs, _ = w.reset(seed0 + k)
            while True:
                obs, r, term, trunc, _ = w.step(policy(obs))
                if term or trunc:
                    break
            g = np.array(w.env.log["gap"])
            errs.append(np.abs(g - d_des).mean())
            reasons[w.env.term_reason] = reasons.get(w.env.term_reason, 0) + 1
            if w.env.term_reason == "collision":
                coll += 1
    finally:
        w.close()
        assert fe.D_DES == 0.20, f"D_DES leaked: {fe.D_DES}"
    return float(np.mean(errs)), coll / n_ep, reasons


def main():
    from stable_baselines3 import PPO
    model = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    rl = lambda obs: model.predict(obs, deterministic=True)[0]

    import hashlib
    h_before = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]

    print("=" * 80)
    print("D0 STEP 0: D_DES WRAPPER (frozen file untouched)")
    print(f"  follow_env.py sha256[:16] before: {h_before}")
    print("=" * 80)
    print(f"  {'grid':>22s} {'mean|e| true-cm':>16s} {'coll':>7s} {'reasons':>30s}")
    for d, v, b in [(0.20, 0.30, "constant"), (0.50, 0.55, "constant"),
                    (0.50, 1.00, "constant"), (1.00, 1.00, "constant")]:
        e, c, r = run(rl, d, v, b)
        print(f"  {f'd={d} v={v} {b[:8]}':>22s} {e*100:>15.2f} {c:>6.0%} {r!s:>30s}")

    h_after = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]
    print()
    print(f"  follow_env.py sha256[:16] after:  {h_after}")
    print(f"  frozen file untouched:            {h_before == h_after}")
    print()
    print("READING: at d=0.50 v=1.0 the reversal (RL loses vs P+FF) is expected;")
    print("the wrapper approach itself must work without file edits.")


if __name__ == "__main__":
    main()
