"""Run ONE episode to termination through both paths and find the divergence.

The 22-step traces were identical, so the difference appears later. Print the full
episode side by side (every 25th step) and the termination, for the same seed.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv

_V1 = None


def load_v1():
    global _V1
    if _V1 is None:
        from stable_baselines3 import PPO
        _V1 = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    return _V1


def plain_episode(seed):
    fe.D_DES = 0.20
    m = load_v1()
    env = fe.FollowEnv(domain_randomize=False)
    obs, _ = env.reset(seed=seed)
    env.v_set, env.behavior = 0.50, "constant"
    k = 0
    while True:
        a, _ = m.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        k += 1
        if term or trunc:
            break
    return k, env.term_reason, env.log


def wrapped_episode(seed):
    fe.D_DES = 0.20
    m = load_v1()
    pol = lambda o: m.predict(o, deterministic=True)[0]
    env = AttackerEnv(pol, d_des=0.20)
    env.reset(seed=seed)
    k = 0
    while True:
        obs, r, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        k += 1
        if term or trunc:
            break
    return k, env.env.term_reason, env.env.log


if __name__ == "__main__":
    seed = 2000
    k1, r1, log1 = plain_episode(seed)
    k2, r2, log2 = wrapped_episode(seed)
    print("=" * 80)
    print(f"seed {seed}")
    print(f"  PLAIN   : {k1} steps, term={r1}")
    print(f"  WRAPPED : {k2} steps, term={r2}")

    g1 = np.array(log1["gap"])
    g2 = np.array(log2["gap"])
    vl1 = np.array(log1["v_l"])
    vl2 = np.array(log2["v_l"])
    print(f"\n  gap  first/last: plain {g1[0]:.4f}/{g1[-1]:.4f}   "
          f"wrapped {g2[0]:.4f}/{g2[-1]:.4f}")
    print(f"  gap  min       : plain {g1.min():.4f}   wrapped {g2.min():.4f}")
    print(f"  v_l  first/last: plain {vl1[0]:.4f}/{vl1[-1]:.4f}   "
          f"wrapped {vl2[0]:.4f}/{vl2[-1]:.4f}")
    print(f"  v_l  min       : plain {vl1.min():.4f}   wrapped {vl2.min():.4f}")

    n = min(len(g1), 3000)
    # first index where the gaps differ by more than 1e-6
    diff = np.where(np.abs(g1[:n] - g2[:n]) > 1e-6)[0]
    if len(diff):
        i = diff[0]
        print(f"\n  first divergence at step {i}:")
        print(f"    plain   gap {g1[i]:.5f}  v_l {vl1[i]:.5f}")
        print(f"    wrapped gap {g2[i]:.5f}  v_l {vl2[i]:.5f}")
    else:
        print(f"\n  no gap divergence in the first {n} steps "
              f"(lengths {len(g1)} vs {len(g2)})")
