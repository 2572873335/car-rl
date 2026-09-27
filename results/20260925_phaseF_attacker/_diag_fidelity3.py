"""Pin the inner env's v_set/behavior in the wrapped path and re-compare.

The wrapped run leaves v_set/behavior at the env's random draw (0.356..0.472),
while the plain comparison sets v_set=0.50, behavior='constant'. The patched
_leader_v should make v_set irrelevant, so this probe tests that directly: if
pinning them makes the two paths agree, then something v_set-dependent is NOT
being overridden; if they still disagree, the cause lies elsewhere.
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


def wrapped_run(seed, pin, n_ep=20):
    fe.D_DES = 0.20
    m = load_v1()
    env = AttackerEnv(lambda o: m.predict(o, deterministic=True)[0], d_des=0.20)
    coll = 0
    for s in range(seed, seed + n_ep):
        env.reset(seed=s)
        if pin:
            env.env.v_set, env.env.behavior = 0.50, "constant"
        while True:
            obs, r, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
            if term or trunc:
                break
        if env.env.term_reason == "collision":
            coll += 1
    return coll


def plain_run(seed, n_ep=20):
    fe.D_DES = 0.20
    m = load_v1()
    env = fe.FollowEnv(domain_randomize=False)
    coll = 0
    for s in range(seed, seed + n_ep):
        obs, _ = env.reset(seed=s)
        env.v_set, env.behavior = 0.50, "constant"
        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if env.term_reason == "collision":
            coll += 1
    return coll


if __name__ == "__main__":
    print("=" * 80)
    print("EFFECT OF PINNING v_set/behavior INSIDE THE WRAPPER")
    print("=" * 80)
    for seed in [2000, 60000]:
        p = plain_run(seed)
        wu = wrapped_run(seed, pin=False)
        wp = wrapped_run(seed, pin=True)
        print(f"  seeds {seed}+")
        print(f"    plain (v_set=0.50 pinned)        : {p:>2d}/20")
        print(f"    wrapped, NOT pinned             : {wu:>2d}/20")
        print(f"    wrapped, v_set=0.50 pinned      : {wp:>2d}/20")
