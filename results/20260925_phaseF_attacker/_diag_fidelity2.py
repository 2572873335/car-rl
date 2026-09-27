"""Confirm the wrapper is faithful when driven the way TRAINING drives it.

The previous probe compared plain `env.step(a)` against a wrapper path where the
caller fetched `env.env._obs()` itself and then called `env.step([0])`. That
double-steps the observation: AttackerEnv.step() ALSO calls `self.env._obs()`
internally and hands THAT to the defender. So my probe fed the defender a
one-step-stale observation -- a bug in the probe.

Correct comparison: use AttackerEnv's own internal path (which is what training
uses), and drive the PLAIN env with the same observation the wrapper would give.
Both should then produce identical episodes.

Also: the earlier fidelity check used `wrapped(v_set_equiv, seeds, attacker=None)`
which passed a=0 every step -- that IS the training path, so if it really differs
from plain at the same leader speed something is wrong. This resolves which.
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


def plain_run(seed, n_ep=20):
    """D0-style: plain env, follower driven by the model on the env's own obs."""
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


def wrapped_run(seed, n_ep=20):
    """AttackerEnv's OWN path with an idle attacker -- exactly what training does."""
    fe.D_DES = 0.20
    m = load_v1()
    env = AttackerEnv(lambda o: m.predict(o, deterministic=True)[0], d_des=0.20)
    coll = 0
    for s in range(seed, seed + n_ep):
        env.reset(seed=s)
        while True:
            obs, r, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
            if term or trunc:
                break
        if env.env.term_reason == "collision":
            coll += 1
    return coll


if __name__ == "__main__":
    print("=" * 76)
    print("FAITHFULNESS, driven as TRAINING drives it")
    print("=" * 76)
    for seed in [2000, 60000]:
        p = plain_run(seed)
        w = wrapped_run(seed)
        print(f"  seeds {seed}+:  plain(collision) {p:>2d}/20   "
              f"wrapped(collision) {w:>2d}/20   "
              f"{'MATCH' if p == w else 'DIFFER'}")
