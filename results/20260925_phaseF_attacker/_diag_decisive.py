"""Decisive: trace BOTH paths at the SAME step index and print the first gap
divergence, plus every quantity that could differ.

Prior probes compared misaligned indices. This one records, at each step k, the
observation each defender ACTS ON, and the gap AFTER that step, for both paths.
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


def plain_trace(seed, n=200):
    fe.D_DES = 0.20
    m = load_v1()
    env = fe.FollowEnv(domain_randomize=False)
    obs, _ = env.reset(seed=seed)
    env.v_set, env.behavior = 0.50, "constant"
    out = []
    for k in range(n):
        acting_obs = obs.copy()
        a, _ = m.predict(acting_obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        out.append((acting_obs, float(a[0]), env.log["gap"][-1], env.term_reason))
        if term or trunc:
            break
    return out


def wrapped_trace(seed, n=200):
    fe.D_DES = 0.20
    m = load_v1()
    env = AttackerEnv(lambda o: m.predict(o, deterministic=True)[0], d_des=0.20)
    env.reset(seed=seed)
    out = []
    for k in range(n):
        acting_obs = env.env._obs().copy()
        a = m.predict(acting_obs, deterministic=True)[0]
        env.step(np.array([0.0], dtype=np.float32))
        out.append((acting_obs, float(a[0]), env.env.log["gap"][-1],
                    env.env.term_reason))
        if env.env.term_reason != "timeout":
            break
    return out


if __name__ == "__main__":
    p = plain_trace(2000)
    w = wrapped_trace(2000)
    print(f"  plain len {len(p)}   wrapped len {len(w)}")
    print(f"  {'k':>4s} {'plain_obs[2]':>13s} {'plain_a':>8s} {'plain_gap':>10s}"
          f" | {'wr_obs[2]':>10s} {'wr_a':>8s} {'wr_gap':>10s}")
    for i in range(min(len(p), len(w))):
        po, pa, pg, _ = p[i]
        wo, wa, wg, _ = w[i]
        mark = ""
        if abs(po[2] - wo[2]) > 1e-6 or abs(pa - wa) > 1e-6:
            mark = "  <<< DIVERGE"
        if i < 6 or mark:
            print(f"  {i:>4d} {po[2]:>13.5f} {pa:>8.4f} {pg:>10.5f}"
                  f" | {wo[2]:>10.5f} {wa:>8.4f} {wg:>10.5f}{mark}")
        if mark:
            print(f"  => first divergence at step {i}")
            break
