"""Locate the unfaithfulness: trace the same episode through both paths and diff.

Wrapper-fidelity check showed PLAIN env and AttackerEnv disagree by 16-18/20
collisions at the SAME leader speed and seeds. This traces one episode through
each and prints the per-step quantities, so the divergence point is visible
instead of inferred.

Key suspects to check (each printed):
  * leader speed actually used
  * follower speed
  * gap
  * the action handed to the follower
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv, V_BASE, V_SPAN

_V1 = None


def load_v1():
    global _V1
    if _V1 is None:
        from stable_baselines3 import PPO
        _V1 = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    return _V1


def trace_plain(seed=2000, n=12):
    fe.D_DES = 0.20
    m = load_v1()
    env = fe.FollowEnv(domain_randomize=False)
    obs, _ = env.reset(seed=seed)
    env.v_set, env.behavior = 0.50, "constant"
    rows = []
    for k in range(n):
        a, _ = m.predict(obs, deterministic=True)
        vl = env._leader_v()
        obs, r, term, trunc, _ = env.step(a)
        rows.append((round(env.t, 3), round(vl, 4), round(float(a[0]), 4),
                     round(env.follower.v, 4), round(env.leader.v, 4),
                     round(env.log["gap"][-1], 4)))
        if term or trunc:
            break
    return rows


def trace_wrapped(seed=2000, n=12):
    fe.D_DES = 0.20
    m = load_v1()
    pol = lambda o: m.predict(o, deterministic=True)[0]
    env = AttackerEnv(pol, d_des=0.20)
    env.reset(seed=seed)
    rows = []
    for k in range(n):
        vl_used = env.env._leader_v()
        obs, r, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        rows.append((round(env.env.t, 3), round(vl_used, 4),
                     round(env.env._attacker_v, 4),
                     round(env.env.follower.v, 4), round(env.env.leader.v, 4),
                     round(env.env.log["gap"][-1], 4)))
        if term or trunc:
            break
    return rows


if __name__ == "__main__":
    print("=" * 96)
    print("PLAIN (t, v_l, action, v_follower, v_leader, gap)")
    print("=" * 96)
    for r in trace_plain():
        print(f"   {r}")
    print()
    print("=" * 96)
    print("WRAPPED (t, v_l_used, _attacker_v, v_follower, v_leader, gap)")
    print("=" * 96)
    for r in trace_wrapped():
        print(f"   {r}")
    print()
    print(f"  V_BASE={V_BASE} V_SPAN={V_SPAN} -> idle attacker commands leader "
          f"{V_BASE + V_SPAN*0.0:.3f}")
