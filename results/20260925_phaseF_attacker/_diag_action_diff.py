"""Print action + follower speed step by step for both paths until divergence.

The 22-step trace matched; the wrapped episode collides at step 62 while plain
times out at 1001. Gap closes fast in wrapped (0.61 -> 0.12 in 1.24 s) so the
DEFENDER'S ACTION must differ. This prints the action in both paths.
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


def plain_rows(seed, n=70):
    fe.D_DES = 0.20
    m = load_v1()
    env = fe.FollowEnv(domain_randomize=False)
    obs, _ = env.reset(seed=seed)
    env.v_set, env.behavior = 0.50, "constant"
    rows = []
    for k in range(n):
        a, _ = m.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        rows.append((k, round(float(obs[0]), 5), round(float(a[0]), 5),
                     round(env.follower.v, 5), round(env.log["gap"][-1], 5)))
        if term or trunc:
            break
    return rows


def wrapped_rows(seed, n=70):
    fe.D_DES = 0.20
    m = load_v1()
    env = AttackerEnv(lambda o: m.predict(o, deterministic=True)[0], d_des=0.20)
    env.reset(seed=seed)
    rows = []
    for k in range(n):
        d_obs = env.env._obs()
        a = m.predict(d_obs, deterministic=True)[0]
        obs, r, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        rows.append((k, round(float(d_obs[0]), 5), round(float(a[0]), 5),
                     round(env.env.follower.v, 5),
                     round(env.env.log["gap"][-1], 5)))
        if term or trunc:
            break
    return rows


if __name__ == "__main__":
    seed = 2000
    p = plain_rows(seed)
    w = wrapped_rows(seed)
    print(f"  {'k':>4s} | {'plain: e0':>9s} {'a':>8s} {'vf':>7s} {'gap':>8s}"
          f" | {'wrapped: e0':>11s} {'a':>8s} {'vf':>7s} {'gap':>8s}")
    for i in range(min(len(p), len(w))):
        kp, e0p, ap, vfp, gp = p[i]
        kw, e0w, aw, vfw, gw = w[i]
        flag = "  <<<" if abs(ap - aw) > 1e-5 else ""
        print(f"  {kp:>4d} | {e0p:>9.5f} {ap:>8.4f} {vfp:>7.4f} {gp:>8.5f}"
              f" | {e0w:>11.5f} {aw:>8.4f} {vfw:>7.4f} {gw:>8.5f}{flag}")
        if flag:
            print(f"  => divergence at step {i}: plain len {len(p)}, wrapped len {len(w)}")
            break
