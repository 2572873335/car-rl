"""Compare the FULL observation vector between paths at the divergence point.

Earlier I compared only obs[2] and concluded "same obs, different action". That was
invalid: the action depends on all four slots. Step 40 of the plain path is
[0.4332, -0.375, 0.92308, 0.15734]; the wrapped path's step 40 evidently differs
in obs[1] and/or obs[3]. This prints both full vectors at the divergence.
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


def plain_trace(seed, n=60):
    fe.D_DES = 0.20
    m = load_v1()
    env = fe.FollowEnv(domain_randomize=False)
    obs, _ = env.reset(seed=seed)
    env.v_set, env.behavior = 0.50, "constant"
    out = []
    for k in range(n):
        acting = obs.copy()
        a, _ = m.predict(acting, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        out.append((acting, float(a[0])))
        if term or trunc:
            break
    return out


def wrapped_trace(seed, n=60):
    fe.D_DES = 0.20
    m = load_v1()
    env = AttackerEnv(lambda o: m.predict(o, deterministic=True)[0], d_des=0.20)
    env.reset(seed=seed)
    out = []
    for k in range(n):
        acting = env.env._obs().copy()
        a = m.predict(acting, deterministic=True)[0]
        env.step(np.array([0.0], dtype=np.float32))
        out.append((acting, float(a[0])))
        if env.env.term_reason != "timeout":
            break
    return out


if __name__ == "__main__":
    p = plain_trace(2000)
    w = wrapped_trace(2000)
    n = min(len(p), len(w))
    print(f"  plain len {len(p)}  wrapped len {len(w)}")
    print(f"  {'k':>4s}  {'full obs (plain)':>40s} {'a_p':>8s}"
          f"  {'full obs (wrapped)':>40s} {'a_w':>8s}")
    for i in range(min(n, 45)):
        po, pa = p[i]
        wo, wa = w[i]
        same = np.allclose(po, wo, atol=1e-6)
        if i >= 36 or not same:
            print(f"  {i:>4d}  {str(np.round(po,5)):>40s} {pa:>8.4f}"
                  f"  {str(np.round(wo,5)):>40s} {wa:>8.4f}"
                  f"{'' if same else '  <<< DIVERGE'}")
        if not same:
            break
