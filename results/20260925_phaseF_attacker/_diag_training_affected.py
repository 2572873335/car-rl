"""Was the TRAINING path affected, or only my probes?

Critical distinction. AttackerEnv.step() does:

    d_obs = self.env._obs()          # <- side probe, before step()
    d_act = self.defender(d_obs)
    obs, ... = self.env.step(d_act)

By the analysis, d_obs[1] is ALWAYS 0 (prev_e was just set to the current e by the
previous step's line 182). If so, the defender inside training saw de == 0 at
every step -- it was trained/queried blind to gap rate.

Both frozen defenders (P+FF and v1) consume obs[1]:
  * v1 RL: obs[1] is input slot 2 of a 4-input MLP -> its policy output changes.
  * P+FF: baseline_action uses de explicitly:
        a = (kp*e - (0.0 if use_ff else de)) / ACT_GAIN
    so with de == 0 the P+FF branch degenerates to a = kp*e/ACT -- pure P with the
    feed-forward term silently zeroed.

This measures the consequence directly: evaluate P+FF and v1 INSIDE AttackerEnv
(de == 0 path) vs in the plain env (correct de), same seeds.
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


def check_de_inside_step(seed=2000):
    """Instrument AttackerEnv.step to report the obs[1] it hands the defender."""
    fe.D_DES = 0.20
    m = load_v1()
    seen = []
    def spy_defender(o):
        seen.append(float(o[1]))
        return m.predict(o, deterministic=True)[0]
    env = AttackerEnv(spy_defender, d_des=0.20)
    env.reset(seed=seed)
    for _ in range(50):
        env.step(np.array([0.0], dtype=np.float32))
        if env.env.term_reason != "timeout":
            break
    return seen


def plain_de(seed=2000, n=50):
    fe.D_DES = 0.20
    m = load_v1()
    env = fe.FollowEnv(domain_randomize=False)
    obs, _ = env.reset(seed=seed)
    env.v_set, env.behavior = 0.50, "constant"
    seen = []
    for _ in range(n):
        seen.append(float(obs[1]))
        a, _ = m.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return seen


if __name__ == "__main__":
    a = check_de_inside_step()
    b = plain_de()
    print("=" * 84)
    print("obs[1] THE DEFENDER ACTUALLY RECEIVED")
    print("=" * 84)
    print(f"  inside AttackerEnv (training path) : n={len(a)}")
    print(f"    first 12: {[round(x,4) for x in a[:12]]}")
    print(f"    all zero? {all(abs(x) < 1e-12 for x in a)}")
    print()
    print(f"  plain env (correct)                : n={len(b)}")
    print(f"    first 12: {[round(x,4) for x in b[:12]]}")
    print(f"    all zero? {all(abs(x) < 1e-12 for x in b)}")
