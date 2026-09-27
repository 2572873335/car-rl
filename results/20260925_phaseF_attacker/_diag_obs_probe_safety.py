"""Settle it: does AttackerEnv.step() (the TRAINING path) hand the defender a
correct obs[1], or a stale zero?

The full-obs trace shows the WRAPPED probe sees obs[1] = 0.0 always, while plain
sees -0.056. But those probes fetch obs differently:
  * wrapped probe:  acting = env.env._obs()   <-- called as a SIDE probe
  * AttackerEnv.step(): d_obs = self.env._obs() <-- also a side probe, then step()

If _obs() has a side effect ordering issue (it reads self.prev_e which step()
updates), then BOTH read the same stale value -- meaning the TRAINING path is
also feeding the defender a wrong obs[1], and the whole Step 1 run was trained
against a partially-blind defender.

This measures _obs()[1] before and after a step, and compares the value training
actually uses against what the plain env would return post-step.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv


def obs_de(env):
    return float(env._obs()[1])


if __name__ == "__main__":
    fe.D_DES = 0.20
    print("=" * 88)
    print("IS _obs() SAFE TO CALL AS A SIDE PROBE?  (it reads self.prev_e)")
    print("=" * 88)

    env = fe.FollowEnv(domain_randomize=False)
    print("\n  PLAIN env (correct usage: obs from step() return)")
    o, _ = env.reset(seed=2000)
    env.v_set, env.behavior = 0.50, "constant"
    print(f"    reset: obs[1] = {o[1]:+.6f}")
    for k in range(4):
        o, _, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
        print(f"    step {k}: obs[1] = {o[1]:+.6f}   (prev_e now {env.prev_e:+.6f})")

    env2 = fe.FollowEnv(domain_randomize=False)
    print("\n  SIDE-PROBE usage: _obs() called separately before step()")
    o2, _ = env2.reset(seed=2000)
    env2.v_set, env2.behavior = 0.50, "constant"
    print(f"    reset: _obs()[1] = {obs_de(env2):+.6f}")
    for k in range(4):
        side = obs_de(env2)                 # side probe BEFORE the step
        o2, _, _, _, _ = env2.step(np.array([1.0], dtype=np.float32))
        print(f"    step {k}: side-probe _obs()[1] = {side:+.6f}   "
              f"step-return obs[1] = {o2[1]:+.6f}")

    print("\n  => if the side-probe column is not equal to the step-return column,")
    print("     then calling _obs() as a probe gives a DIFFERENT (stale) value,")
    print("     and AttackerEnv.step() -- which does exactly that -- feeds the")
    print("     defender a wrong obs[1].")
