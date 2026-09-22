"""Discriminating test: is the stage-2 std divergence caused by the HIGH-SPEED
regime itself, or by the BAND WIDTH?

Established so far (commit fde106e + the v3 run):
  * v1  stage2 band U(0.25,0.50)  -> std 0.761 -> 1.40   stable
  * v2  stage2 band U(0.25,1.00)  -> std 0.814 -> 6.13   diverged
  * v3  stage2 band U(0.25,1.00)  -> std 0.761 -> 3.98   diverged (after a
        narrow stage 1, so the curriculum change alone did not prevent it)

So stage 2 with the wide band diverges and stage 2 with the narrow band does not.
This test isolates which property of the wide band is responsible:

  treatment = stage2 continued from the v3 stage-1 checkpoint, band U(0.80,1.00)
              (narrow width, high speed only)

  If std diverges -> the HIGH-SPEED REGIME is the cause (authority/authority
                     asymmetry; would need a frozen-file change -> iron rule 2).
  If std stays bounded -> the BAND WIDTH is the cause (a training-budget /
                     difficulty effect; fixable without touching the env).

Full-length: 5M steps (610 updates), matching v1/v2/v3 stage-2, so a stable
or divergent std here is a real result, not an artifact of under-training.
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np

import follow_env as fe
from regime_ext_env import RegimeExtWrapper

V_LO, V_HI = 0.80, 1.00
STEPS = 5_000_000


def make_env(rank):
    def _init():
        env = fe.FollowEnv(domain_randomize=True, gap_max=1.2,
                           behaviors=("constant", "sinusoid", "brake"))
        env = RegimeExtWrapper(env, v_lo=V_LO, v_hi=V_HI, seed=20_000 + rank)
        env.reset(seed=20_000 + rank)
        return env
    return _init


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    n_envs = 32
    venv = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)]))
    model = PPO.load("ckpt/follow_stage1_v3_speedext.zip", env=venv,
                     tensorboard_log="./tb_logs", verbose=1)
    print(f"HIGH-ONLY band U({V_LO},{V_HI}), {STEPS} steps "
          f"({STEPS/(n_envs*256):.0f} updates)", flush=True)
    model.learn(total_timesteps=STEPS)
    model.save("ckpt/probe_highonly_5m.zip")
    venv.close()
    print("saved -> ckpt/probe_highonly_5m.zip", flush=True)
