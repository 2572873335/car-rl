"""Regime-extension retrain (D0 section 8, plan_D0_scenario_audit_v3.md).

Purpose: test H-D0.5 -- whether RL's loss at 1.0 m/s is caused by the training
distribution not covering that speed band, rather than an architectural ceiling.
Widening the band and retraining should recover the advantage if the hypothesis
holds.

This is a RETRAIN, so iron rule 4 applies (>=300 updates). It does NOT touch the
frozen environment files (iron rule 2): the wider v_set band is injected by
regime_ext_env.RegimeExtWrapper, and the gate step05_gate.py has verified the
injection reaches the SubprocVecEnv workers.

Recipe matches the original stage2 as closely as the widened band allows:
  PPO MlpPolicy[128,128], lr 3e-4, n_steps 256, batch 512, n_envs 32,
  gamma 0.99, gae_lambda 0.95, clip 0.2, ent_coef 0.01, seed 0, 5M timesteps.
  Curriculum preserved: stage1 (easy: short gaps, gentle leaders) then stage2.

Updates: 5e6 / (32 * 256) = 610 >= 300  (iron rule 4)

Outputs ckpt/follow_stage2_v2_speedext.zip -- named v2 to keep it distinct from
follow_stage2_final_v1.zip; review2 ruled these must never be called "the same
controller".

Run:  uv run python train_regime_ext.py 2>&1 | tee results/20260922_D0_regime_ext/train.log
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np

import follow_env as fe
from regime_ext_env import RegimeExtWrapper, make_env


def train(stage, timesteps, n_envs, seed, load, easy, out):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    venv = VecMonitor(SubprocVecEnv([make_env(i, easy) for i in range(n_envs)]))

    if load:
        model = PPO.load(load, env=venv, tensorboard_log="./tb_logs", verbose=1)
        print(f"continuing from {load}", flush=True)
    else:
        model = PPO("MlpPolicy", venv,
                    learning_rate=3e-4, n_steps=256, batch_size=512,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    tensorboard_log="./tb_logs", verbose=1, seed=seed)

    updates = timesteps / (n_envs * 256)
    print(f"[{stage}] timesteps={timesteps} n_envs={n_envs} "
          f"-> {updates:.0f} updates (iron rule 4 needs >=300)", flush=True)
    if updates < 300:
        print("WARNING: fewer than 300 updates", flush=True)

    model.learn(total_timesteps=timesteps)
    model.save(out)
    venv.close()
    print(f"[{stage}] saved -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--timesteps", type=int, default=2_500_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", default=None)
    ap.add_argument("--easy", action="store_true")
    ap.add_argument("--out", default="ckpt/follow_stage2_v2_speedext.zip")
    a = ap.parse_args()
    train(a.stage, a.timesteps, a.n_envs, a.seed, a.load, a.easy, a.out)
