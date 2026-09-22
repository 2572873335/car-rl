"""v3 regime-extension: put the curriculum on the SPEED axis.

Why v3 exists (RUNLOG 2026-09-22, commit fde106e):
  v2 widened v_set to U(0.25,1.00) but applied the wide band in BOTH stages --
  make_env(easy=True) only narrowed gap_max and behaviors, never the speed
  range. The "easy" stage therefore already ran the full wide band, so the
  curriculum was empty where it mattered. v2's policy std diverged 0.81 -> 6.13
  (v1: 0.76 -> 1.40) and every cell got worse, including v=0.30 which is
  in-distribution for both -- the signature of a failed optimisation, not a
  distribution effect.

v3 changes EXACTLY ONE THING: stage 1 uses the narrow band U(0.25,0.50) (same as
v1's stage 1), and only stage 2 widens to U(0.25,1.00). A second candidate cause
(the quadratic reward penalty blowing up at high-speed errors) is deliberately
NOT touched -- changing two things at once would leave nothing attributable.

RegimeExtWrapper already takes v_lo/v_hi, so no wrapper change is needed; only
the make_env call differs between stages.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np

import follow_env as fe
from regime_ext_env import RegimeExtWrapper


def make_env_stage(rank, stage):
    """stage 'narrow' -> U(0.25,0.50) (v1-like); 'wide' -> U(0.25,1.00)."""
    v_lo, v_hi = (0.25, 0.50) if stage == "narrow" else (0.25, 1.00)
    easy = (stage == "narrow")

    def _init():
        env = fe.FollowEnv(domain_randomize=True,
                           gap_max=0.5 if easy else 1.2,
                           behaviors=("constant", "sinusoid") if easy
                           else ("constant", "sinusoid", "brake"))
        env = RegimeExtWrapper(env, v_lo=v_lo, v_hi=v_hi, seed=10_000 + rank)
        env.reset(seed=10_000 + rank)
        return env
    return _init


def train(stage_name, timesteps, n_envs, seed, load, out, band):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    venv = VecMonitor(SubprocVecEnv([make_env_stage(i, band)
                                     for i in range(n_envs)]))
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
    print(f"[{stage_name}] band={band} timesteps={timesteps} n_envs={n_envs} "
          f"-> {updates:.0f} updates (need >=300)", flush=True)
    model.learn(total_timesteps=timesteps)
    model.save(out)
    venv.close()
    print(f"[{stage_name}] saved -> {out}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1")
    ap.add_argument("--band", default="narrow", choices=["narrow", "wide"])
    ap.add_argument("--timesteps", type=int, default=2_500_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", default=None)
    ap.add_argument("--out", default="ckpt/follow_stage2_v3_speedext.zip")
    a = ap.parse_args()
    train(a.stage, a.timesteps, a.n_envs, a.seed, a.load, a.out, a.band)
