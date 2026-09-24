"""Phase F Step 1: train the attacker against BOTH frozen defenders (H-A1').

THE DESIGN POINT THAT MATTERS
    H-A1' requires ONE attacker to reach >=60% crash rate against the rule
    defender AND >=60% against the v1 RL policy. Training one attacker per
    defender and reporting both would be two policies, not one -- and would not
    satisfy the criterion's purpose.

    The criterion is only non-empty because NO KNOWN degenerate script defeats
    both (owner Q1; measured in plan section 1.3: the square wave kills P+FF
    20/20 but v1 0/20, while 0.5 Hz does the reverse). So the attacker must
    discover a strategy that works against both simultaneously -- which is what
    "beyond every known degenerate family" means.

    Implementation: the defender is resampled per EPISODE inside a single env,
    so a single policy must handle both opponents.

Iron rule 4 applies (conclusion-producing run): 5M steps at 32x256 = 610 updates
>= 300. Time box 3 days per the plan; failure within the box stops Phase F and is
reported as a negative result.

Run: uv run python train_attacker.py            # both defenders (the gate)
     uv run python train_attacker.py --which pff  # diagnostic only
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import gymnasium as gym

import follow_env as fe
from adversary_env import AttackerEnv

OUT = f"{REPO}/results/20260925_phaseF_attacker"

# loaded once per process, reused across episodes
_V1 = None


def _v1_policy():
    global _V1
    if _V1 is None:
        from stable_baselines3 import PPO
        _V1 = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    return lambda o: _V1.predict(o, deterministic=True)[0]


PFF = lambda o: fe.baseline_action(o, use_ff=True)


class BothDefenders(AttackerEnv):
    """Resamples the defender each episode, so one attacker must beat both."""

    def __init__(self, d_des=0.20, t_max=20.0):
        super().__init__(PFF, d_des=d_des, t_max=t_max)
        self._kind = "pff"

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # choose the defender for this episode from the env's own rng
        self._kind = "pff" if self.env.rng.random() < 0.5 else "v1"
        self.defender = PFF if self._kind == "pff" else _v1_policy()
        return obs, info

    def current_defender(self):
        return self._kind


class Wrap(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


def make_env(rank, which, d_des=0.20):
    def _init():
        if which == "both":
            e = BothDefenders(d_des=d_des)
        else:
            pol = PFF if which == "pff" else _v1_policy()
            e = AttackerEnv(pol, d_des=d_des)
        e.reset(seed=20_000 + rank)
        return Wrap(e)
    return _init


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["both", "pff", "v1"], default="both",
                    help="'both' is the H-A1' configuration; others are diagnostics")
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    updates = a.timesteps / (a.n_envs * 256)
    print(f"[attacker vs {a.which}] timesteps={a.timesteps} n_envs={a.n_envs} "
          f"-> {updates:.0f} updates (iron rule 4 needs >=300)", flush=True)
    assert updates >= 300, "iron rule 4 violated"

    venv = VecMonitor(SubprocVecEnv([make_env(i, a.which)
                                     for i in range(a.n_envs)]))
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=512,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]), verbose=1, seed=a.seed)
    model.learn(total_timesteps=a.timesteps)
    out = f"{OUT}/attacker_vs_{a.which}.zip"
    model.save(out)
    venv.close()
    print(f"[attacker vs {a.which}] saved -> {out}", flush=True)
