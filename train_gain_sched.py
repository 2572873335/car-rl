"""H-D0.8: does action re-parameterisation in the high-speed regime stabilise
wide-band training?

Pre-registered in research/ASSUMPTIONS.md. Two arms, because the direction is an
empirical question rather than a foregone one:

  arm A  g_high = 0.4   shrinks the zero-gradient mass (saturation at
                        |a|>0.938 at v_l=1.0, vs 0.375 with g=1)
  arm B  g_high = 1.6   amplifies small residual outputs (the owner's stated
                        direction) but saturates SOONER (|a|>0.234)

Both leave the in-distribution gain (v <= 0.50) exactly as trained, so any
change is attributable to the high-speed regime.

What a positive result would and would not mean (measured, _check_reachable.py):
the reachable v_cmd set is invariant under gain, so a win means "training
stability is sensitive to action parameterisation", NOT "authority was
restored". A loss for both arms does NOT falsify the authority hypothesis,
since neither arm changes the ceiling.

Recipe otherwise matches v1/v2/v3 stage 2: 5M steps, 610 updates, seed 0,
continued from the narrow-stage-1 checkpoint so only the stage-2 band and gain
differ.

Run: uv run python train_gain_sched.py --arm A|B
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

from gain_sched_env import make_env

ARMS = {"A": 0.4, "B": 1.6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", default="ckpt/follow_stage1_v3_speedext.zip")
    a = ap.parse_args()

    g_high = ARMS[a.arm]
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    venv = VecMonitor(SubprocVecEnv(
        [make_env(i, "wide", g_high=g_high) for i in range(a.n_envs)]))
    model = PPO.load(a.load, env=venv, tensorboard_log="./tb_logs", verbose=1)

    updates = a.timesteps / (a.n_envs * 256)
    out = f"ckpt/follow_stage2_v4_gain{a.arm}.zip"
    print(f"[arm {a.arm}] g_high={g_high} band U(0.25,1.00) "
          f"timesteps={a.timesteps} -> {updates:.0f} updates", flush=True)
    model.learn(total_timesteps=a.timesteps)
    model.save(out)
    venv.close()
    print(f"[arm {a.arm}] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
