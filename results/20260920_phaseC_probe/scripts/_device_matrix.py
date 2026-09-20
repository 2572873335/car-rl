"""Resolve the CPU vs CUDA contradiction with the reviewer.

Reviewer measured CPU 19% FASTER (8 parallel SubprocVecEnv, real OvertakeEnv).
I measured CUDA 1.9x FASTER (self-play custom VecEnv, single process).

Hypothesis: BOTH are correct, because the comparison depends on WHERE the
env stepping happens:
  * SubprocVecEnv  -> env work runs in worker PROCESSES, so the main process
                      is free to use many CPU cores; GPU offers little.
  * single-process custom VecEnv (what Phase C self-play MUST use, because a
    shared world cannot be split across processes) -> env stepping BLOCKS the
    main process, so offloading the network to the GPU helps.

Test both designs x both devices, now that the GPU is otherwise idle.
"""
import sys
import time
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from overtake_env import OvertakeEnv
from _run_real_probe import SelfPlayVecEnv


def make_ot(rank):
    def _init():
        e = OvertakeEnv(domain_randomize=True)
        e.reset(seed=20_000 + rank)
        return e
    return _init


def bench_model(venv, device, updates=15, n_steps=256, label=""):
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=n_steps,
                batch_size=512, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]),
                device=device, seed=0, verbose=0)
    steps = venv.num_envs * n_steps * updates
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, progress_bar=False)
    el = time.perf_counter() - t0
    venv.close()
    return steps / el, el


def main():
    print("=" * 78)
    print("device x VecEnv-design matrix (RTX 5060, GPU otherwise idle)")
    print("=" * 78)

    print()
    print("[A] SubprocVecEnv (8 separate env PROCESSES) -- reviewer's setup")
    for dev in ["cpu", "cuda"]:
        venv = SubprocVecEnv([make_ot(i) for i in range(8)])
        rate, el = bench_model(venv, dev)
        print(f"    {dev:5s}: {rate:8,.0f} steps/s  ({el:.1f}s)")

    print()
    print("[B] single-process custom VecEnv (8 worlds = 16 rows)")
    print("    -- the design Phase C self-play REQUIRES (shared world)")
    for dev in ["cpu", "cuda"]:
        venv = SelfPlayVecEnv(n_worlds=8, seed=0)
        rate, el = bench_model(venv, dev)
        print(f"    {dev:5s}: {rate:8,.0f} steps/s  ({el:.1f}s)")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print("  If [A] favours cpu while [B] favours cuda, both measurements")
    print("  were right and the plan must state the DEVICE PER DESIGN, not a")
    print("  single global choice.  Phase C uses design [B].")


if __name__ == "__main__":
    main()
