"""PARALLEL throughput check on frozen OvertakeEnv (SubprocVecEnv).

Needs an explicit __main__ guard: SubprocVecEnv forks worker processes that
re-import this module, so top-level code would re-execute recursively in every
worker (this is a REAL pitfall for the Phase C training script too).

Lesson from the first attempt: without the guard ->
  EOFError: unexpected EOF   in multiprocessing/forkserver.py
"""
import time

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from overtake_env import OvertakeEnv, baseline_action_ot


def make_env(rank):
    def _init():
        env = OvertakeEnv(domain_randomize=True)
        env.reset(seed=20_000 + rank)
        return env
    return _init


def bench(vec, n_envs, label, seconds=12.0):
    obs = vec.reset()
    n_steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        acts = np.stack([baseline_action_ot(o) for o in obs])
        obs, _, dones, _ = vec.step(acts)
        n_steps += n_envs
        if dones.any():
            obs = vec.reset()
    el = time.perf_counter() - t0
    rate = n_steps / el
    print(f"  {label:14s} {rate:9,.0f} steps/s   "
          f"1.28M = {1_280_000/rate/60:6.2f} min   100k = {100_000/rate/60:5.2f} min")
    vec.close()
    return rate


def main():
    print("=" * 76)
    print("PARALLEL throughput on frozen OvertakeEnv (rule policy drives every row)")
    print("=" * 76)
    results = {}
    for n in [1, 4, 8, 16]:
        vec = DummyVecEnv([make_env(0)]) if n == 1 else \
            SubprocVecEnv([make_env(i) for i in range(n)])
        results[n] = bench(vec, n, f"n_envs={n}")

    print()
    print("=" * 76)
    print("scaling efficiency vs single env")
    print("=" * 76)
    base = results[1]
    for n in [1, 4, 8, 16]:
        print(f"  n_envs={n:2d}: {results[n]/base:5.2f}x speedup "
              f"(ideal {n:2d}x, efficiency {results[n]/base/n*100:5.1f}%)")

    print()
    print("=" * 76)
    print("iron-rule-4 arithmetic (updates = timesteps / (n_rows * n_steps))")
    print("=" * 76)
    for n_worlds, ts in [(8, 100_000), (8, 1_280_000), (4, 614_400), (2, 307_200)]:
        rows = 2 * n_worlds
        upd = ts / (rows * 256)
        ok = "OK " if upd >= 300 else "VIOLATES rule 4"
        print(f"  {n_worlds} worlds ({rows:2d} rows), {ts:>9,} steps -> "
              f"{upd:6.1f} updates  {ok}")


if __name__ == "__main__":
    main()
