"""Phase C probe, run 2 -- corrected analysis.

Fixes over run 1:
  * snapshot via save/load (deepcopy of a torch module is forbidden:
    "Only Tensors created explicitly by the user support the deepcopy protocol")
  * entropy SIGN handled correctly.  SB3 logs `train/entropy_loss` as
    -entropy, so MORE NEGATIVE == MORE entropy.  Run 1's comparison
    (last <= 0.8*first) silently passed on the sign flip; report both the
    raw values and the interpretation explicitly.
  * bucketed trends: does the collision rate actually fall over training?
    This is the crux of the probe ("no longer crashing").
  * P3 measured on a batch of diverse observations, not one rollout.
"""
import sys

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import time
import tempfile
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World
from _run_real_probe import SelfPlayVecEnv, OBS_DIM, ACT_DIM


class Trend(BaseCallback):
    """Bucket episode outcomes so we can SEE whether failures decrease."""

    def __init__(self, venv, bucket=100):
        super().__init__()
        self.venv = venv
        self.bucket = bucket
        self.marks = []
        self.t0 = time.perf_counter()
        self._last_n = 0

    def _on_step(self):
        n = len(self.venv._ep_ret)
        if n - self._last_n >= self.bucket:
            self._last_n = n
            recent = np.array(self.venv._ep_ret[-self.bucket:])
            lg = self.logger.name_to_value
            self.marks.append(dict(
                steps=self.num_timesteps,
                mean_ret=float(recent.mean()),
                coll=float(np.mean(recent <= -900)),   # ~all-collision episodes
                ent=lg.get("train/entropy_loss"),
                vl=lg.get("train/value_loss"),
            ))
        return True


def load_snapshot(model):
    p = tempfile.mktemp(suffix=".zip")
    model.save(p)
    snap = PPO.load(p)
    return snap


def win_rate(model, opp, n_ep=40, seed0=7000):
    """Shared-policy (car 0) vs opponent (car 1), neutral randomised start."""
    wins = losses = ties = 0
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0, _ = model.predict(obs[0], deterministic=True)
            if opp is None:
                a1 = np.zeros(ACT_DIM)
            elif opp == "random":
                a1 = np.random.default_rng().uniform(-1, 1, ACT_DIM)
            else:
                a1, _ = opp.predict(obs[1], deterministic=True)
            obs, _, done = w.step(np.stack([a0, a1]))
            if done:
                break
        d = w.delta(0)
        if d > 0.05:
            wins += 1
        elif d < -0.05:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def main():
    STEPS = 1_280_000
    venv = SelfPlayVecEnv(n_worlds=8, seed=0)
    print(f"rows={venv.num_envs}  updates={STEPS/(venv.num_envs*256):.1f}")

    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256,
                batch_size=512, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]),
                device="cuda", seed=0, verbose=0)
    assert model.policy.features_extractor.features_dim == OBS_DIM

    untrained = load_snapshot(model)
    print("before-training win rate vs UNTRAINED self:",
          win_rate(model, untrained))

    cb = Trend(venv, bucket=100)
    t0 = time.perf_counter()
    model.learn(total_timesteps=STEPS, callback=cb, progress_bar=False)
    el = time.perf_counter() - t0
    print(f"trained {STEPS:,} steps in {el:.1f}s ({STEPS/el:,.0f} steps/s)")

    print()
    print("=" * 78)
    print("TREND: does the failure rate fall over training?  (bucket=100 eps)")
    print("=" * 78)
    print(f"  {'steps':>9s} {'mean_ret':>10s} {'~collision%':>12s} "
          f"{'entropy_loss':>13s} {'value_loss':>12s}")
    for m in cb.marks:
        e = m['ent']
        v = m['vl']
        print(f"  {m['steps']:>9,} {m['mean_ret']:>10.1f} "
              f"{100*m['coll']:>11.1f}% "
              f"{('%.3f' % e) if e is not None else '  n/a':>13s} "
              f"{('%.3e' % v) if v is not None else '  n/a':>12s}")

    print()
    print("=" * 78)
    print("P1 -- entropy, with the SIGN read correctly")
    print("=" * 78)
    ents = [m['ent'] for m in cb.marks if m['ent'] is not None]
    if len(ents) >= 4:
        first = np.mean(ents[:3])
        last = np.mean(ents[-3:])
        print(f"  entropy_loss first3={first:+.3f}  last3={last:+.3f}")
        print(f"  NOTE: SB3 logs -entropy, so more NEGATIVE = MORE exploration.")
        if last < first:
            print(f"  => entropy ROSE ({-first:.2f} -> {-last:.2f} nats). The")
            print(f"     policy got MORE exploratory; it did NOT converge to a")
            print(f"     committed behaviour.  Run 1's 'PASS' was a sign bug.")
        else:
            print(f"  => entropy FELL ({-first:.2f} -> {-last:.2f} nats).")
    vls = [m['vl'] for m in cb.marks if m['vl'] is not None]
    if vls:
        print(f"  value_loss first3={np.mean(vls[:3]):.3e} "
              f"last3={np.mean(vls[-3:]):.3e} max={max(vls):.3e}")
        print(f"  (run 1 max was 1.28e5; the TD3+BC divergence failure mode in")
        print(f"   ASSUMPTIONS.md hit 2.2e6, so this is large but not that)")

    print()
    print("=" * 78)
    print("P2 -- win rate vs opponents (neutral start, car 0 = policy)")
    print("=" * 78)
    print(f"  vs do-nothing      : {win_rate(model, None)}")
    print(f"  vs random          : {win_rate(model, 'random')}")
    trained = load_snapshot(model)
    print(f"  vs trained self    : {win_rate(model, trained)}")
    print(f"  (run 1, untrained-vs-untrained baseline was (7, 13, 0))")

    print()
    print("=" * 78)
    print("P3 -- action diversity over 500 steps from varied starts")
    print("=" * 78)
    acts = []
    for k in range(25):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(500 + k))
        for _ in range(20):
            a, _ = model.predict(obs[0], deterministic=True)
            acts.append(a)
            obs, _, done = w.step(np.stack([a, np.zeros(ACT_DIM)]))
            if done:
                break
    acts = np.array(acts)
    sd = acts.std(axis=0)
    print(f"  n={len(acts)}  speed-dim std={sd[0]:.4f}  lane-dim std={sd[1]:.4f}")
    print(f"  P3 (speed std > 0.05): {'PASS' if sd[0] > 0.05 else 'FAIL'}")

    venv.close()


if __name__ == "__main__":
    main()
