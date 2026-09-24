"""Step 0.5 (plan v1.1, Q4): rehearse the real thing before committing to Step 2.

Owner's ruling: a random-amplitude smoke test proves nothing about training under
adversarial pressure, so Step 0.5 now trains the DEFENDER for 1M steps against a
scripted random-waveform attacker and passes only on BOTH gates:
  gate 1  policy std stays bounded (no divergence)
  gate 2  completion rate is sane (episodes actually finish, not all terminated)

Gate 2 exists because "std bounded" alone can be satisfied by a collapsed policy.

The attacker here is a scripted random waveform in the style of family A (the
learned attacker is Step 1), so this rehearses the training conditions without
needing the attacker to exist yet.

Run: uv run python results/20260925_phaseF_step05/step05_train.py
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import gymnasium as gym

import follow_env as fe

OUT = f"{REPO}/results/20260925_phaseF_step05"
os.makedirs(OUT, exist_ok=True)


class RandomWaveLeader(fe.FollowEnv):
    """Leader speed driven by a per-episode random waveform (family-A style)."""

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        r = self.rng
        kind = r.integers(0, 4)
        if kind == 0:                      # square
            period = r.uniform(0.2, 0.8)
            self._wf = lambda t: 0.95 if int(t / period) % 2 == 0 else 0.05
        elif kind == 1:                    # sine
            f = r.uniform(0.1, 0.8)
            ph = r.uniform(0, 2 * np.pi)
            self._wf = lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * f * t + ph)
        elif kind == 2:                    # piecewise constant
            seg = r.uniform(0.2, 0.8)
            seedv = r.uniform(0, 1000)
            rr = np.random.default_rng(int(seedv))
            vals = rr.uniform(0.05, 1.0, 200)
            self._wf = lambda t: vals[int(t / seg) % 200]
        else:                              # chirp
            self._wf = lambda t: 0.5 + 0.45 * np.sin(
                2 * np.pi * (0.1 + 0.15 * min(t, 4.0)) * t)
        return obs, info

    def _leader_v(self):
        fn = getattr(self, "_wf", None)
        if fn is None:
            return super()._leader_v()
        return float(np.clip(fn(self.t), 0.05, 1.0))


class Wrap(gym.Wrapper):
    """Passes the env through so SB3 accepts it."""

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


def make_env(rank):
    def _init():
        e = RandomWaveLeader(domain_randomize=True)
        e.reset(seed=10_000 + rank)
        return Wrap(e)
    return _init


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback

    N_ENVS = 16
    # 1M steps at 16x256 = 244 updates, below the iron-rule-4 floor of 300.
    # FORMALLY EXEMPT per DATA_MANAGEMENT section 10.8: this is a rehearsal gate
    # (smoke/probe), not a conclusion-producing training run. Declared here and
    # in the RUNLOG entry, as that clause requires.
    STEPS = 1_000_000     # ~244 updates at 16*256

    stds, lens = [], []

    class Probe(BaseCallback):
        def __init__(self):
            super().__init__()
            self.hist = []

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            try:
                std = float(self.model.policy.log_std.exp().mean().item())
                self.hist.append(std)
            except Exception:
                pass
            return True

    venv = VecMonitor(SubprocVecEnv([make_env(i) for i in range(N_ENVS)]))
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=512,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]), verbose=1, seed=0)
    cb = Probe()
    model.learn(total_timesteps=STEPS, callback=cb)
    model.save(f"{OUT}/step05_defender.zip")
    venv.close()

    hist = np.array(cb.hist) if cb.hist else np.array([np.nan])
    print("\n" + "=" * 78)
    print("STEP 0.5 -- rehearsal gate")
    print("=" * 78)
    print(f"  updates recorded     : {len(hist)}")
    print(f"  std first/last/max   : {hist[0]:.3f} / {hist[-1]:.3f} / {hist.max():.3f}")
    g1 = hist.max() < 2.0
    print(f"  gate 1 (std max < 2.0)          : {'PASS' if g1 else 'FAIL'}")

    # gate 2: completion rate on the same attack family
    from stable_baselines3.common.vec_env import DummyVecEnv
    ev = RandomWaveLeader(domain_randomize=False)
    fin, coll = 0, 0
    N = 40
    for k in range(N):
        obs, _ = ev.reset(seed=60_000 + k)
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = ev.step(a)
            if term or trunc:
                break
        if ev.t < ev.t_max - 1e-6:        # ended before the horizon
            fin += 1
        if ev.term_reason == "collision":
            coll += 1
    rate = fin / N
    g2 = rate < 0.5                    # not everything terminating early
    print(f"  early-termination rate          : {rate:.1%} ({fin}/{N})")
    print(f"  collisions                      : {coll}/{N}")
    print(f"  gate 2 (early-term rate < 50%)  : {'PASS' if g2 else 'FAIL'}")
    ok = g1 and g2
    print(f"\n  STEP 0.5: {'PASS -> proceed to Step 1' if ok else 'FAIL -> do not proceed'}")
