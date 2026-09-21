"""V2V transfer test, v4 -- INCREMENTAL, UNBUFFERED output.

Earlier attempts produced 0-byte / 592-byte output files because the script
only printed after each arm finished, and the process was killed before it
got there. Now every evaluation bucket prints immediately (and run with
`python -u` so nothing sits in a buffer).

Question: does the W5 verdict (joint self-play fails; frozen-opponent learns)
hold under the V2V / opponent-relative prior that the W6 plan adopts?

Higher-value alternative interpretation also tested: W6's plan proposes V2V
for comparability with the frozen env. If the verdict does NOT transfer,
the plan must say so explicitly rather than inherit W5's conclusion.
"""
import sys

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import tempfile
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from _v2v_transfer_test import WorldV2V, JointVec, OBS_DIM, ACT_DIM
from _selfplay_design_probe import build_paths

sys.stdout.reconfigure(line_buffering=True)


def line(*a):
    print(*a, flush=True)


class FrozenVec(VecEnv):
    def __init__(self, opponent, n_worlds=8, seed=0):
        self.opp = opponent
        self.n_worlds = n_worlds
        outer, inner = build_paths()
        self.worlds = [WorldV2V(outer, inner) for _ in range(n_worlds)]
        self.rngs = [np.random.default_rng(seed + 1000 * i) for i in range(n_worlds)]
        self._ret = np.zeros(n_worlds)
        self.ep_ret, self.ep_reason = [], []
        super().__init__(num_envs=n_worlds,
                         observation_space=spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
                         action_space=spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32))

    def reset(self):
        self._ret[:] = 0.0
        return np.stack([w.reset(r)[0] for w, r in zip(self.worlds, self.rngs)])

    def step_async(self, a):
        self._acts = np.asarray(a, np.float64).reshape(self.n_worlds, ACT_DIM)

    def step_wait(self):
        obs_l, rew_l, done_l, inf_l = [], [], [], []
        for k, w in enumerate(self.worlds):
            a1, _ = self.opp.predict(w.obs()[1], deterministic=True)
            o, r, d = w.step(np.stack([self._acts[k], a1]))
            self._ret[k] += r[0]
            obs_l.append(o[0]); rew_l.append(r[0]); done_l.append(d)
            inf_l.append({"reason": w.reason})
            if d:
                self.ep_ret.append(self._ret[k]); self.ep_reason.append(w.reason)
                self._ret[k] = 0.0
                self.rngs[k] = np.random.default_rng(int(self.rngs[k].integers(1 << 30)))
        return (np.stack(obs_l), np.array(rew_l, np.float32),
                np.array(done_l, bool), inf_l)

    def close(self): pass
    def env_method(self, *a, **k): raise NotImplementedError
    def get_attr(self, *a, **k): return [None] * self.num_envs
    def set_attr(self, *a, **k): pass
    def env_is_wrapped(self, *a, **k): return [False] * self.num_envs


class PrintTrend(BaseCallback):
    """Print a line every `bucket` episodes, immediately."""

    def __init__(self, venv, tag, bucket=100):
        super().__init__()
        self.v, self.tag, self.b, self._n = venv, tag, bucket, 0

    def _on_step(self):
        n = len(self.v.ep_reason)
        if n - self._n >= self.b:
            self._n = n
            rec = self.v.ep_reason[-self.b:]
            coll = sum(1 for r in rec if r == "collision") / len(rec)
            ret = float(np.mean(self.v.ep_ret[-self.b:]))
            line(f"  [{self.tag}] steps={self.num_timesteps:>9,} "
                 f"ret={ret:>8.1f} coll={100*coll:>5.1f}%")
        return True


def main():
    line("=" * 70)
    line("ARM A: JOINT self-play, V2V prior, 312 updates (matches W5 budget)")
    line("=" * 70)
    venv = JointVec(n_worlds=8, seed=0)
    m = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=512,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    cb = PrintTrend(venv, "jointV2V", 100)
    m.learn(total_timesteps=16 * 256 * 312, callback=cb, progress_bar=False)
    venv.close()
    line("ARM A DONE")

    line("=" * 70)
    line("ARM B: FROZEN opponent, V2V prior, 200 updates")
    line("=" * 70)
    warm = JointVec(n_worlds=8, seed=0)
    mw = PPO("MlpPolicy", warm, learning_rate=3e-4, n_steps=256, batch_size=512,
             gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
             policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    mw.learn(total_timesteps=16 * 256 * 20, progress_bar=False)
    warm.close()
    p = tempfile.mktemp(suffix=".zip")
    mw.save(p)
    frozen = PPO.load(p)

    fv = FrozenVec(frozen, n_worlds=8, seed=0)
    mf = PPO("MlpPolicy", fv, learning_rate=3e-4, n_steps=256, batch_size=512,
             gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
             policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    cb2 = PrintTrend(fv, "frozenV2V", 50)
    mf.learn(total_timesteps=8 * 256 * 200, callback=cb2, progress_bar=False)
    fv.close()
    line("ARM B DONE")
    line("ALL DONE")


if __name__ == "__main__":
    main()
