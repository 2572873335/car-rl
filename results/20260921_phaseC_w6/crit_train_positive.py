"""Produce a VALID positive control: train a PPO in-world (criterion world v2).

The frozen checkpoint is out of distribution here (trained vs a 0.15-0.30 m/s
leader; this world's leader cruises 0.40-0.80 m/s). Per the plan's rule, no
threshold may be registered without a positive pole, so build one.

Task: follower (car 0) starts 0.30-0.90 m behind a scripted cruising leader and
must overtake. Scripted leader keeps the task well-posed and stationary (no
non-stationarity), so this is a plain single-agent RL problem.
"""
import sys

# F18/F19 BOOTSTRAP: resolve helper modules from the REPO, never /tmp.
# (/tmp copies caused a stale-module contamination and made these scripts
#  unrunnable from a clean checkout.)
import os as _os, sys as _sys
_RESULTS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_REPO = _os.path.dirname(_RESULTS)
_SCRIPTS = _os.path.join(_RESULTS, '20260920_phaseC_probe', 'scripts')
for _p in ('/tmp', '/home/zy/car_rl/code0919'):
    while _p in _sys.path:
        _sys.path.remove(_p)
for _p in (_SCRIPTS, _REPO):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from _criterion_v2 import WorldV2
from _selfplay_design_probe import build_paths

OBS_DIM, ACT_DIM = 6, 2


class FollowVec(VecEnv):
    """n worlds; only car 0 is controlled, car 1 is the scripted leader."""

    def __init__(self, n_worlds=8, seed=0):
        self.n_worlds = n_worlds
        self.paths = build_paths()
        self.worlds = [WorldV2(*self.paths, prior="v2v") for _ in range(n_worlds)]
        self.rngs = [np.random.default_rng(seed + 1000 * i) for i in range(n_worlds)]
        self.ep_task, self.ep_coll, self.ep_delta = [], [], []
        super().__init__(num_envs=n_worlds,
                         observation_space=spaces.Box(-np.inf, np.inf,
                                                      (OBS_DIM,), np.float32),
                         action_space=spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32))

    def reset(self):
        return np.stack([w.reset(r)[0] for w, r in zip(self.worlds, self.rngs)])

    def step_async(self, a):
        self._acts = np.asarray(a, np.float64).reshape(self.n_worlds, ACT_DIM)

    def step_wait(self):
        obs_l, rew_l, done_l, inf_l = [], [], [], []
        for k, w in enumerate(self.worlds):
            a0 = self._acts[k]
            a1 = np.zeros(ACT_DIM)          # scripted leader holds cruise speed
            o, d = w.step(np.stack([a0, a1]), leader_action=a1)
            # reward: make progress relative to the leader, penalise crashing
            r = 0.05 * w.delta(0)
            if w.reason == "collision":
                r -= 5.0
            elif w.reason == "offtrack":
                r -= 2.0
            obs_l.append(o[0]); rew_l.append(r); done_l.append(d)
            inf_l.append({"reason": w.reason})
            if d:
                self.ep_task.append(w.m_task())
                self.ep_coll.append(1.0 if w.reason == "collision" else 0.0)
                self.ep_delta.append(w.m_delta())
                self.rngs[k] = np.random.default_rng(
                    int(self.rngs[k].integers(1 << 30)))
        return (np.stack(obs_l), np.array(rew_l, np.float32),
                np.array(done_l, bool), inf_l)

    def close(self): pass
    def env_method(self, *a, **k): raise NotImplementedError
    def get_attr(self, *a, **k): return [None] * self.num_envs
    def set_attr(self, *a, **k): pass
    def env_is_wrapped(self, *a, **k): return [False] * self.num_envs


class Trend(BaseCallback):
    def __init__(self, venv, bucket=100):
        super().__init__()
        self.v, self.b, self.marks, self._n = venv, bucket, [], 0

    def _on_step(self):
        n = len(self.v.ep_task)
        if n - self._n >= self.b:
            self._n = n
            t = np.mean(self.v.ep_task[-self.b:])
            c = np.mean(self.v.ep_coll[-self.b:])
            self.marks.append((self.num_timesteps, t, c))
            print(f"  [pos-train] steps={self.num_timesteps:>8,} "
                  f"task_rate={t:.2f} coll={c:.2f}", flush=True)
        return True


def main():
    venv = FollowVec(n_worlds=8, seed=0)
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256,
                batch_size=512, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]),
                device="cpu", seed=0, verbose=0)
    assert model.policy.features_extractor.features_dim == OBS_DIM
    print("training in-world positive control (300 updates)...", flush=True)
    model.learn(total_timesteps=8 * 256 * 300, callback=Trend(venv, 100),
                progress_bar=False)
    model.save(str(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "positive_inworld")))
    print("saved -> /tmp/positive_inworld.zip", flush=True)

    # final evaluation
    from _criterion_v2 import run
    import tempfile
    p = PPO.load(str(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "positive_inworld")))
    d, t, k, c, rs = run(lambda o, L: p.predict(o, deterministic=True)[0])
    print()
    print("=" * 70)
    print("IN-WORLD POSITIVE CONTROL, final eval (40 episodes)")
    print(f"  mean_delta = {d.mean():+.3f}  uniq = {len(set(np.round(d,6)))}")
    finite_t = [x for x in t if np.isfinite(x)]
    print(f"  overtake rate = {len(finite_t)/len(t):.2f}  "
          f"mean_t_overtake = {np.mean(finite_t) if finite_t else float('nan'):.2f}s")
    print(f"  task success = {k.mean():.2f}   collisions = {c:.2f}")
    print(f"  reasons = {rs}")
    venv.close()


if __name__ == "__main__":
    main()
