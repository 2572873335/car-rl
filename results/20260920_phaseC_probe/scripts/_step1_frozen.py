"""Reviewer's Step 1: the cheapest decisive falsification of self-play.

"Freeze one side, train only the other." Removes non-stationarity -- the
notorious MARL failure mode -- so if shared-parameter PPO STILL cannot learn,
self-play is hopeless and RQ3 should be dropped. If it CAN learn, the
mechanism is validated and only non-stationarity remains as a W6 problem.

This is exactly roadmap's own "联赛退化档" (league degenerate mode).

Design: warm up the shared policy briefly, freeze that snapshot AS the
opponent, then train ONLY car 0 against it. Report whether car 0's collision
rate falls and its win rate rises -- i.e. whether learning happens at all.
"""
import sys
import time
import tempfile
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World

OBS_DIM, ACT_DIM = 6, 2


class FrozenOppVecEnv(VecEnv):
    """ONE trainable car per world, opposed by a FROZEN policy.

    Because the opponent is frozen, this is a standard stationary
    single-agent RL problem -- no shared world, no cross-row coupling.
    (That is precisely the point: it isolates non-stationarity.)
    """

    def __init__(self, opponent, n_worlds=8, seed=0):
        self.opp = opponent
        self.n_worlds = n_worlds
        self.paths = build_paths()
        self.worlds = [World(*self.paths) for _ in range(n_worlds)]
        self.rngs = [np.random.default_rng(seed + 1000 * i) for i in range(n_worlds)]
        self._ret = np.zeros(n_worlds)
        self._ep_ret = []
        self._coll = []
        super().__init__(
            num_envs=n_worlds,                       # only car 0 trains
            observation_space=spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
            action_space=spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32),
        )

    def reset(self):
        self._ret[:] = 0.0
        outs = []
        for w, r in zip(self.worlds, self.rngs):
            o = w.reset(r)
            outs.append(o[0])
        return np.stack(outs)

    def step_async(self, actions):
        self._acts = np.asarray(actions, np.float64).reshape(self.n_worlds, ACT_DIM)

    def step_wait(self):
        obs_l, rew_l, done_l, inf_l = [], [], [], []
        for k, w in enumerate(self.worlds):
            a0 = self._acts[k]
            # frozen opponent drives car 1
            a1, _ = self.opp.predict(w.obs()[1], deterministic=True)
            o, r, d = w.step(np.stack([a0, a1]))
            self._ret[k] += r[0]
            obs_l.append(o[0])
            rew_l.append(r[0])
            done_l.append(d)
            inf_l.append({"reason": w.reason})
            if d:
                self._ep_ret.append(self._ret[k])
                self._coll.append(w.reason == "collision")
                self._ret[k] = 0.0
                self.rngs[k] = np.random.default_rng(
                    int(self.rngs[k].integers(1 << 30)))
        return (np.stack(obs_l), np.array(rew_l, np.float32),
                np.array(done_l, bool), inf_l)

    def close(self):
        pass
    def env_method(self, *a, **k):
        raise NotImplementedError
    def get_attr(self, *a, **k):
        return [None] * self.num_envs
    def set_attr(self, *a, **k):
        pass
    def env_is_wrapped(self, *a, **k):
        return [False] * self.num_envs


def snapshot(model):
    p = tempfile.mktemp(suffix=".zip")
    model.save(p)
    return PPO.load(p)


def trend(venv, marks, every=50):
    n = len(venv._ep_ret)
    if n >= every:
        r = np.array(venv._ep_ret[-every:])
        c = np.array(venv._coll[-every:])
        marks.append((n, r.mean(), c.mean()))
        return True
    return False


def main():
    print("=" * 78)
    print("STEP 1: FREEZE ONE SIDE (league mode) -- the decisive test")
    print("=" * 78)

    # --- warm-up: get a non-trivial opponent via brief shared-param training ---
    from _run_real_probe import SelfPlayVecEnv
    warm = SelfPlayVecEnv(n_worlds=8, seed=0)
    m = PPO("MlpPolicy", warm, learning_rate=3e-4, n_steps=256, batch_size=512,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    m.learn(total_timesteps=16 * 256 * 20, progress_bar=False)   # 20 updates
    warm.close()
    frozen = snapshot(m)
    print("  warm-up done (20 shared-param updates) -> opponent frozen")

    # --- main: train car 0 ONLY, against the frozen opponent ---
    venv = FrozenOppVecEnv(frozen, n_worlds=8, seed=0)
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=512,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)

    marks = []
    class CB:
        def __call__(self, *a, **k):
            return True
    from stable_baselines3.common.callbacks import BaseCallback
    class T(BaseCallback):
        def _on_step(self):
            trend(venv, marks)
            return True

    STEPS = 8 * 256 * 200        # 200 updates, 409,600 steps, stationary task
    t0 = time.perf_counter()
    model.learn(total_timesteps=STEPS, callback=T(), progress_bar=False)
    el = time.perf_counter() - t0
    print(f"  trained {STEPS:,} steps in {el:.1f}s ({STEPS/el:,.0f} steps/s)")
    print(f"  updates = {STEPS/(venv.num_envs*256):.0f}")

    print()
    print(f"  {'episodes':>9s} {'mean_return':>12s} {'collision%':>11s}")
    for n, mr, cr in marks[::max(1, len(marks)//12)]:
        print(f"  {n:>9d} {mr:>12.1f} {100*cr:>10.1f}%")

    if len(marks) >= 4:
        first = np.mean([x[2] for x in marks[:3]])
        last = np.mean([x[2] for x in marks[-3:]])
        print()
        print("=" * 78)
        print("VERDICT")
        print("=" * 78)
        print(f"  collision rate: first3={100*first:.1f}%  "
              f"last3={100*last:.1f}%")
        if last < first - 0.1:
            print("  => LEARNING HAPPENED with a frozen opponent.")
            print("     Self-play is viable; non-stationarity (not the")
            print("     mechanism) is the remaining W6 problem.  PROBE PASSES.")
        else:
            print("  => NO LEARNING even with a frozen opponent.")
            print("     The blocker is NOT non-stationarity.  Per roadmap's")
            print("     failure plan, RQ3 should be DROPPED / Phase D pulled in.")
    venv.close()


if __name__ == "__main__":
    main()
