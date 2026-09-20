"""ANSWER THE PROBE QUESTION EMPIRICALLY, NOT IN 3 DAYS.

The W5 gate asks: can shared-parameter zero-sum self-play train here?
Measured throughput (~1,470 steps/s on cuda, 16 rows) means the plan's own
recommended 1.28M steps costs ~15 minutes -- not 3 days.  So run it now and
report P1/P2/P3 against the plan's thresholds.

Uses the F5-fixed world (neutral start).  Logs, per update:
  - policy entropy      (P1: must fall; also settles whether SB3 exposes it)
  - value loss          (P1: < 1e4, no divergence)
  - mean episode return (F4: expected to RISE toward 0 as failures drop)
and periodically evaluates win rate vs a frozen snapshot (P2 replacement:
the rule machine is NOT a valid opponent per F5 follow-up).
"""
import sys
import time
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World

OBS_DIM, ACT_DIM = 6, 2


class SelfPlayVecEnv(VecEnv):
    """num_envs = 2 * n_worlds rows over n_worlds shared worlds.
    obs space is PER-AGENT shaped (6,) -- SB3 flattens leading dims."""

    def __init__(self, n_worlds=8, seed=0):
        self.n_worlds = n_worlds
        self.paths = build_paths()
        self.worlds = [World(*self.paths) for _ in range(n_worlds)]
        self.rngs = [np.random.default_rng(seed + 1000 * i) for i in range(n_worlds)]
        self._ret = np.zeros(n_worlds)
        self._ep_ret = []
        super().__init__(
            num_envs=2 * n_worlds,
            observation_space=spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
            action_space=spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32),
        )

    def reset(self):
        self._ret[:] = 0.0
        return np.concatenate([w.reset(r) for w, r in zip(self.worlds, self.rngs)])

    def step_async(self, actions):
        self._acts = np.asarray(actions, np.float64).reshape(self.n_worlds, 2, ACT_DIM)

    def step_wait(self):
        obs_l, rew_l, done_l, inf_l = [], [], [], []
        for k, w in enumerate(self.worlds):
            o, r, d = w.step(self._acts[k])
            self._ret[k] += r.sum()          # episode return = r0 + r1
            obs_l.append(o)
            rew_l.append(r)
            done_l.extend([d, d])
            inf_l.extend([{"reason": w.reason}] * 2)
            if d:
                self._ep_ret.append(self._ret[k])
                self._ret[k] = 0.0
                self.rngs[k] = np.random.default_rng(
                    int(self.rngs[k].integers(1 << 30)))
        return (np.concatenate(obs_l), np.concatenate(rew_l),
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


class Metrics(BaseCallback):
    """Pull entropy / value loss straight from SB3's logger so we can settle
    P1 executability ('can we even read policy entropy?')."""

    def __init__(self, every=20):
        super().__init__()
        self.every = every
        self.rows = []
        self.t0 = time.perf_counter()

    def _on_step(self):
        if self.n_calls % self.every == 0:
            lg = self.logger.name_to_value
            ent = lg.get("train/entropy_loss")
            vl = lg.get("train/value_loss")
            self.rows.append((self.num_timesteps, ent, vl))
        return True


def snapshot_opponent(model):
    """Freeze current weights as the opponent (P2 replacement for the
    non-adversarial rule machine)."""
    import copy
    return copy.deepcopy(model)


def win_rate_vs(model, opp, n_ep=20, seed0=7000):
    wins = losses = ties = 0
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0, _ = model.predict(obs[0], deterministic=True)
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
    STEPS = 1_280_000          # plan S5: 312.5 updates at 16 rows
    venv = SelfPlayVecEnv(n_worlds=8, seed=0)
    print(f"rows={venv.num_envs}  updates={STEPS/(venv.num_envs*256):.1f}"
          f"  (iron rule 4 needs >=300)")

    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256,
                batch_size=512, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]),
                device="cuda", seed=0, verbose=0)
    assert model.policy.features_extractor.features_dim == OBS_DIM

    opp0 = snapshot_opponent(model)          # untrained snapshot
    print(f"win rate vs UNTRAINED snapshot (before): {win_rate_vs(model, opp0)}")

    cb = Metrics(every=20)
    t0 = time.perf_counter()
    model.learn(total_timesteps=STEPS, callback=cb, progress_bar=False)
    el = time.perf_counter() - t0
    print(f"trained {STEPS:,} steps in {el:.1f}s ({STEPS/el:,.0f} steps/s)")

    print()
    print("=" * 78)
    print("P1: entropy trend (first 10 vs last 10 samples) + value loss")
    print("=" * 78)
    ents = [r[1] for r in cb.rows if r[1] is not None]
    vls = [r[2] for r in cb.rows if r[2] is not None]
    if ents:
        first, last = np.mean(ents[:10]), np.mean(ents[-10:])
        print(f"  samples available: {len(ents)}  (SB3 DOES expose entropy)")
        print(f"  entropy first10={first:+.4f}  last10={last:+.4f}  "
              f"ratio={last/first if first else float('nan'):.3f}")
        print(f"  P1 entropy criterion (last <= 0.8*first): "
              f"{'PASS' if first and last <= 0.8*first else 'FAIL'}")
    else:
        print("  entropy NOT available from logger under this name")
    if vls:
        print(f"  value_loss first10={np.mean(vls[:10]):.3e} "
              f"last10={np.mean(vls[-10:]):.3e}  max={max(vls):.3e}")
        print(f"  P1 divergence criterion (max < 1e4): "
              f"{'PASS' if max(vls) < 1e4 else 'FAIL'}")
    if venv._ep_ret:
        er = np.array(venv._ep_ret)
        print(f"  episode returns (r0+r1): n={len(er)} mean={er.mean():+.1f} "
              f"min={er.min():+.0f} max={er.max():+.0f}")

    print()
    print("=" * 78)
    print("P2 replacement: win rate vs TRAINED snapshot")
    print("=" * 78)
    opp1 = snapshot_opponent(model)
    w, l, t = win_rate_vs(model, opp1)
    print(f"  vs its own final snapshot: W{w} L{l} T{t}")
    print("  (should be ~balanced if self-play is stable; a lopsided result")
    print("   means one side collapsed)")

    print()
    print("=" * 78)
    print("P3: action diversity (not collapsed to a constant)")
    print("=" * 78)
    w_tmp = World(*build_paths())
    obs = w_tmp.reset(np.random.default_rng(123))
    acts = []
    for _ in range(200):
        a, _ = model.predict(obs[0], deterministic=True)
        acts.append(a)
        obs, _, d = w_tmp.step(np.stack([a, np.zeros(ACT_DIM)]))
        if d:
            obs = w_tmp.reset(np.random.default_rng(123))
    acts = np.array(acts)
    sd = acts.std(axis=0)
    print(f"  action std over 200 steps: {np.round(sd, 4)}")
    print(f"  P3 criterion (speed-dim std > 0.05): "
          f"{'PASS' if sd[0] > 0.05 else 'FAIL'}")

    venv.close()


if __name__ == "__main__":
    main()
