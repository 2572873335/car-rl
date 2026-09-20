"""REAL pilot gate for Phase C (run in /tmp, no repo files touched).

Integrates the two-car self-play world with the custom VecEnv + SB3 PPO
using SHARED parameters, on the REAL track geometry, and checks whether it
actually starts learning. This is the plan's Day-1 gate, exercised early to
de-risk the 3-day probe.

Addresses the plan's own open question 6: "is there a cheaper way to falsify
this in 1 day?"  If this smoke works, the mechanism is proven and the
remaining Phase C risk is only 'does it converge', not 'is it wired right'.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from _selfplay_design_probe import World, build_paths
from overtake_env import baseline_action_ot

OBS_DIM = 6
ACT_DIM = 2


class SelfPlayVecEnv(VecEnv):
    """num_envs = 2 * n_worlds agent-rows over n_worlds shared worlds.

    Parameter sharing: ONE policy drives every row.
    Key lesson baked in (from the earlier mechanism probe): the
    observation_space must declare the PER-AGENT shape, because SB3
    flattens leading dims -- Box(shape=(2,6)) would become 12 features
    and crash against the 6-dim rows.
    """

    def __init__(self, n_worlds=2, seed=0):
        self.n_worlds = n_worlds
        self.paths = build_paths()
        self.worlds = [World(*self.paths) for _ in range(n_worlds)]
        self.rngs = [np.random.default_rng(seed + 1000 * i) for i in range(n_worlds)]
        super().__init__(
            num_envs=2 * n_worlds,
            observation_space=spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
            action_space=spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32),
        )

    def reset(self):
        return np.concatenate([w.reset(r) for w, r in zip(self.worlds, self.rngs)])

    def step_async(self, actions):
        self._acts = np.asarray(actions, np.float64).reshape(self.n_worlds, 2, ACT_DIM)

    def step_wait(self):
        obs_l, rew_l, done_l, inf_l = [], [], [], []
        for k, w in enumerate(self.worlds):
            o, r, d = w.step(self._acts[k])
            obs_l.append(o)
            rew_l.append(r)
            # both agents share the horizon -> both rows die together
            done_l.extend([d, d])
            inf_l.extend([{"reason": w.reason}, {"reason": w.reason}])
            if d:
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


def eval_vs_opponent(model, opponent_fn, n_ep=20, seed0=9000):
    """Win rate of the shared policy (as car 0) against a fixed opponent."""
    world = World(*build_paths())
    wins = losses = ties = 0
    for k in range(n_ep):
        rng = np.random.default_rng(seed0 + k)
        obs = world.reset(rng)
        while True:
            a0, _ = model.predict(obs[0], deterministic=True)
            a1 = opponent_fn(obs[1])
            _, _, done = world.step(np.stack([a0, a1]))
            if done:
                break
        d = world.delta(0)
        if d > 0.05:
            wins += 1
        elif d < -0.05:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def main():
    print("=" * 78)
    print("REAL pilot gate: shared-parameter PPO on the two-car self-play world")
    print("=" * 78)

    venv = SelfPlayVecEnv(n_worlds=4, seed=0)      # 8 rows
    print(f"  n_worlds=4 -> num_envs(rows)={venv.num_envs}")
    print(f"  per-agent obs space = {venv.observation_space}")
    print(f"  acting space        = {venv.action_space}")

    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256,
                batch_size=512, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]),
                device="cpu", seed=0, verbose=0)

    fd = model.policy.features_extractor.features_dim
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  policy features_dim = {fd}  (MUST be {OBS_DIM})")
    assert fd == OBS_DIM, f"obs flattening bug: {fd} != {OBS_DIM}"
    print(f"  shared policy params = {n_params}")
    print("  features_dim assert PASSED -> obs-flattening pitfall avoided")

    def rnd_opp(obs):
        return np.random.default_rng().uniform(-1, 1, ACT_DIM)

    def rule_opp(obs):
        return baseline_action_ot(obs)

    print()
    print("  --- win rate BEFORE training (untrained shared policy) ---")
    w, l, t = eval_vs_opponent(model, rule_opp, n_ep=20)
    print(f"    vs rule-based : W{w} L{l} T{t}")

    print()
    print("  --- training (small smoke: 8 rows x 256 steps = 2048/update) ---")
    total = 2 * 256 * 40          # 40 updates -- deliberately short smoke
    model.learn(total_timesteps=total, progress_bar=False)
    print(f"    trained {total} steps = {total/(venv.num_envs*256):.0f} updates")

    print()
    print("  --- win rate AFTER smoke training ---")
    w2, l2, t2 = eval_vs_opponent(model, rule_opp, n_ep=20)
    print(f"    vs rule-based : W{w2} L{l2} T{t2}")

    print()
    print("=" * 78)
    print("PILOT GATE RESULT")
    print("=" * 78)
    print(f"  mechanism wired correctly : YES (features_dim={fd}, shared params)")
    print(f"  training loop ran         : YES ({total} steps, no crash)")
    print(f"  win rate moved?           : {w} -> {w2} (smoke only, not a result)")
    print()
    print("  => The cheapest falsification test is exactly this: if the")
    print("     mechanism wires up and the loop runs, 'can we do it at all'")
    print("     is answered in ~minutes, not 3 days.  The 3-day budget is")
    print("     then available for the convergence question.")

    venv.close()


if __name__ == "__main__":
    main()
