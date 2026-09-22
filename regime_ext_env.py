"""Regime-extension wrapper for D0 section 8 (plan_D0_scenario_audit_v3.md).

Trains the follower with the leader-speed distribution widened to cover the
AGV-typical band, WITHOUT editing the frozen follow_env.py (iron rule 2).

Why a wrapper is needed at all: FollowEnv.reset() samples

    self.v_set = float(r.uniform(0.25, 0.50))

inline, so the training range is fixed in the frozen file. This wrapper
re-samples v_set after each reset from a wider range.

Range is U(0.25, 1.00), NOT U(0.25, 1.50). Measured (plan section 8.1.1,
_step05_vset_gate.py): _leader_v() ends in np.clip(v, 0.05, 1.0), so any v_set
above 1.0 is clipped away -- a U(0.25,1.50) draw would collapse ~40% of its
mass onto a point mass at exactly 1.0 m/s. 1.00 is the widest band the
environment can actually produce, and it fully covers the target 1.0 m/s cell.

Must subclass gymnasium.Wrapper: SB3's SubprocVecEnv rejects a bare object
("not a Gymnasium environment") and every worker dies at construction. Step 0.5
caught this before any training was launched.

This is a top-level module (not __main__) so SubprocVecEnv workers can import
it. get_v_set() exists so the Step 0.5 gate can query what each worker actually
sampled -- the whole point of the gate is that the injection must reach the
child processes, not just the parent.
"""
import gymnasium as gym
import numpy as np

import follow_env as fe


class RegimeExtWrapper(gym.Wrapper):
    """Re-samples v_set from a wider range after each reset."""

    def __init__(self, env, v_lo=0.25, v_hi=1.00, seed=None):
        super().__init__(env)
        self.v_lo, self.v_hi = v_lo, v_hi
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # The frozen reset() sampled U(0.25,0.50); overwrite before the first
        # step so the whole episode uses the extended draw.
        self.env.v_set = float(self.rng.uniform(self.v_lo, self.v_hi))
        return obs, info

    def get_v_set(self):
        """Read back the realized v_set (used by the Step 0.5 gate)."""
        return float(self.env.v_set)


def make_env(rank, easy=False, v_lo=0.25, v_hi=1.00):
    """Same shape as train_ppo.make_env, plus the widened v_set range."""
    def _init():
        env = fe.FollowEnv(domain_randomize=True,
                           gap_max=0.5 if easy else 1.2,
                           behaviors=("constant", "sinusoid") if easy
                           else ("constant", "sinusoid", "brake"))
        env = RegimeExtWrapper(env, v_lo=v_lo, v_hi=v_hi, seed=10_000 + rank)
        env.reset(seed=10_000 + rank)
        return env
    return _init
