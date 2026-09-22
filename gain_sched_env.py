"""Speed-adaptive action re-parameterisation wrapper (D0 option 1.5, corrected).

WHAT THIS DOES
    FollowEnv computes, inside the frozen file:
        a   = clip(action, -1, 1)
        v_cmd = clip(v_l + ACT_GAIN * a, 0, V_MAX)      ACT_GAIN=0.8, V_MAX=1.3
    Because that map is monotone in `a`, rescaling the action at the wrapper
        a' = g * a      (env clips to [-1,1] afterwards)
    re-parameterises which policy output produces which v_cmd. For v_l above
    `v_split` the gain becomes `g_high`, so the residual can be shaped
    differently in the high-speed regime while the in-distribution gain is
    left exactly as trained.

WHAT THIS CANNOT DO  (measured, _check_reachable.py)
    The REACHABLE SET of v_cmd is [max(0, v_l - 0.8), min(1.3, v_l + 0.8)]
    and is INVARIANT under g. At v_l = 1.0 it stays [0.2, 1.3] -- a 1.30x
    speed-up cap -- no matter what the wrapper sends, because the ceiling
    (V_MAX) and the slope (ACT_GAIN) both live in the frozen file.

    So this experiment CANNOT test "authority was restored". What it does test
    is whether training stability is sensitive to the ACTION PARAMETERISATION,
    specifically to how much of the policy's output distribution falls inside
    the zero-gradient saturated region:

        v_l = 0.30 : saturation needs a > 1.25  -> unreachable, 0% of the mass
                     of any a in [-1,1] can saturate
        v_l = 1.00 : saturation at   a > 0.375  -> ~71% of a N(0,1) policy's
                     mass lies beyond it (two-sided P(|a|>0.375) = 0.708)

    g_high < 1 pushes that boundary OUT (at g=0.4, a_sat = 0.94 at v_l=1.0,
    P(|a|>0.938) = 0.35), shrinking the zero-gradient mass. g_high > 1 pulls it
    IN, enlarging it. That is the discriminating variable, and it is the
    opposite of the "raise the gain at high speed" direction: raising it
    saturates sooner.

Frozen file: untouched (sha256 verified before/after).
"""
import gymnasium as gym
import numpy as np

import follow_env as fe


class GainSchedWrapper(gym.Wrapper):
    def __init__(self, env, v_lo=0.25, v_hi=1.00, v_split=0.50,
                 g_low=1.0, g_high=0.4, seed=None):
        super().__init__(env)
        self.v_lo, self.v_hi = v_lo, v_hi
        self.v_split = v_split
        self.g_low, self.g_high = g_low, g_high
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # widen the training band (same mechanism as RegimeExtWrapper)
        self.env.v_set = float(self.rng.uniform(self.v_lo, self.v_hi))
        return obs, info

    def _gain(self):
        return self.g_high if float(self.env.v_set) > self.v_split else self.g_low

    def step(self, action):
        a = np.asarray(action, dtype=np.float64) * self._gain()
        return self.env.step(np.clip(a, -1.0, 1.0))

    def get_gain(self):
        """For the gate: confirm the wrapper is actually in the worker."""
        return float(self._gain())

    def get_v_set(self):
        return float(self.env.v_set)


def make_env(rank, stage, v_split=0.50, g_low=1.0, g_high=0.4):
    """stage 'narrow' -> U(0.25,0.50); 'wide' -> U(0.25,1.00)."""
    v_lo, v_hi = (0.25, 0.50) if stage == "narrow" else (0.25, 1.00)
    easy = (stage == "narrow")

    def _init():
        env = fe.FollowEnv(domain_randomize=True,
                           gap_max=0.5 if easy else 1.2,
                           behaviors=("constant", "sinusoid") if easy
                           else ("constant", "sinusoid", "brake"))
        env = GainSchedWrapper(env, v_lo=v_lo, v_hi=v_hi, v_split=v_split,
                               g_low=g_low, g_high=g_high, seed=30_000 + rank)
        env.reset(seed=30_000 + rank)
        return env
    return _init
