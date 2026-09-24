"""Attacker environment for Phase F Step 1 (reward scale corrected).

WHAT THIS IS
    The attacker drives the LEADER; a frozen defender drives the follower inside
    the frozen follow_env. Wrapping the pair as one single-agent env is what makes
    the loop escapable (plan section 0): the attacker acts on leader speed, the
    frozen defender reacts, the attacker is paid for how badly the defender does.

ANTI-LEAK (plan section 2.2)
    The attacker observes only physically observable quantities -- what a V2V
    leader could see: gap error, gap rate, follower speed, its own speed. It never
    sees the defender's internal state or its action.

REWARD SIGN -- the owner flagged this as the review watch-point. Stated in code:
    The repo encodes gap error as e = gap - D_DES and the FOLLOWER is penalised
    with -(e/0.12)**2. The attacker wants that penalty to be as negative as
    possible, so it is rewarded for LARGE |e|:
        r = +W_ERR * |e| / 0.12          grows with tracking error
          + W_CRASH * 1[collision]      the countable win condition
          - W_IDLE                      discourage standing still
    With the sign flipped the attacker becomes a helpful tracking controller and
    H-A1' would be an impossible gate.

REWARD SCALE -- measured, not guessed. A first version used W_CRASH=100 with the
dense term unnormalised; measurement showed this was inverted in practice: idling
for a full episode scored 339.8 while a successful crash scored 243.6, because the
dense term accumulates over up to 1000 steps while the crash bonus is paid once.
The dense ceiling is ~420/episode at |e|=0.05 m, so W_CRASH must exceed it:

    W_ERR   0.1     dense ceiling ~42/episode (|e| up to 0.05 m)
    W_CRASH 500     comfortably dominates any dense accumulation
    W_IDLE  0.05    a full episode of idling costs ~50

Verify with _check_reward_scale.py before trusting any training run.
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

W_ERR = 0.1
W_CRASH = 500.0
W_IDLE = 0.05
V_BASE, V_SPAN = 0.50, 0.45
ATT_OBS_DIM = 4


class AttackerEnv(gym.Env):
    def __init__(self, defender, d_des=0.20, t_max=20.0):
        self.defender = defender
        self.env = fe.FollowEnv(domain_randomize=False, t_max=t_max)
        self.d_des = d_des
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(ATT_OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _att_obs(self, def_obs, leader_v):
        e = float(def_obs[0]) * 0.5
        de = float(def_obs[1]) * 2.0
        v_f = float(def_obs[2]) * 1.3
        return np.array([e / 0.5, de / 2.0, v_f / 1.3, leader_v / 1.3],
                        dtype=np.float32)

    def _install_patch(self):
        """Give the attacker control of the leader's speed command."""
        if getattr(self.env, "_attacker_patched", False):
            return
        def patched():
            return float(getattr(self.env, "_attacker_v", V_BASE))
        self.env._leader_v = patched
        self.env._attacker_patched = True

    def reset(self, *, seed=None, options=None):
        fe.D_DES = self.d_des
        obs, info = self.env.reset(seed=seed)
        self._install_patch()
        self.env._attacker_v = V_BASE      # BUGFIX: the patch reads it on the
        self._attacker_v = V_BASE          # inner env, so set it there too
        return self._att_obs(obs, self._attacker_v), info

    def step(self, action):
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        v_cmd = float(np.clip(V_BASE + V_SPAN * a, 0.05, 1.0))
        self.env._attacker_v = v_cmd       # BUGFIX: patch reads it on the inner env
        self._attacker_v = v_cmd

        d_obs = self.env._obs()
        d_act = self.defender(d_obs)
        obs, _r, term, trunc, info = self.env.step(d_act)

        e = float(obs[0]) * 0.5
        # ---- REWARD SIGN: paid for LARGE |e| (see module docstring) --------
        reward = W_ERR * abs(e) / 0.12
        reward -= W_IDLE
        if term and self.env.term_reason == "collision":
            reward += W_CRASH
        elif term:
            reward += 0.1 * W_CRASH     # non-crash disruption, worth far less
        return self._att_obs(obs, self._attacker_v), reward, term, trunc, info

    def render(self, *a, **k):
        pass

    def close(self):
        pass
