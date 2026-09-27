"""Is the post-move harness a FAIR stronger test, or an off-distribution trap?

Motivation: under the post-move harness the learned attacker scored 20/20 vs v1,
while every other harness gave 0/20. That is the wrong direction -- v1 normally
outperforms P+FF (D0), so a harness that suddenly makes v1 crash-prone is
suspect. Hypothesis: the post-move obs is OFF-DISTRIBUTION for v1.

The frozen env computes obs AFTER both cars move. v1 was trained on that
distribution. The post-move harness shows the defender an obs taken after the
LEADER moved but BEFORE the follower moved -- a state v1 never trained on, where
the gap has already changed but the reported own-speed has not. Under the benign
null attack this should still produce clean tracking if the harness is fair.

Test: hold the leader at a constant speed (the null attack, no adversarial
content) and compare v1's tracking under the two harnesses. If v1 cannot track
a constant-speed leader under post-move, the harness is invalid for v1 and its
20/20 says nothing about the attack surface.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv, V_BASE, V_SPAN


def load_v1():
    from stable_baselines3 import PPO
    m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip", device="cpu")
    return lambda o: m.predict(o, deterministic=True)[0]


class Corrected(AttackerEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._def_obs = self.env._obs().copy()
        return obs, info

    def step(self, action):
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        v = float(np.clip(V_BASE + V_SPAN * a, 0.05, 1.0))
        self.env._attacker_v = v
        self._attacker_v = v
        obs, _r, term, trunc, info = self.env.step(self.defender(self._def_obs))
        self._def_obs = obs.copy()
        return self._att_obs(obs, self._attacker_v), 0.0, term, trunc, info


class PostMove(AttackerEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._def_obs = self.env._obs().copy()
        return obs, info

    def step(self, action):
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        v = float(np.clip(V_BASE + V_SPAN * a, 0.05, 1.0))
        self.env._attacker_v = v
        self._attacker_v = v
        env = self.env
        env.leader.step(env._leader_v(),
                        env.leader_steer.omega_cmd(env.leader, env.path), fe.DT)
        gap, e_lat = env._measure()
        e = gap - fe.D_DES
        de = (e - env.prev_e) / fe.DT
        d_obs = np.array([e / 0.5, de / 2.0, env.follower.v / 1.3, e_lat / 0.25],
                         dtype=np.float32)
        d_act = self.defender(d_obs)
        v_l = env._leader_v()
        d_a = float(np.clip(np.asarray(d_act).ravel()[0], -1.0, 1.0))
        v_cmd = float(np.clip(v_l + fe.ACT_GAIN * d_a, 0.0, fe.V_MAX))
        env.follower.step(v_cmd,
                          env.follower_steer.omega_cmd(env.follower, env.path),
                          fe.DT)
        env.t += fe.DT
        env.k += 1
        gap, e_lat = env._measure()
        e = gap - fe.D_DES
        de = (e - env.prev_e) / fe.DT
        env.prev_e = e
        term, trunc, reason = False, False, "timeout"
        if gap < fe.COLLISION_GAP:
            term, reason = True, "collision"
        elif gap > fe.MAX_GAP:
            term, reason = True, "lost"
        elif abs(e_lat) > fe.OFFTRACK:
            term, reason = True, "offtrack"
        if env.t >= env.t_max:
            trunc = True
        env.term_reason = reason
        obs = np.array([e / 0.5, de / 2.0, env.follower.v / 1.3, e_lat / 0.25],
                       dtype=np.float32)
        return self._att_obs(obs, self._attacker_v), 0.0, term, trunc, {}


def null_attack(env_cls, pol, n_ep=10):
    """Hold the leader at V_BASE (no adversarial content) and measure tracking."""
    env = env_cls(pol, d_des=0.20)
    errs, terms = [], {}
    for k in range(n_ep):
        env.reset(seed=7000 + k)
        ep = []
        while True:
            a = np.array([(V_BASE - V_BASE) / V_SPAN], dtype=np.float32)  # a = 0
            obs, _r, term, trunc, _ = env.step(a)
            ep.append(abs(float(obs[0]) * 0.5))
            if term or trunc:
                env.term_reason = env.env.term_reason
                terms[env.env.term_reason] = terms.get(env.env.term_reason, 0) + 1
                break
        errs.append(np.mean(ep) * 200.0)   # obs-cm, settled-less; comparable
    return float(np.mean(errs)), terms


v1 = load_v1()
print("=" * 84)
print("NULL ATTACK (leader held at constant 0.50 m/s) -- is the harness fair?")
print("=" * 84)
for tag, cls in [("corrected", Corrected), ("post-move", PostMove)]:
    m, t = null_attack(cls, v1)
    print(f"  v1 under {tag:>9s} harness: mean|e| = {m:7.3f} obs-cm   term={t}")

print()
print("=" * 84)
print("Reading: if v1 tracks fine under corrected but blows up under post-move")
print("even with NO adversarial content, the post-move obs is off-distribution")
print("for v1 and its 20/20 crash figure is a harness artefact, not a finding.")
print("=" * 84)
