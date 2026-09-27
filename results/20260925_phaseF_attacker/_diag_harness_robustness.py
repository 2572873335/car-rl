"""Two robustness checks for the Step 1 retraction review.

CHECK 1 -- P+FF episodes are bit-identical between the buggy and the corrected
    harness. If so, the "0/20 vs P+FF" result is not merely robust to the
    harness choice, it is the SAME MEASUREMENT under both -- and H-A1' required
    >=60% against BOTH defenders, so P+FF alone settles the gate.

CHECK 2 -- a stronger-than-fair harness. The corrected harness feeds the
    defender the PRE-step state (same as the buggy path, but with a true de).
    An alternative reading of "same-tick settlement" would let the defender see
    the state AFTER the leader has already moved -- strictly more information.
    If the learned attacker still fails there, no harness choice in this family
    rescues it.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
import adversary_env as ae
from adversary_env import AttackerEnv, V_BASE, V_SPAN

CKPT = f"{REPO}/results/20260925_phaseF_attacker/attacker_vs_both.zip"
PFF = lambda o: fe.baseline_action(o, use_ff=True)


def load_v1():
    from stable_baselines3 import PPO
    m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip", device="cpu")
    return lambda o: m.predict(o, deterministic=True)[0]


def load_attacker():
    from stable_baselines3 import PPO
    return PPO.load(CKPT, device="cpu")


class Corrected(AttackerEnv):
    """Defender sees the previous step's true post-step obs (truthful de)."""

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
    """STRONGER than fair: the defender reacts to the state after the leader
    moved this tick. Implemented by splitting the frozen step: move the leader,
    show the defender the resulting obs, then settle the follower."""

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # recompute de honestly for the first tick
        self._def_obs = self.env._obs().copy()
        return obs, info

    def step(self, action):
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        v = float(np.clip(V_BASE + V_SPAN * a, 0.05, 1.0))
        self.env._attacker_v = v
        self._attacker_v = v

        env = self.env
        # 1. leader moves alone (mirrors frozen step's first half)
        env.leader.step(env._leader_v(),
                        env.leader_steer.omega_cmd(env.leader, env.path),
                        fe.DT)
        # 2. now hand the defender a truthful obs of this post-leader-move state
        gap, e_lat = env._measure()
        e = gap - fe.D_DES
        de = (e - env.prev_e) / fe.DT
        d_obs = np.array([e / 0.5, de / 2.0, env.follower.v / 1.3, e_lat / 0.25],
                         dtype=np.float32)
        d_act = self.defender(d_obs)

        # 3. settle the follower exactly as the frozen env would
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


print("=" * 84)
print("CHECK 1 -- P+FF trajectories: buggy vs corrected harness")
print("=" * 84)
ident = True
for k in range(5):
    seq = {}
    for tag, cls in [("buggy", AttackerEnv), ("corr", Corrected)]:
        env = cls(PFF, d_des=0.20)
        env.reset(seed=7000 + k)
        vs, r = [], None
        while True:
            obs, _r, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
            vs.append(env.env._leader_v())
            if term or trunc:
                r = env.env.term_reason
                break
        seq[tag] = (np.array(vs), r)
    same = np.array_equal(seq["buggy"][0], seq["corr"][0]) and \
        seq["buggy"][1] == seq["corr"][1]
    ident &= same
    print(f"  ep {k}: len {len(seq['buggy'][0]):>3d} vs {len(seq['corr'][0]):>3d}   "
          f"term {seq['buggy'][1]:>9s} vs {seq['corr'][1]:>9s}   identical={same}")
print(f"\n  => all episodes bit-identical: {ident}")
print("  (P+FF ignores obs[1], so the defect CANNOT have changed its behaviour)")
print("  (H-A1' needs >=60% vs BOTH; P+FF is one of the two -> the gate fails")
print("   on a measurement the defect does not touch)")

print()
print("=" * 84)
print("CHECK 2 -- learned attacker under a STRONGER (post-move) harness")
print("=" * 84)
model = load_attacker()
v1 = load_v1()
for name, pol in [("P+FF", PFF), ("v1 RL", v1)]:
    for tag, cls in [("buggy     ", AttackerEnv),
                     ("corrected ", Corrected),
                     ("post-move ", PostMove)]:
        env = cls(pol, d_des=0.20)
        crash = 0
        for k in range(20):
            obs, _ = env.reset(seed=7000 + k)
            while True:
                a, _ = model.predict(obs, deterministic=True)
                obs, _r, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            if env.env.term_reason == "collision":
                crash += 1
        print(f"  vs {name:>5s}  {tag} harness: {crash:>2d}/20 crashes")

print()
print("=" * 84)
print("Reading: if CHECK 2 stays at/near 0/20 even for post-move, then the")
print("Step 1 failure is NOT an artefact of when the defender reads the obs.")
print("=" * 84)
