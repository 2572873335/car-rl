"""Decisive premise check for the crash-adversary plan: does the FOLLOWER task
have a meaningful attack surface at all?

The owner's design targets the FOLLOWING task (跟车方): an adversary trains a
leader to crash the follower; H-A1 requires the adversary to reach >=60% crash
vs the rule defender.

The prior probes already found: scripted braking crashes the v1 RL policy
(9-20/30) but NEVER the rule defender (0/30) in the cells tested -- the rule
defender survives every attack. If no attack family reaches >=60% vs the RULE
defender, H-A1 is unreachable as stated and the plan's attacker-vs-defender
evaluation has no positive pole.

But that finding is about the SCRIPTED attack. The question the plan must answer
is whether a LEARNED leader can do better. Before designing a 5M-step adversary
training run, bound what is even possible: the rule defender's braking authority
vs the worst achievable leader deceleration.

The frozen dynamics: leader and follower both have a_max in [1.0, 2.0] (dr) or
1.5 (nominal). A leader that brakes harder than the follower can decelerate
cannot close the gap by itself. So the attack surface is a RACE: can the leader
force the follower into a state where IT must decelerate harder than its own
a_max?

This probe measures the rule defender's crash rate across the space of
leader scripts, and asks whether ANY script reaches >=60% -- i.e. whether H-A1
has a positive pole at all. 20 episodes per cell is the same sample size the
plan will use for hypothesis tests.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe


class ScriptedLeader(fe.FollowEnv):
    def set_fn(self, fn):
        self._fn = fn

    def _leader_v(self):
        return float(np.clip(self._fn(self.t), 0.05, 1.0))


def run(policy, fn, d_des, n_ep=20, seed0=2000):
    fe.D_DES = d_des
    env = ScriptedLeader(domain_randomize=False)
    try:
        coll = 0
        for k in range(n_ep):
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            if env.term_reason == "collision":
                coll += 1
        return coll
    finally:
        fe.D_DES = 0.20


if __name__ == "__main__":
    from stable_baselines3 import PPO
    m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    pff = lambda o: fe.baseline_action(o, use_ff=True)
    rl = lambda o: m.predict(o, deterministic=True)[0]

    # families that stress the FOLLOWER's braking authority
    fams = {}
    # sawtooth: leader instantly drops to 0.05 then ramps -- the follower must
    # react to an unbounded jerk every cycle
    for p in [1.0, 2.0, 4.0]:
        fams[f"sawtooth p={p}"] = (lambda t, p=p:
                                   0.05 + 0.85 * ((t % p) / p))
    # square waves of varying dwell (the follower must survive a step-down)
    for dwell in [0.1, 0.2, 0.4, 0.8]:
        fams[f"step-down {dwell}s"] = (
            lambda t, dwell=dwell: 0.95 if (int(t / dwell) % 2 == 0) else 0.05)
    # sustained high speed (no brake -- the follower must keep up, then the
    # leader suddenly decelerates)
    fams["cruise-then-stop"] = lambda t: 0.95 if t < 10 else 0.05
    # sinusoid at the follower's natural resonance (already found: f=0.5
    # crashes v1 RL 20/20 at d=0.2)
    fams["osc f=0.5"] = lambda t: 0.5 + 0.45 * np.sin(np.pi * t)

    print("=" * 78)
    print("POSITIVE-POLE CHECK: does ANY leader script reach >=60% vs rule defender?")
    print("=" * 78)
    best = {}
    for d in [0.20, 0.50]:
        print(f"  d_des={d}:")
        for name, fn in fams.items():
            c_p = run(pff, fn, d, n_ep=20)
            c_l = run(rl, fn, d, n_ep=20)
            print(f"    {name:>20s}  P+FF {c_p:>2d}/20   v1 RL {c_l:>2d}/20")
            best[name] = max(best.get(name, 0), c_p)
    print()
    reachable = [n for n, c in best.items() if c >= 12]   # >=60%
    print(f"  scripts reaching >=60% vs P+FF: {reachable if reachable else 'NONE'}")
    print(f"  (>=60% of 20 = 12 crashes)")
