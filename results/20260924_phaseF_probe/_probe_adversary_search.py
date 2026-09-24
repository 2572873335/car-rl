"""Can ANY speed-only adversary crash a COMPETENT follower?

This is the plan's load-bearing premise. The owner's H-A1 is "attacker vs the
RULE defender reaches >=60% crash rate". But an earlier probe found the rule
defender (P+FF) crashed 0/30 under the strongest SCRIPTED attack, while the v1
RL policy crashed 9/30 in one cell.

If no speed-only leader trajectory can crash a competent follower, then:
  * H-A1's 60% is unreachable against a rule defender, and
  * an adversary trained against a weak RL policy would learn to exploit THAT
    policy's specific slack, not to defeat competent following.

So: search a family of aggressive leader scripts against P+FF and against v1.
Families: hard brake cycles (swept), sustained crawl, high-frequency oscillation
(to provoke controller resonance), and a bang-bang square wave.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe


class ScriptedLeader(fe.FollowEnv):
    """Leader speed driven by an arbitrary callable of (t), frozen env intact."""

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

    fams = {}
    # hard brake cycles, swept
    for p in [1.0, 2.0, 4.0, 8.0]:
        for dur in [0.5, 1.0, 2.0]:
            fams[f"brake p={p} d={dur}"] = (
                lambda t, p=p, dur=dur: 0.9 if (t % p) >= dur else 0.05)
    # sustained crawl
    fams["crawl 0.05"] = lambda t: 0.05
    fams["crawl 0.15"] = lambda t: 0.15
    # oscillation
    for f in [0.5, 1.0, 2.0]:
        fams[f"osc f={f}"] = (lambda t, f=f: 0.5 + 0.45 * np.sin(2 * np.pi * f * t))
    # bang-bang at max slew
    fams["bang-bang 0.2s"] = lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
    fams["bang-bang 0.05s"] = lambda t: 0.95 if int(t / 0.05) % 2 == 0 else 0.05

    print("=" * 80)
    print("ADVERSARIAL LEADER SEARCH vs COMPETENT (P+FF) and WEAK (v1 RL)")
    print("=" * 80)
    print(f"  {'leader script':>22s} {'d_des':>6s} {'P+FF coll':>10s} {'v1 RL coll':>11s}")
    worst_pff = 0
    for d in [0.20, 0.50]:
        for name, fn in fams.items():
            c_r = run(pff, fn, d, n_ep=20)
            c_l = run(rl, fn, d, n_ep=20)
            worst_pff = max(worst_pff, c_r)
            print(f"  {name:>22s} {d:>6.2f} {c_r:>7d}/20 {c_l:>8d}/20")
    print()
    print(f"  worst P+FF crash rate over the whole search: {worst_pff}/20")
