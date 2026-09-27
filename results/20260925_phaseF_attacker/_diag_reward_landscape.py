"""Diagnose WHY the attacker failed to find the crash: exploration vs reward design.

The decisive question: does the KNOWN-WINNING scripted attack actually score well
under the attacker's own reward function?

  * If the script scores MUCH higher than the learned policy achieved, the reward
    is correctly shaped and the problem is EXPLORATION -- the learner never
    sampled the right action sequence.
  * If the script scores the SAME or WORSE, the reward does not point at crashing,
    and no amount of search will fix it -- the reward design is wrong.

Measures, all through the same AttackerEnv and the same reward function:
  - scripted step-down (0.1/0.2/0.4 s)  [known: crashes P+FF 20/20, v1 13-18/20]
  - the trained attacker checkpoint      [known: 0/20 both]
  - idle                                  [baseline]
  - random action                         [exploration baseline]

Also reports the crash rate each achieves, so reward and outcome can be read
together -- that is what separates "reward ignores crashes" from "reward likes
crashes but they were rarely found".
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv, V_BASE, V_SPAN


def to_action(v_l):
    """Map a desired leader speed back to the attacker action."""
    return np.clip(np.array([(v_l - V_BASE) / V_SPAN], dtype=np.float32), -1, 1)


def episode(env, act_fn, seed):
    obs, _ = env.reset(seed=seed)
    tot, n = 0.0, 0
    while True:
        a = act_fn(env, obs)
        obs, r, term, trunc, _ = env.step(a)
        tot += r
        n += 1
        if term or trunc:
            break
    return tot, n, env.env.term_reason == "collision"


def evaluate(pol, act_fn, n_ep=20, seed0=60000):
    env = AttackerEnv(pol, d_des=0.20)
    rets, ns, crashes = [], [], 0
    for k in range(n_ep):
        t, n, c = episode(env, act_fn, seed0 + k)
        rets.append(t)
        ns.append(n)
        crashes += int(c)
    return float(np.mean(rets)), float(np.mean(ns)), crashes


if __name__ == "__main__":
    from stable_baselines3 import PPO

    pff = lambda o: fe.baseline_action(o, use_ff=True)
    v1m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    v1 = lambda o: v1m.predict(o, deterministic=True)[0]

    att = PPO.load(f"{REPO}/results/20260925_phaseF_attacker/attacker_vs_both.zip")

    def mk_step(period):
        return lambda env, obs: to_action(
            0.95 if int(env.env.t / period) % 2 == 0 else 0.05)

    def mk_idle(env, obs):
        return np.array([0.0], dtype=np.float32)

    def mk_rand(env, obs):
        return np.array([env.env.rng.uniform(-1, 1)], dtype=np.float32)

    def mk_learned(env, obs):
        return att.predict(obs, deterministic=True)[0]

    print("=" * 90)
    print("ATTACKER REWARD LANDSCAPE -- does the winning script score well?")
    print("=" * 90)

    hits = {"P+FF": pff, "v1 RL": v1}
    behaviours = [
        ("step-down 0.1s", mk_step(0.1)),
        ("step-down 0.2s", mk_step(0.2)),
        ("step-down 0.4s", mk_step(0.4)),
        ("learned attacker", mk_learned),
        ("idle", mk_idle),
        ("random", mk_rand),
    ]

    for dname, pol in hits.items():
        print(f"\n  defender = {dname}")
        print(f"    {'attack':>18s} {'mean reward':>12s} {'crash':>8s} {'steps':>7s}")
        for bname, fn in behaviours:
            r, n, c = evaluate(pol, fn)
            print(f"    {bname:>18s} {r:>12.1f} {c:>4d}/20 {n:>7.0f}")
