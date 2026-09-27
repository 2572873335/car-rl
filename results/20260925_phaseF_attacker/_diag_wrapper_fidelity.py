"""Resolve the contradiction the reward-landscape probe just exposed.

  * D0's sweep reports ZERO RL collisions at d=0.20 v=0.55 (behaviour constant).
  * The reward-landscape probe reports idle (leader held at 0.50 constant)
    crashing v1 RL 16/20 through AttackerEnv.

Both cannot be right. If the wrapper is unfaithful, the entire Step 1 diagnosis
("search failure") is confounded and the attacker's 0/20 says nothing.

Compares, on identical seeds and identical leader speed:
  (1) PLAIN frozen FollowEnv  -- no wrapper, env.v_set set after reset, as D0 did
  (2) AttackerEnv with an idle attacker (a = 0 -> leader held at V_BASE = 0.50)
across TWO seed ranges, since D0 used 2000+ and the probes used 60000+.

Also reports the fine-grained termination reasons and mean final gap, so a
difference can be attributed rather than merely observed.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv, V_BASE


def plain(v_set, behavior, seeds):
    """D0-style: plain frozen env, v_set assigned after reset."""
    fe.D_DES = 0.20
    env = fe.FollowEnv(domain_randomize=False)
    m = load_v1()
    coll, reasons = 0, {}
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        env.v_set, env.behavior = v_set, behavior
        while True:
            a, _ = m.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        reasons[env.term_reason] = reasons.get(env.term_reason, 0) + 1
        if env.term_reason == "collision":
            coll += 1
    return coll, reasons


def wrapped(v_set_equiv, seeds, attacker=None):
    """AttackerEnv path, same leader speed, same seeds."""
    m = load_v1()
    pol = lambda o: m.predict(o, deterministic=True)[0]
    env = AttackerEnv(pol, d_des=0.20)
    coll, reasons = 0, {}
    for s in seeds:
        env.reset(seed=int(s))
        while True:
            if attacker is None:
                a = np.array([0.0], dtype=np.float32)
            else:
                a = attacker(env, None)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        reasons[env.env.term_reason] = reasons.get(env.env.term_reason, 0) + 1
        if env.env.term_reason == "collision":
            coll += 1
    return coll, reasons


_V1 = None


def load_v1():
    global _V1
    if _V1 is None:
        from stable_baselines3 import PPO
        _V1 = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    return _V1


if __name__ == "__main__":
    fe.D_DES = 0.20
    print(f"  AttackerEnv V_BASE = {V_BASE} (idle attacker holds leader at this speed)")
    print("=" * 88)
    for label, seeds in [("seeds 2000+", list(range(2000, 2020))),
                         ("seeds 60000+", list(range(60000, 60020)))]:
        print(f"\n  {label}")
        c1, r1 = plain(0.50, "constant", seeds)
        print(f"    PLAIN env, v_set=0.50 constant : coll {c1:>2d}/20  {r1}")
        c2, r2 = wrapped(0.50, seeds)
        print(f"    AttackerEnv, idle attacker     : coll {c2:>2d}/20  {r2}")
        c3, r3 = plain(0.55, "constant", seeds)
        print(f"    PLAIN env, v_set=0.55 constant : coll {c3:>2d}/20  {r3}  (D0's boundary cell)")

    print("\n" + "=" * 88)
    print("  READ: if PLAIN and AttackerEnv disagree, the wrapper is unfaithful and")
    print("  the Step 1 diagnosis is confounded.")
