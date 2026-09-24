"""Is a crash reachable at all, for a COMPETENT defender?

The earlier probe found P+FF never crashes under scripted braking (0/6 cells).
That matters enormously for the adversary proposal: if a competent defender
cannot be driven to crash by the strongest scripted script, the adversary has
nothing to learn -- the task would be vacuous.

The D0 record says the RL checkpoint crashed 6/10 in one cell (d=0.20, v=1.0,
brake) while P+FF stayed at 0/10. If that is right, the crash is a property of a
WEAK policy, not a reachable state of a competent one -- and an adversary trained
against a weak defender would learn to beat weakness, not to defeat competence.

Measure both, in the cell where the record says a difference exists.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe


def run(policy, d_des, v_set, behavior, n_ep=30, seed0=7000, dr=False):
    fe.D_DES = d_des
    env = fe.FollowEnv(domain_randomize=dr)
    try:
        coll, reasons = 0, {}
        for k in range(n_ep):
            obs, _ = env.reset(seed=seed0 + k)
            env.v_set, env.behavior = v_set, behavior
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            rr = env.term_reason
            reasons[rr] = reasons.get(rr, 0) + 1
            if rr == "collision":
                coll += 1
    finally:
        fe.D_DES = 0.20
    return coll, reasons


if __name__ == "__main__":
    from stable_baselines3 import PPO
    m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")

    pols = {
        "P+FF (rule)": lambda o: fe.baseline_action(o, use_ff=True),
        "v1 RL      ": lambda o: m.predict(o, deterministic=True)[0],
    }
    print("=" * 84)
    print("CRASH REACHABILITY UNDER SCRIPTED BRAKING")
    print("=" * 84)
    print(f"  {'cell':>26s} {'controller':>12s} {'coll/n':>9s}  reasons")
    for d, v in [(0.20, 1.00), (0.20, 0.55), (0.50, 1.00)]:
        for name, pol in pols.items():
            coll, reasons = run(pol, d, v, "brake", n_ep=30)
            print(f"  {f'd={d} v={v} brake':>26s} {name:>12s} "
                  f"{coll:>4d}/30  {reasons}")
    fe.D_DES = 0.20
    print()
    print("  READING: if P+FF stays at 0 and RL crashes, the crash is a property")
    print("  of a weak policy, not a reachable state of a competent one.")
