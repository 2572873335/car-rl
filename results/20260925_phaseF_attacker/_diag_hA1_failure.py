"""Diagnose the H-A1' failure: why did the attacker not learn to crash?

Two candidates, and they lead to different plan branches:

  (a) The attacker's ACTION RANGE is too narrow. It can only set
      v_l = clip(0.50 + 0.45*a, 0.05, 1.0), i.e. 0.05..0.95. But the crash
      geometry found by review1/review2 needs the leader to STEP DOWN hard while
      the follower carries a closing rate. If the trained policy only produces
      gentle oscillations it cannot create that.

  (b) The TASK is genuinely hard to exploit from the leader's seat. review1's
      forced-win probe showed scripted step-downs DO crash P+FF 20/20 -- so the
      surface exists. But those scripts use the FULL range 0.95<->0.05 at fixed
      periods, and the attacker must DISCOVER that against a randomly-resampled
      defender.

This probe separates them: run the KNOWN-WINNING scripted attack through the
SAME AttackerEnv/harness the learned attacker used, and confirm it still crashes.
If it does, the surface is intact and the failure is the learner's (a search
problem). If it does not, the wrapper broke the attack surface (an env bug).
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv, V_BASE, V_SPAN

print("=" * 84)
print("H-A1' FAILURE DIAGNOSIS")
print("=" * 84)
print(f"  attacker action -> v_l = clip({V_BASE} + {V_SPAN}*a, 0.05, 1.0)")
print(f"  => reachable leader speed range: "
      f"{np.clip(V_BASE + V_SPAN*-1.0, 0.05, 1.0):.2f} .. "
      f"{np.clip(V_BASE + V_SPAN*1.0, 0.05, 1.0):.2f}")
print(f"  (a scripted step-down needs the leader to swing 0.95 <-> 0.05)")


def run_scripted(pol, period, n_ep=20, d=0.20):
    """Drive the SAME env with a full-range scripted attack, via the env's own
    attacker-action interface, so the path is identical to training."""
    env = AttackerEnv(pol, d_des=d)
    crash = 0
    for k in range(n_ep):
        env.reset(seed=60000 + k)
        while True:
            # map the desired leader speed back to the action the env expects
            want = 0.95 if int(env.env.t / period) % 2 == 0 else 0.05
            a = np.array([(want - V_BASE) / V_SPAN], dtype=np.float32)
            a = np.clip(a, -1.0, 1.0)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if env.env.term_reason == "collision":
            crash += 1
    return crash


if __name__ == "__main__":
    from stable_baselines3 import PPO
    pff = lambda o: fe.baseline_action(o, use_ff=True)
    v1m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    v1 = lambda o: v1m.predict(o, deterministic=True)[0]

    print("\n  scripted full-range attack, driven through the SAME AttackerEnv:")
    for name, pol in [("P+FF", pff), ("v1 RL", v1)]:
        for period in [0.1, 0.2, 0.4]:
            c = run_scripted(pol, period)
            print(f"    vs {name:>6s}, step-down {period}s: {c:>2d}/20 crashes")

    print("\n  => if these crash, the attack surface is intact and the learned")
    print("     attacker simply failed to find it (a search problem, branch 2).")
