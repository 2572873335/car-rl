"""Verify the attacker's reward actually favours crashing.

This check exists because the FIRST version was inverted in practice and nothing
in the training run would have revealed it -- the attacker would simply have
learned to sit still, and the 3-day gate would have failed for the wrong reason.

The test compares the total episode return of three attacker behaviours against
the SAME frozen defender:
  idle      attacker holds zero action
  cosine    attacker oscillates gently (mild disruption)
  bangbang  attacker alternates extremes (the known killer of P+FF)

Requirement: bangbang must outscore idle by a wide margin. If it does not, the
reward is mis-weighted and training must NOT start.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv, W_CRASH, W_ERR, W_IDLE


def episode(env, act_fn, seed):
    obs, _ = env.reset(seed=seed)
    tot, n = 0.0, 0
    while True:
        r = act_fn(env, obs)
        obs, rew, term, trunc, _ = env.step(r)
        tot += rew
        n += 1
        if term or trunc:
            break
    return tot, n, env.env.term_reason


def main():
    pff = lambda o: fe.baseline_action(o, use_ff=True)

    behaviours = {
        "idle    ": lambda env, o: np.array([0.0], dtype=np.float32),
        "cosine  ": lambda env, o: np.array(
            [0.8 * np.cos(2 * np.pi * env.env.t / 2.0)], dtype=np.float32),
        "bangbang": lambda env, o: np.array(
            [1.0 if int(env.env.t / 0.1) % 2 == 0 else -1.0], dtype=np.float32),
    }

    print("=" * 84)
    print("ATTACKER REWARD SCALE CHECK  (defender = frozen P+FF)")
    print(f"  W_ERR={W_ERR}  W_CRASH={W_CRASH}  W_IDLE={W_IDLE}")
    print("=" * 84)
    print(f"  {'behaviour':>10s} {'mean return':>13s} {'mean steps':>11s} "
          f"{'last term':>12s}")
    means = {}
    for name, fn in behaviours.items():
        env = AttackerEnv(pff, d_des=0.20)
        rets, ns, terms = [], [], []
        for s in range(5):
            t, n, tr = episode(env, fn, 2000 + s)
            rets.append(t)
            ns.append(n)
            terms.append(tr)
        means[name] = float(np.mean(rets))
        print(f"  {name:>10s} {np.mean(rets):>13.1f} {np.mean(ns):>11.0f} "
              f"{terms[-1]:>12s}")

    print()
    idle = means["idle    "]
    bb = means["bangbang"]
    print(f"  idle {idle:.1f}  vs  bangbang {bb:.1f}   ->  diff {bb - idle:+.1f}")
    ok = bb > idle + 100
    print(f"  crashing is favoured by a wide margin: {ok}")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL -- do not train'}")


if __name__ == "__main__":
    main()
