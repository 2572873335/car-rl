"""Evaluate the attacker: H-A1' crash-rate measurement against both defenders.

The gate: ONE attacker must reach >=60% crash rate against the rule defender
(P+FF) AND >=60% against the v1 RL policy. 20 episodes per defender is the
sample size the plan specifies; per A-2, a result landing near the boundary is
extended to 50 seeds, and the threshold is never relaxed.

Reports the five-class termination audit (iron rule 5), not just the crash count:
an attacker could in principle "win" by driving the follower off-track or into a
lost state, and those must be visible rather than folded into a success rate.

Also reports a triviality diagnostic (owner Q1): the trajectory similarity of the
learned attacker to the known degenerate families. It is reported, NOT gated --
the two-defender requirement is the gate; this is informed disclosure.

Run: uv run python results/20260925_phaseF_attacker/eval_attacker.py --ckpt <path>
"""
import argparse
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from adversary_env import AttackerEnv

OUT = f"{REPO}/results/20260925_phaseF_attacker"


def make_defender(kind):
    if kind == "pff":
        return (lambda o: fe.baseline_action(o, use_ff=True)), "P+FF (rule)"
    from stable_baselines3 import PPO
    m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    return (lambda o: m.predict(o, deterministic=True)[0]), "v1 RL (frozen)"


def eval_attacker(model, kind, n_ep=20, seed0=7000, d_des=0.20):
    pol, label = make_defender(kind)
    env = AttackerEnv(pol, d_des=d_des)
    counts, traj = {}, []
    for k in range(n_ep):
        obs, _ = env.reset(seed=seed0 + k)
        vs = []
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            vs.append(env.env._leader_v())
            if term or trunc:
                break
        counts[env.env.term_reason] = counts.get(env.env.term_reason, 0) + 1
        traj.append(np.array(vs))
    return counts, traj, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-ep", type=int, default=20)
    a = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(a.ckpt)

    print("=" * 84)
    print("H-A1' EVALUATION -- one attacker, both defenders")
    print(f"  ckpt: {a.ckpt}")
    print("=" * 84)

    rates, all_traj = {}, {}
    for kind in ["pff", "v1"]:
        counts, traj, label = eval_attacker(model, kind, n_ep=a.n_ep)
        crash = counts.get("collision", 0)
        rate = crash / a.n_ep
        rates[kind] = rate
        all_traj[kind] = traj
        print(f"\n  vs {label}")
        print(f"    termination audit (iron rule 5): {counts}")
        print(f"    CRASH RATE: {crash}/{a.n_ep} = {rate:.0%}")

    print()
    print("=" * 84)
    thr = 0.60
    ok = all(r >= thr for r in rates.values())
    print(f"  H-A1' threshold: >= {thr:.0%} against BOTH")
    for k, r in rates.items():
        print(f"    {k:>4s}: {r:.0%}  {'PASS' if r >= thr else 'FAIL'}")
    print(f"  H-A1' VERDICT: {'PASS' if ok else 'FAIL'}")

    # boundary handling per A-2
    near = any(abs(r - thr) < 0.10 for r in rates.values())
    if near:
        print(f"\n  NOTE: a rate is within 0.10 of the threshold -- per A-2 this")
        print(f"  must be extended to 50 seeds before the verdict is final.")

    # triviality diagnostic (reported, not gated -- owner Q1)
    print()
    print("=" * 84)
    print("  TRIVIALITY DIAGNOSTIC (reported, not gated)")
    print("=" * 84)
    fams = {
        "square 0.1s": lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05,
        "square 0.2s": lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05,
        "sine 0.5Hz": lambda t: 0.5 + 0.45 * np.sin(np.pi * t),
    }
    for kind in ["pff", "v1"]:
        pol, _ = make_defender(kind)
        sig = np.mean([np.std(t) for t in all_traj[kind]])
        print(f"  vs {kind:>4s}: attacker leader-speed std {sig:.3f}")
        for fname, fn in fams.items():
            env = AttackerEnv(pol, d_des=0.20)
            obs, _ = env.reset(seed=7000)
            ref = []
            while True:
                env.env._attacker_v = float(np.clip(fn(env.env.t), 0.05, 1.0))
                d_obs = env.env._obs()
                obs, _r, term, trunc, _ = env.env.step(pol(d_obs))
                ref.append(env.env._leader_v())
                if term or trunc:
                    break
            print(f"       {fname:>12s}: std {np.std(ref):.3f}")


if __name__ == "__main__":
    main()
