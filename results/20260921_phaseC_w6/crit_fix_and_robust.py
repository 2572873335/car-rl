"""Confirm the separation at the threshold the sweep pointed to, and check the
remaining worry: is thr=1.0 robustly separating, or an artefact of the
opponent being weak?

At thr=0.3 the degenerate policies scored 0.50 because on the gap = -0.5
episodes the "follower" is STARTED AHEAD, so doing nothing already leads by
0.5 > 0.3 -- a free win half the time. The +-0.5 symmetric start is therefore
NOT "remove the initial-condition variable"; for the follower role it
introduces a free-win case.

Two candidate fixes to test:
  (F1) raise the threshold to >= 1.0  (sweep suggested 1.0 works)
  (F2) drop the -0.5 half: follower ALWAYS starts a FIXED distance BEHIND
       (gap = +0.5 only), which is the honest reading of "fixed behind start"

Report both, and cross-check the positive control against a STRONGER opponent
(the trained checkpoint itself as the leader) so the criterion is not simply
measuring "beat a weak rule machine".
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _rolefixed_probe import RoleWorld
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot


def outcomes(policy_fn, opponent_fn, gaps, n_ep=40, seed0=97000, prior="v2v"):
    out = []
    for k in range(n_ep):
        gap = gaps[k % len(gaps)]
        w = RoleWorld(*build_paths(), prior=prior)
        obs = w.reset(np.random.default_rng(seed0 + k), gap=gap)
        while True:
            a0, a1 = policy_fn(obs[0], w.L), opponent_fn(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        out.append((w.delta(0), w.reason, gap))
    return out


def wr(out, thr):
    return sum(1 for d, _, _ in out if d > thr) / len(out)


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    ckpf = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rng = np.random.default_rng(31)

    pols = [("zero-action", lambda o, L: np.zeros(2)),
            ("full-throttle", lambda o, L: np.array([1.0, 0.0])),
            ("random", lambda o, L: rng.uniform(-1, 1, 2)),
            ("FROZEN CKPT(+)", ckpf)]

    print("=" * 80)
    print("FIX COMPARISON  (opponent = rule machine)")
    print("=" * 80)

    for label, gaps in [("F1: +-0.5 symmetric start (user spec, as written)",
                         [0.5, -0.5]),
                        ("F2: always +0.5 BEHIND (honest fixed-behind)",
                         [0.5])]:
        out = {n: outcomes(fn, rule, gaps) for n, fn in pols}
        print()
        print(f"  {label}")
        print(f"    {'policy':>16s} {'win@0.3':>9s} {'win@1.0':>9s} {'mean_d':>9s}")
        for n, _ in pols:
            print(f"    {n:>16s} {wr(out[n],0.3):>9.2f} {wr(out[n],1.0):>9.2f} "
                  f"{np.mean([d for d,_,_ in out[n]]):>+9.3f}")

    print()
    print("=" * 80)
    print("ROBUSTNESS: is the positive control passing only because the rule")
    print("machine is a weak leader?  Re-run F2 with the CHECKPOINT as leader.")
    print("=" * 80)
    gaps = [0.5]
    print(f"  {'policy':>16s} {'vs rule':>9s} {'vs ckpt':>9s}")
    for n, fn in pols:
        a = wr(outcomes(fn, rule, gaps), 1.0)
        b = wr(outcomes(fn, ckpf, gaps), 1.0)
        print(f"  {n:>16s} {a:>9.2f} {b:>9.2f}")
    print()
    print("  If the positive control still wins against a STRONG leader, the")
    print("  criterion measures overtaking skill, not opponent weakness.")
    print("  If it collapses, the criterion is opponent-relative and the")
    print("  acceptance threshold must be stated together with the opponent.")

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    o = {n: outcomes(fn, rule, [0.5]) for n, fn in pols}
    deg = max(wr(o[n], 1.0) for n in ["zero-action", "full-throttle", "random"])
    pos = wr(o["FROZEN CKPT(+)"], 1.0)
    print(f"  F2 (always behind) @ thr=1.0: positive={pos:.2f}, "
          f"worst degenerate={deg:.2f}")
    print(f"  {'SEPARATES' if pos > 0.5 and deg < 0.2 else 'DOES NOT SEPARATE'}")


if __name__ == "__main__":
    main()
