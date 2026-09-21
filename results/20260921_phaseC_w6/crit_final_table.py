"""FINAL mandated degenerate-reference table under the chosen design.

Design (fixes from the probes):
  - role-fixed: car 0 = follower ONLY, always starting a FIXED +0.5 m behind
    (the symmetric +-0.5 start was rejected: it handed do-nothing free wins)
  - V2V prior (self-relative leaves the positive control at 0 -- no positive pole)
  - threshold swept; 0.3 fails (degenerate policies reach 0.50), 1.0 separates
  - opponent FIXED and stated: the repaired rule machine (the M1 anchor)

Must-pass items:
  zero-action ~0 | full-throttle no wins | random ~0 | POSITIVE ckpt >> 0
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _rolefixed_probe import RoleWorld
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

GAP = 0.5


def outcomes(policy_fn, opponent_fn, n_ep=40, seed0=98000, prior="v2v"):
    out = []
    for k in range(n_ep):
        w = RoleWorld(*build_paths(), prior=prior)
        obs = w.reset(np.random.default_rng(seed0 + k), gap=GAP)
        while True:
            a0, a1 = policy_fn(obs[0], w.L), opponent_fn(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        out.append((w.delta(0), w.reason))
    return out


def wr(out, thr):
    return sum(1 for d, _ in out if d > thr) / len(out)


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    ckpf = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rng = np.random.default_rng(41)

    pols = [("zero-action", lambda o, L: np.zeros(2)),
            ("full-throttle", lambda o, L: np.array([1.0, 0.0])),
            ("random", lambda o, L: rng.uniform(-1, 1, 2)),
            ("FROZEN CKPT(+)", ckpf)]

    out = {n: outcomes(fn, rule) for n, fn in pols}

    print("=" * 82)
    print("MANDATED DEGENERATE-REFERENCE TABLE")
    print("  design: follower-only, fixed +0.5 m behind start, V2V prior")
    print("  opponent: repaired rule machine (31/40 inner-lane use)")
    print("=" * 82)
    print(f"  {'policy':>16s} {'win@0.3':>8s} {'win@1.0':>8s} {'win@1.5':>8s} "
          f"{'mean_d':>9s} {'coll%':>7s}  reasons")
    print("-" * 82)
    for n, _ in pols:
        o = out[n]
        md = float(np.mean([d for d, _ in o]))
        coll = sum(1 for _, r in o if r == "collision") / len(o)
        rs = {}
        for _, r in o:
            rs[r] = rs.get(r, 0) + 1
        print(f"  {n:>16s} {wr(o,0.3):>8.2f} {wr(o,1.0):>8.2f} {wr(o,1.5):>8.2f} "
              f"{md:>+9.3f} {100*coll:>6.1f}%  {rs}")

    print()
    print("=" * 82)
    print("DISCRIMINATION CHECK at thr=1.0")
    print("=" * 82)
    deg = max(wr(out[n], 1.0) for n in ["zero-action", "full-throttle", "random"])
    pos = wr(out["FROZEN CKPT(+)"], 1.0)
    print(f"  worst degenerate = {deg:.2f}  (must be < 0.20)")
    print(f"  positive control = {pos:.2f}  (must be > 0.50)")
    ok = deg < 0.20 and pos > 0.50
    print(f"  => {'ALL FOUR ITEMS PASS -- criterion usable' if ok else 'FAILS'}")

    print()
    print("=" * 82)
    print("DOCUMENTED CAVEATS (must appear in the plan, not hidden)")
    print("=" * 82)
    print("  1. The criterion is OPPONENT-RELATIVE. Measured: the positive")
    print("     control scores 1.00 vs the rule machine but 0.00 vs itself --")
    print("     identical policies leave the 0.5 m start gap unchanged. So a")
    print("     threshold is meaningless without naming the opponent.")
    print("  2. The 'as leader' role adds NO discrimination: as leader, a win is")
    print("     'the follower failed to get ahead', satisfied by timeout, so")
    print("     every policy scores 1.00 there. min(follower, leader) is")
    print("     therefore carried entirely by the follower column.")
    print("  3. thr=0.3 is NOT usable: on a symmetric start it gave degenerate")
    print("     policies 0.50. At thr>=1.0 the plateau 1.0-2.0 is clean.")


if __name__ == "__main__":
    main()
