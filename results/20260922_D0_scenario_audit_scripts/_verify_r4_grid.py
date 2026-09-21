"""Verify review R4: does RL lose to P+FF at v=1.0 m/s?

This is the load-bearing claim for the audit's value: if the headline
"RL 2x better than hand-tuned" REVERSES exactly on the AGV-relevant grid
(1.0 m/s), that is a major finding the plan neither predicted nor listed.

Method: monkeypatch D_DES and v_set at runtime (no frozen-file edits), run the
frozen checkpoint vs baseline_action(use_ff=True) at several grids.

Report both unit conventions explicitly, since the review flagged that the
plan never defined them:
  obs-cm  = 200 * |e| [m]   (what every existing repo number uses)
  true-cm = 100 * |e| [m]
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np


def run(policy, d_des, v_set, behavior="constant", n_ep=30, seed0=2000):
    import follow_env as fe
    fe.D_DES = d_des
    env = fe.FollowEnv(domain_randomize=False)

    # monkeypatch reset to force v_set / behavior
    orig_reset = env.reset

    def forced_reset(seed=None, options=None):
        obs, info = orig_reset(seed=seed)
        env.v_set = v_set
        env.behavior = behavior
        return obs, info

    env.reset = forced_reset

    errs_true = []
    coll = 0
    for k in range(n_ep):
        obs, _ = env.reset(seed=seed0 + k)
        while True:
            a = policy(obs, env)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if env.term_reason == "collision":
            coll += 1
        g = env.log["gap"]
        # true |gap - d_des|, averaged
        errs_true.append(np.mean(np.abs(np.array(g) - d_des)))
    return float(np.mean(errs_true)), coll / n_ep, env.term_reason


def main():
    from stable_baselines3 import PPO
    model = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")

    def rl(obs, env):
        return model.predict(obs, deterministic=True)[0]

    def pff(obs, env):
        import follow_env as fe
        return fe.baseline_action(obs, use_ff=True)

    print("=" * 82)
    print("R4 VERIFICATION: RL vs P+FF across grids")
    print("  obs-cm = 200*|e| [m] (repo convention)   true-cm = 100*|e| [m]")
    print("=" * 82)
    print(f"  {'grid':>24s} {'RL true':>9s} {'RL obs':>8s} {'PFF true':>10s} "
          f"{'PFF obs':>8s} {'RL wins?':>9s} {'coll':>6s}")

    grids = [
        (0.20, 0.30, "constant"),   # as-trained
        (0.20, 0.50, "constant"),
        (0.50, 0.55, "constant"),
        (0.50, 1.00, "constant"),   # AGV-relevant
        (0.50, 1.00, "sinusoid"),
        (1.00, 1.00, "constant"),
    ]
    for d, v, b in grids:
        rt, rc, _ = run(rl, d, v, b)
        pt, pc, _ = run(pff, d, v, b)
        wins = "RL" if rt < pt else "P+FF"
        print(f"  {f'd={d} v={v} {b[:6]}':>24s} {rt*100:>8.2f} {rt*200:>8.2f} "
              f"{pt*100:>9.2f} {pt*200:>8.2f} {wins:>9s} "
              f"{rc:.0%}/{pc:.0%}")

    print()
    print("READING: if RL loses at d=0.50 v=1.00, R4 is confirmed -- the")
    print("headline reverses on the AGV-relevant grid.")

    import follow_env as fe
    fe.D_DES = 0.20


if __name__ == "__main__":
    main()
