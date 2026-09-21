"""Confirm the positive pole (review2 evidence).

At leader band 0.15-0.30 (the band the checkpoint was trained against):
    ckpt  FULL = 1.00, collisions 0%
    loiter FULL = 0.00
    rule  FULL = 0.00  (but ot=1.00, coll 0% -- it overtakes, never returns)

Verify across independent seeds and more episodes, and check the checkpoint's
win comes from real manoeuvring (not e.g. the leader being trivially slow).

Also honestly flag the design tension this creates:
  choosing the band that suits the checkpoint means the world's leader is the
  same slow leader as the frozen env -- so the criterion risks measuring the
  ORIGINAL task rather than a peer-race skill. Must be stated.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _criterion_v3 import WorldV3
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

ZERO2 = np.zeros(2)


class WT(WorldV3):
    def __init__(self, outer, inner, lead_lo, lead_hi):
        super().__init__(outer, inner, prior="v2v")
        self.lead_lo, self.lead_hi = lead_lo, lead_hi

    def reset(self, rng):
        obs = super().reset(rng)
        self.lead_v = float(rng.uniform(self.lead_lo, self.lead_hi))
        return obs


def run(pol, seed0, n_ep=30, lo=0.15, hi=0.30):
    outer, inner = build_paths()
    fulls, ot, colls, ts = [], [], 0, []
    for k in range(n_ep):
        w = WT(outer, inner, lo, hi)
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0 = pol(obs[0], w.L)
            obs, done = w.step(np.stack([a0, ZERO2]), leader_action=ZERO2)
            if done:
                break
        fulls.append(w.m_full())
        if not np.isnan(w.t_first_ahead):
            ot.append(1.0); ts.append(w.t_first_ahead)
        else:
            ot.append(0.0)
        if w.reason == "collision":
            colls += 1
    return np.mean(fulls), np.mean(ot), colls / n_ep, np.mean(ts) if ts else float("nan")


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))

    print("=" * 80)
    print("POSITIVE POLE VERIFICATION  (leader band 0.15-0.30, FIXED adapter)")
    print("=" * 80)
    print(f"  {'seed0':>8s} | {'ckpt: FULL':>11s} {'ot':>5s} {'coll':>6s} "
          f"| {'loiter: FULL':>13s} | {'rule: FULL':>11s}")
    print("-" * 80)
    for seed0 in [123000, 200000, 300000]:
        cf, co, cc, ct = run(ck, seed0)
        lf, lo_, lc, _ = run(loiter, seed0)
        rf, ro, rc, _ = run(rule, seed0)
        print(f"  {seed0:>8d} | {cf:>11.2f} {co:>5.2f} {cc:>5.0%} "
              f"| {lf:>13.2f} | {rf:>11.2f}")

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    cf, co, cc, ct = run(ck, 123000, n_ep=40)
    lf, _, _, _ = run(loiter, 123000, n_ep=40)
    print(f"  positive (ckpt) : FULL={cf:.2f} ot={co:.2f} coll={cc:.0%} "
          f"mean_t_ot={ct:.2f}s")
    print(f"  loiter          : FULL={lf:.2f}")
    if cf >= 0.8 and lf <= 0.1:
        print()
        print("  => VALID POSITIVE EXISTS. The criterion separates the")
        print("     positive control from every degenerate policy, including")
        print("     loiter (which defeated criterion v1).")
        print()
        print("  HONEST DESIGN TENSION (must be recorded, not hidden):")
        print("     the positive only stands at the band the checkpoint was")
        print("     trained for (slow leader). At faster leader bands it")
        print("     collapses. So this criterion measures a TASK-LEVEL")
        print("     overtake against a slow leader -- NOT a peer-race skill.")
        print("     It is legitimate for M1 ('can the league learn the task?')")
        print("     but it does NOT establish competitive/self-play ability.")
    else:
        print("  => positive pole not established at this setting.")


if __name__ == "__main__":
    main()
