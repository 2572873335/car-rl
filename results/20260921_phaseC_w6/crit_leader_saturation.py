"""Why is "as leader" 1.00 for EVERYONE, including zero-action?

Suspicion: as leader, the episode commonly ends in COLLISION (the follower
rams the stationary/slow leader) or times out, and both count as "leader did
not get overtaken" -> leader "wins". If so, the leader column is a degenerate
win path and min() is effectively just the follower column.

Also check the opposite: does the FROZEN CKPT's follower=1.00 come from real
overtaking, or from the opponent crashing (which would also count as... no,
crash means follower NOT ahead, so follower would LOSE). Verify.

Report termination-reason distributions per role.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _rolefixed_probe import RoleWorld, WIN_MARGIN
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot


def run(policy_fn, opponent_fn, prior, as_role, n_ep=40, seed0=91000, gap=0.5):
    reasons, margins, follower_ahead_ct = {}, [], 0
    for k in range(n_ep):
        w = RoleWorld(*build_paths(), prior=prior)
        obs = w.reset(np.random.default_rng(seed0 + k), gap=gap)
        while True:
            if as_role == "follower":
                a0, a1 = policy_fn(obs[0], w.L), opponent_fn(obs[1], w.L)
            else:
                a0, a1 = opponent_fn(obs[0], w.L), policy_fn(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        reasons[w.reason] = reasons.get(w.reason, 0) + 1
        d = w.delta(0)                     # car 0 = follower by construction
        margins.append(d)
        if d > WIN_MARGIN:
            follower_ahead_ct += 1
    return reasons, np.mean(margins), follower_ahead_ct / n_ep


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    zero = lambda o, L: np.zeros(2)
    ckpf = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]

    print("=" * 78)
    print("Termination reasons by role (V2V prior, follower starts 0.5 m behind)")
    print("=" * 78)

    for label, pol in [("zero-action", zero), ("FROZEN CKPT", ckpf)]:
        for role in ["follower", "leader"]:
            r, mean_delta, ahead = run(pol, rule, "v2v", role)
            print(f"  {label:>12s} as {role:8s}: reasons={r}  "
                  f"mean_delta={mean_delta:+.3f}  follower_ahead={ahead:.2f}")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print("  'as leader' counts a win whenever the follower does NOT finish")
    print("  > +0.3 ahead. A collision or timeout satisfies that trivially.")
    print("  If the leader column is dominated by collision/timeout, it is a")
    print("  DEGENERATE win path -> the min() is carried entirely by the")
    print("  follower column.  That is acceptable for a criterion (min still")
    print("  excludes degenerate policies) but must be stated, because it")
    print("  means 'as leader' contributes no discrimination of its own.")
    print()
    print("  The FROZEN CKPT follower column must come from REAL overtaking")
    print("  (reasons 'timeout' with delta > 0.3), not from crashes.")


if __name__ == "__main__":
    main()
