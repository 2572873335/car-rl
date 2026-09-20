"""Isolate the OPPONENT's contribution (fixing the confound in the first test:
lane use was counted across BOTH cars, so the shared policy's own inner-lane
use falsely credited the opponent).

Question: is the existing frozen checkpoint a competent PEER RACER, or does
it only work as a follower-vs-slow-leader?

Three arms, all with car 0 = a FIXED weak reference (do-nothing), so anything
car 1 does is attributable to car 1 alone:
  arm A: car1 = frozen checkpoint
  arm B: car1 = scripted rule machine
  arm C: car1 = do-nothing (floor)
Also measures the checkpoint-vs-checkpoint deadlock directly.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from stable_baselines3 import PPO
from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World
from _ckpt_as_opponent import frozen_layout

CKPT = "/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip"


def measure(car1_fn, label, n_ep=40, seed0=12000, car0_fn=None):
    """car0 = weak reference by default; car1 = the arm under test.
    Counts ONLY car 1's lane use and lead-taking."""
    if car0_fn is None:
        car0_fn = lambda o, L: np.zeros(2)
    stats = dict(coll=0, car1_ahead_at_end=0, car1_used_inner=0,
                 car1_lead_changes=0, reasons={}, v1=[])
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        prev_leader = 0 if w.delta(0) > 0 else 1
        c1_inner = False
        while True:
            a0 = car0_fn(obs[0], w.L)
            a1 = car1_fn(obs[1], w.L)
            obs, _, done = w.step(np.stack([a0, a1]))
            if w.lane[1] == 1:                    # ONLY car 1
                c1_inner = True
            now = 0 if w.delta(0) > 0 else 1
            if now != prev_leader:
                stats["car1_lead_changes"] += 1
                prev_leader = now
            if done:
                break
        stats["reasons"][w.reason] = stats["reasons"].get(w.reason, 0) + 1
        if w.reason == "collision":
            stats["coll"] += 1
        if w.delta(0) < -0.05:                    # car 1 ahead
            stats["car1_ahead_at_end"] += 1
        if c1_inner:
            stats["car1_used_inner"] += 1
        stats["v1"].append(np.mean(w.cars[1].v))
    print(f"  {label}")
    print(f"    collisions            : {stats['coll']}/{n_ep}")
    print(f"    car1 used inner lane  : {stats['car1_used_inner']}/{n_ep}")
    print(f"    car1 lead changes     : {stats['car1_lead_changes']}")
    print(f"    car1 ahead at end     : {stats['car1_ahead_at_end']}/{n_ep}")
    print(f"    car1 mean speed       : {np.mean(stats['v1']):.3f} m/s")
    print(f"    reasons               : {stats['reasons']}")
    return stats


def main():
    ckpt = PPO.load(CKPT)
    from overtake_env import baseline_action_ot
    from _verify_f5_fix import selfplay_to_frozen

    print("=" * 78)
    print("Isolating CAR 1's behaviour (car 0 = do-nothing, so car 1 is")
    print("purely responsible for anything it does)")
    print("=" * 78)
    print()
    print("  ARM C (floor): car1 = do-nothing")
    measure(lambda o, L: np.zeros(2), "car1 = do-nothing")
    print()
    print("  ARM B: car1 = scripted rule machine")
    measure(lambda o, L: baseline_action_ot(selfplay_to_frozen(o)),
            "car1 = rule machine")
    print()
    print("  ARM A: car1 = frozen RL checkpoint  <-- the candidate league member")
    measure(lambda o, L: ckpt.predict(frozen_layout(o, L),
                                      deterministic=True)[0],
            "car1 = frozen checkpoint")

    print()
    print("=" * 78)
    print("SYMMETRIC DEADLOCK CHECK: checkpoint vs itself")
    print("=" * 78)
    r = measure(lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0],
                "car0 = car1 = checkpoint", car0_fn=lambda o, L: ckpt.predict(
                    frozen_layout(o, L), deterministic=True)[0])
    print()
    print("  (First test reported 0 lead changes and 18/40 collisions here.)")


if __name__ == "__main__":
    main()
