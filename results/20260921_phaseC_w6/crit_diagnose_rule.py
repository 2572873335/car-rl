"""Diagnose: why does the rule machine score FULL=0 / 100% collision here,
when it succeeds (10/10) in the frozen env?

The mismatch must be in the world, not the policy. Compare the two step
functions field by field, and instrument a single episode to see WHERE the
collision happens (car 1 is the leader; the collision is at t=0? later?).

Key suspects:
  - my leader is driven with `leader_action` = zeros, and its speed is
    `lead_v + lead_bias*a1` = lead_v. In the frozen env the leader follows the
    same path with PurePursuit. Check the steering path matches.
  - my follower starts at gap 0.30-0.90; frozen env gap0 ~ U(0.30, 1.0). Fine.
  - the CRITICAL one: which path does each car follow? In the frozen env, the
    leader is on the OUTER loop and the follower may switch to INNER. Here
    WorldV2 does the same... but check the initial lateral offset: my cars
    both start on the outer loop, and the follower must move to the inner lane
    at a lateral separation of 0.15. If instead both sit on the outer path
    when lane=1 is chosen, they would overlap.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _criterion_v3 import WorldV3
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot, OvertakeEnv

ZERO2 = np.zeros(2)


def instrument(seed=123000, n_steps=400):
    outer, inner = build_paths()
    w = WorldV3(outer, inner, prior="v2v")
    obs = w.reset(np.random.default_rng(seed))
    print(f"  initial gap0 = {w.gap0:.3f}  leader cruise = {w.lead_v:.3f}")
    print(f"  car0 (follower) pos = {w.cars[0].pos}, lane={w.lane[0]}")
    print(f"  car1 (leader)   pos = {w.cars[1].pos}, lane={w.lane[1]}")
    print()
    print(f"  {'t':>6s} {'d':>8s} {'v0':>6s} {'v1':>6s} {'lane0':>6s} "
          f"{'dist2d':>8s} {'gap':>7s}")
    for k in range(n_steps):
        a0 = baseline_action_ot(frozen_layout(obs[0], w.L))
        obs, done = w.step(np.stack([a0, ZERO2]), leader_action=ZERO2)
        if k % 20 == 0 or k < 3 or done:
            d2d = float(np.hypot(*(w.cars[0].pos - w.cars[1].pos)))
            print(f"  {w.t:>6.2f} {w.delta(0):>+8.3f} {w.cars[0].v:>6.2f} "
                  f"{w.cars[1].v:>6.2f} {w.lane[0]:>6d} {d2d:>8.3f} "
                  f"{w.gap0:>7.3f}")
        if done:
            print(f"  -> terminated: {w.reason} at t={w.t:.2f}")
            break


def compare_with_frozen():
    """Run the rule machine in the FROZEN env to confirm it still works."""
    print()
    print("=" * 74)
    print("CONTROL: rule machine in the FROZEN env (seeds 2000-2009)")
    print("=" * 74)
    ok = 0
    for s in range(2000, 2010):
        env = OvertakeEnv(domain_randomize=False)
        obs, _ = env.reset(seed=s)
        while True:
            obs, _, term, trunc, _ = env.step(baseline_action_ot(obs))
            if term or trunc:
                break
        if env.term_reason == "success":
            ok += 1
    print(f"  frozen-env success = {ok}/10  (expect 10/10)")


def main():
    print("=" * 74)
    print("INSTRUMENTED EPISODE: rule machine as follower, my world")
    print("=" * 74)
    instrument()
    compare_with_frozen()


if __name__ == "__main__":
    main()
