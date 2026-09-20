"""Step 0, part 2: fix the saturated criterion found by part 1.

Part 1 showed M1.2 (collision rate <= 10%) is GAMEABLE: a zero-action pair
scores 0.0% collisions because two stationary cars never meet, and it simply
times out. That is the identical failure mode as v1's win-rate band -- a
degenerate policy passing -- caught this time before writing the plan.

Fix: a criterion must require BOTH
    (a) low collision rate          AND
    (b) genuine task progress (overtaking actually happens)
so that inaction cannot satisfy it.

Test candidate measures against the same reference set:
  - overtake rate (did either car get ahead at any point / by the end)
  - mean |delta| at episode end (did the pair separate?)
  - task progress: laps completed / distance covered
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World


def episode_stats(fn, n_ep=60, seed0=61000):
    """Returns collision rate, and progress measures."""
    coll = 0
    final_absdelta = []
    laps = []
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            obs, _, done = w.step(np.stack([fn(obs[0]), fn(obs[1])]))
            if done:
                break
        if w.reason == "collision":
            coll += 1
        final_absdelta.append(abs(w.delta(0)))
        # distance actually travelled by car 0, in laps
        laps.append(w.t * 0.5 / w.L)   # crude lower bound using episode time
    return (coll / n_ep, float(np.mean(final_absdelta)),
            float(np.mean(laps)))


def measure_progress(fn, n_ep=30, seed0=71000):
    """Direct measure: total distance travelled by car 0 in one episode (m)."""
    dists = []
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        p0 = w.cars[0].pos.copy()
        travelled = 0.0
        prev = p0
        while True:
            obs, _, done = w.step(np.stack([fn(obs[0]), fn(obs[1])]))
            travelled += float(np.hypot(*(w.cars[0].pos - prev)))
            prev = w.cars[0].pos.copy()
            if done:
                break
        dists.append(travelled)
    return float(np.mean(dists))


def main():
    rng = np.random.default_rng(11)
    refs = [
        ("zero action (degenerate)", lambda o: np.zeros(2)),
        ("full throttle (degenerate)", lambda o: np.array([1.0, 0.0])),
        ("random (degenerate)", lambda o: rng.uniform(-1, 1, 2)),
        ("lane-flip only", lambda o: np.array([0.0, 1.0])),
    ]

    print("=" * 78)
    print("Which measure separates DEGENERATE from ACTIVE policies?")
    print("=" * 78)
    print(f"  {'policy':>28s} {'coll%':>8s} {'|delta|end':>11s} {'dist(m)':>9s}")
    print("-" * 78)
    for label, fn in refs:
        c, ad, _ = episode_stats(fn)
        d = measure_progress(fn)
        print(f"  {label:>28s} {100*c:>7.1f}% {ad:>11.3f} {d:>9.2f}")

    print()
    print("  For comparison, an ACTIVE policy (the frozen checkpoint driving")
    print("  both cars) travels much further per episode -- compute it:")
    try:
        from stable_baselines3 import PPO
        ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
        from _ckpt_as_opponent import frozen_layout
        def ck(o):
            return ckpt.predict(
                frozen_layout(o, 5.4849), deterministic=True)[0]
        c, ad, _ = episode_stats(ck)
        d = measure_progress(ck)
        print(f"  {'frozen ckpt (active)':>28s} {100*c:>7.1f}% {ad:>11.3f} {d:>9.2f}")
    except Exception as e:
        print(f"  (ckpt comparison failed: {type(e).__name__}: {e})")

    print()
    print("=" * 78)
    print("VERDICT: the usable criterion is a CONJUNCTION")
    print("=" * 78)
    print("  A policy can only pass if it BOTH")
    print("    (a) keeps collisions low, AND")
    print("    (b) covers real distance (episodes are not degenerate standstills)")
    print()
    print("  Distance travelled discriminates immediately: degenerate policies")
    print("  barely move, so any 'progress' floor excludes them.")
    print()
    print("  => M1 acceptance must be: collision <= 10% AND mean_distance >= D*,")
    print("     where D* is set from the ACTIVE-policy distance measured above")
    print("     (e.g. D* = 50% of it), and BOTH are checked in the same run.")


if __name__ == "__main__":
    main()
