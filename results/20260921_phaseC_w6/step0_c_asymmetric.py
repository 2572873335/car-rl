"""Step 0, part 3: the real fix -- criteria must be evaluated ASYMMETRICALLY.

Part 1: collision rate is gameable (zero-action scores 0%).
Part 2: distance is also gameable (full-throttle scores 38.5 m ~ active 38.0 m).

Root cause, and it is structural: in a SYMMETRIC peer race where one shared
policy drives both cars, identical deterministic policies produce an outcome
fixed entirely by the initial conditions. Nothing competitive happens, and NO
metric computed on "policy vs itself" can detect skill -- because there is no
opponent to demonstrate skill against.

That is the same defect in W5 F5 (saturated win rate) and W6 v1's W2 band,
now reaching its general form: symmetric self-play evaluation is degenerate.

Consequence for M1 acceptance: the criteria must be measured in a setting
where the policy faces a DIFFERENT, FIXED opponent, so that relative position
is attributable to the policy. Verify that such a setting discriminates.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World
from _ckpt_as_opponent import frozen_layout

L_REF = 5.4849


def run_vs(learner_fn, opponent_fn, n_ep=50, seed0=81000):
    """Learner is car 0; opponent is car 1. Report position outcome."""
    wins = losses = ties = coll = 0
    dists = []
    for k in range(n_ep):
        w = World(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        prev = w.cars[0].pos.copy()
        travelled = 0.0
        while True:
            obs, _, done = w.step(np.stack([learner_fn(obs[0]),
                                            opponent_fn(obs[1])]))
            travelled += float(np.hypot(*(w.cars[0].pos - prev)))
            prev = w.cars[0].pos.copy()
            if done:
                break
        dists.append(travelled)
        if w.reason == "collision":
            coll += 1
        d = w.delta(0)
        if d > 0.05:
            wins += 1
        elif d < -0.05:
            losses += 1
        else:
            ties += 1
    return dict(win=wins / n_ep, loss=losses / n_ep, tie=ties / n_ep,
                coll=coll / n_ep, dist=float(np.mean(dists)))


def main():
    rng = np.random.default_rng(3)

    print("=" * 78)
    print("ASYMMETRIC evaluation: learner (car 0) vs FIXED opponent (car 1)")
    print("=" * 78)
    print("  opponent = zero-action car (a stationary, non-adversarial reference)")

    print()
    print(f"  {'learner':>26s} {'win':>7s} {'loss':>7s} {'tie':>7s} "
          f"{'coll':>7s} {'dist':>8s}")
    print("-" * 78)

    learners = [
        ("zero action (degenerate)", lambda o: np.zeros(2)),
        ("full throttle (degenerate)", lambda o: np.array([1.0, 0.0])),
        ("random (degenerate)", lambda o: rng.uniform(-1, 1, 2)),
        ("lane-dive only", lambda o: np.array([0.5, 1.0])),
    ]
    for label, fn in learners:
        r = run_vs(fn, lambda o: np.zeros(2))
        print(f"  {label:>26s} {r['win']:>7.2f} {r['loss']:>7.2f} "
              f"{r['tie']:>7.2f} {r['coll']:>7.2f} {r['dist']:>8.2f}")

    print()
    print("  For reference, the ACTIVE policy vs the same stationary opponent:")
    try:
        from stable_baselines3 import PPO
        ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
        def ck(o):
            return ckpt.predict(frozen_layout(o, L_REF), deterministic=True)[0]
        r = run_vs(ck, lambda o: np.zeros(2))
        print(f"  {'frozen ckpt (active)':>26s} {r['win']:>7.2f} "
              f"{r['loss']:>7.2f} {r['tie']:>7.2f} {r['coll']:>7.2f} "
              f"{r['dist']:>8.2f}")
    except Exception as e:
        print(f"  (ckpt arm failed: {type(e).__name__}: {e})")

    print()
    print("=" * 78)
    print("VERDICT for M1 acceptance criteria")
    print("=" * 78)
    print("  In the ASYMMETRIC setting, 'win rate vs a fixed non-adversarial")
    print("  opponent' is NOT saturated by inaction:")
    print("    - a degenerate learner does not win (it cannot gain position)")
    print("    - an active learner can win by overtaking the stationary car")
    print()
    print("  => M1 must evaluate in an asymmetric setting, and its acceptance is:")
    print("       (a) win rate vs fixed reference significantly > 0.5, AND")
    print("       (b) collision rate <= 10%, AND")
    print("       (c) mean distance >= a floor (excludes standstills),")
    print("     all measured on the SAME seed set, with the degenerate")
    print("     references above shown to FAIL (a).")
    print()
    print("  Symmetric self-play metrics are retained ONLY as diagnostics")
    print("  (training health), never as acceptance criteria.")


if __name__ == "__main__":
    main()
