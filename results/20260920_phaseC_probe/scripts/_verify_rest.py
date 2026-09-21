"""Verify the reviewer's remaining claims, in one pass.

B2b: with the CORRECT adapter, is the frozen checkpoint still a bad opponent?
     (reviewer says the "22/40 collisions" and "floor" columns survive, but
      "0/40 inner lane" and "0 lead changes" were artifacts)
B3:  is the frozen checkpoint really 10/10 in the ORIGINAL frozen env?
B4:  are degenerate policies (do-nothing/random) really inside [0.35,0.65]?
"""
import sys

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import numpy as np
from stable_baselines3 import PPO
from overtake_env import OvertakeEnv, baseline_action_ot
from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World

CKPT = "/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip"


def frozen_layout_true(o, L):
    """CORRECTED adapter: gap_ref = (-delta) % L  (frozen env's definition)."""
    delta = o[0] * 2.5
    gap_ref = (-delta) % L
    return np.array([np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
                     o[0], o[1], o[3], o[4], o[2]], np.float32)


def main():
    ckpt = PPO.load(CKPT)
    print("=" * 78)
    print("B3: frozen checkpoint in its ORIGINAL env (task mode, seeds 2000-2009)")
    print("=" * 78)
    print(f"  checkpoint observation_space = {ckpt.observation_space}")
    succ = coll = 0
    for s in range(2000, 2010):
        env = OvertakeEnv(domain_randomize=False)
        obs, _ = env.reset(seed=s)
        while True:
            a, _ = ckpt.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if env.term_reason == "success":
            succ += 1
        elif env.term_reason == "collision":
            coll += 1
    print(f"  result: success={succ}/10  collision={coll}/10  -> "
          f"{'10/10 CONFIRMED' if succ == 10 else 'NOT 10/10'}")

    print()
    print("=" * 78)
    print("B2b: frozen checkpoint as car1 vs do-nothing car0, CORRECT adapter")
    print("=" * 78)

    def arm(car1_fn, label, n_ep=40, seed0=12000):
        coll = inner = lead = floor = 0
        for k in range(n_ep):
            w = World(*build_paths())
            obs = w.reset(np.random.default_rng(seed0 + k))
            prev = 0 if w.delta(0) > 0 else 1
            c1_inner = False
            while True:
                a0 = np.zeros(2)
                a1 = car1_fn(obs[1], w.L)
                obs, _, done = w.step(np.stack([a0, a1]))
                if w.lane[1] == 1:
                    c1_inner = True
                now = 0 if w.delta(0) > 0 else 1
                if now != prev:
                    lead += 1
                    prev = now
                if done:
                    break
            if w.reason == "collision":
                coll += 1
            if c1_inner:
                inner += 1
            if w.delta(0) < -0.05:
                floor += 1
        print(f"  {label:34s}: coll={coll:2d}/40  car1_inner={inner:2d}/40  "
              f"lead_chg={lead:3d}  car1_ahead={floor:2d}/40")

    arm(lambda o, L: ckpt.predict(frozen_layout_true(o, L), deterministic=True)[0],
        "ckpt (CORRECT adapter)")
    arm(lambda o, L: baseline_action_ot(frozen_layout_true(o, L)),
        "rule machine (CORRECT adapter)")
    arm(lambda o, L: np.zeros(2), "do-nothing (floor reference)")

    print()
    print("=" * 78)
    print("B4: are degenerate policies inside the proposed [0.35, 0.65] band?")
    print("=" * 78)

    def winrate(car0_fn, car1_fn, n_ep=60, seed0=31000):
        wins = 0
        for k in range(n_ep):
            w = World(*build_paths())
            obs = w.reset(np.random.default_rng(seed0 + k))
            while True:
                obs, _, done = w.step(np.stack([car0_fn(obs[0], w.L),
                                                car1_fn(obs[1], w.L)]))
                if done:
                    break
            if w.delta(0) > 0.05:
                wins += 1
        return wins / n_ep

    rng = np.random.default_rng(5)
    grid = [
        ("do-nothing vs do-nothing", lambda o, L: np.zeros(2), lambda o, L: np.zeros(2)),
        ("random vs random", lambda o, L: rng.uniform(-1, 1, 2),
         lambda o, L: rng.uniform(-1, 1, 2)),
        ("do-nothing vs ckpt", lambda o, L: np.zeros(2),
         lambda o, L: ckpt.predict(frozen_layout_true(o, L), deterministic=True)[0]),
        ("do-nothing vs rule", lambda o, L: np.zeros(2),
         lambda o, L: baseline_action_ot(frozen_layout_true(o, L))),
    ]
    for label, f0, f1 in grid:
        wr = winrate(f0, f1)
        inband = "IN BAND (bad)" if 0.35 <= wr <= 0.65 else "outside"
        print(f"  {label:26s}: win rate = {wr:.2f}   {inband}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("  If degenerate policies land inside [0.35,0.65], W2 has no")
    print("  discriminating power (same saturation failure W5 F5 identified).")


if __name__ == "__main__":
    main()
