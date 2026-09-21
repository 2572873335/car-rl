"""W6 design question: can the EXISTING frozen overtake checkpoint act as a
competent OPPONENT in the two-car peer world?

Why this matters: the W5 probe verdict says the league should use TRAINED
SNAPSHOTS as opponents (the scripted rule machine is not an adversary --
40/40 episodes, zero lead changes, zero inner-lane use). The natural first
league member is the checkpoint we already have:
    ckpt_ot/overtake_final_v1.zip   (10/10 overtake, 1.4 s, trained)

But that policy was trained as the FOLLOWER chasing a SLOW, outer-lane-only
leader. Dropping it into a peer race puts it far out of distribution. If it
just sits there or crashes, the W6 plan cannot assume "reuse old checkpoints"
and must budget for producing league members from scratch.

Measured, not assumed (iron rule 1).
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
from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld as World

CKPT = "/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip"


# F12 GUARD: the frozen env defines gap_ref = (s_other - s_self) % L.
# An earlier version here used (L/2 - delta) % L -- off by L/2 --
# which silently disabled the rule machine's dive branch and produced
# a FALSE positive control. Never hardcode or shortcut the gap.
# Guarded by adapter_fidelity_test.py.
def frozen_layout(o_self, L):
    """Map the two-car self-view obs into the FROZEN env's layout so the
    checkpoint's policy reads the quantities it was trained on.

    Frozen layout (overtake_env._obs):
      [ (gap_ref-0.2)/0.5, delta/2.5, v_f/1.3, e_lat/0.25, lane, v_l/0.5 ]
    Self-play layout (this project):
      [ delta/2.5, v_self/1.3, v_other/1.3, e_lat/0.25, lane, clear/2.5 ]

    The checkpoint expects to be the FOLLOWER: gap_ref is the along-track gap
    on the outer loop, delta is its signed progress difference. In a peer race
    both cars are symmetric, so we feed the same self-relative quantities --
    this is the honest "out of distribution by construction" test.
    """
    delta_n, v_self, v_other, e_lat_n, lane, clear_n = o_self
    delta = delta_n * 2.5                      # metres
    # gap_ref: along-track distance to the other car, wrapped positive
    gap_ref = (-delta) % L          # FIXED (F12): frozen env's definition
    return np.array([
        np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
        delta / 2.5,
        v_self,                                # v_f/1.3 (already normalized)
        e_lat_n,
        lane,
        v_other / 0.5 * 1.3,                   # v_l/0.5 <- v_other/1.3 rescale
    ], np.float32)


def run_episodes(model, n_ep=40, seed0=11000, opponent="frozen_ckpt",
                 ckpt_model=None):
    outer, inner = build_paths()
    stats = dict(coll=0, win=0, loss=0, tie=0, inner_use=0, lead_changes=0,
                 reasons={}, v_bar=[])
    for k in range(n_ep):
        w = World(outer, inner)
        obs = w.reset(np.random.default_rng(seed0 + k))
        prev_leader = 0 if w.delta(0) > 0 else 1
        used_inner = False
        while True:
            a0, _ = model.predict(obs[0], deterministic=True)
            if opponent == "frozen_ckpt":
                a1, _ = ckpt_model.predict(
                    frozen_layout(obs[1], w.L), deterministic=True)
            elif opponent == "rule":
                from overtake_env import baseline_action_ot
                from _verify_f5_fix import selfplay_to_frozen
                a1 = baseline_action_ot(selfplay_to_frozen(obs[1]))
            else:
                a1 = np.zeros(2)
            obs, _, done = w.step(np.stack([a0, a1]))
            if max(w.lane) == 1:
                used_inner = True
            now = 0 if w.delta(0) > 0 else 1
            if now != prev_leader:
                stats["lead_changes"] += 1
                prev_leader = now
            if done:
                break
        reasons = stats["reasons"]
        reasons[w.reason] = reasons.get(w.reason, 0) + 1
        if w.reason == "collision":
            stats["coll"] += 1
        d = w.delta(0)
        if d > 0.05:
            stats["win"] += 1
        elif d < -0.05:
            stats["loss"] += 1
        else:
            stats["tie"] += 1
        if used_inner:
            stats["inner_use"] += 1
        stats["v_bar"].append(np.mean(w.cars[0].v))
    return stats


def main():
    print("=" * 78)
    print("Can the EXISTING frozen overtake checkpoint serve as a league opponent?")
    print("=" * 78)

    ckpt = PPO.load(CKPT)
    print(f"  loaded {CKPT}")
    print(f"  policy obs space = {ckpt.observation_space}")

    # A neutral "weak" policy drives car 0 so we measure the CHECKPOINT's
    # behaviour, not our own skill.  We compare it against the rule machine
    # and against a do-nothing car on identical seeds.
    print()
    print("-" * 78)
    print("  opponent = frozen RL checkpoint   (car 1)")
    print("-" * 78)
    s = run_episodes(ckpt, n_ep=40, opponent="frozen_ckpt", ckpt_model=ckpt)
    print(f"    reasons      : {s['reasons']}")
    print(f"    collisions   : {s['coll']}/40")
    print(f"    lead changes : {s['lead_changes']}  (per 40 eps)")
    print(f"    inner lane   : used in {s['inner_use']}/40 episodes")
    print(f"    car0 mean v  : {np.mean(s['v_bar']):.3f} m/s")

    print()
    print("-" * 78)
    print("  opponent = scripted rule machine  (the W5 baseline, for contrast)")
    print("-" * 78)
    s2 = run_episodes(ckpt, n_ep=40, opponent="rule")
    print(f"    reasons      : {s2['reasons']}")
    print(f"    collisions   : {s2['coll']}/40")
    print(f"    lead changes : {s2['lead_changes']}")
    print(f"    inner lane   : used in {s2['inner_use']}/40 episodes")

    print()
    print("=" * 78)
    print("INTERPRETATION for the W6 league design")
    print("=" * 78)
    if s["inner_use"] > 10 or s["lead_changes"] > 20:
        print("  The frozen checkpoint DOES race (uses the inner lane / changes")
        print("  the lead). => it can seed the league, and W6 can start from")
        print("  frozen_ckpt + snapshots rather than from scratch.")
    else:
        print("  The frozen checkpoint does NOT meaningfully race as a peer")
        print("  (little inner-lane use, few lead changes). => it is NOT a")
        print("  ready-made league member; W6 must budget for producing")
        print("  opponents from scratch (or fine-tune it against itself).")
    print()
    print("  This is exactly the 'out of distribution by construction' risk:")
    print("  the checkpoint was trained as a follower vs a SLOW, outer-lane-only")
    print("  leader -- never as a peer racer nor as a defender.")


if __name__ == "__main__":
    main()
