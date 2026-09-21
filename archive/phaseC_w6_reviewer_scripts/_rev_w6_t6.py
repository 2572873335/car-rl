"""Reviewer tests T6/T7 -- are the "rule machine is not an adversary" and
"checkpoint is at the floor" conclusions ADAPTER ARTIFACTS?

The adapter selfplay_to_frozen (in _verify_f5_fix.py:49-53) HARDCODES
slot0 = (1.0-0.2)/0.5 = 1.6  => the rule machine always reads gap = 1.0 m.
baseline_action_ot's dive branch requires gap < 0.45 and its brake branch
requires gap < 0.15 -- so with a faked gap=1.0 BOTH branches are disabled.

T6: rule vs rule with (a) the hardcoded-gap adapter, (b) a TRUE-gap adapter
    reconstructed from the peer obs.
T7: checkpoint as car1 with (a) the plan's frozen_layout (off by L/2),
    (b) a true-gap frozen_layout -- measure lead changes and inner use.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld, selfplay_to_frozen
from overtake_env import baseline_action_ot
from stable_baselines3 import PPO

CKPT = "/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip"


def true_gap_layout(o_self, L=5.484928):
    """Frozen layout with a CORRECT gap.  Peer obs slot0 = -d/2.5 = delta_self/2.5.
    Frozen slot0 wants (gap_ref-0.2)/0.5 with gap_ref = forward dist self->other
    = d % L = (-delta_self) % L."""
    delta_n, v_self, v_other, e_lat_n, lane, clear_n = o_self
    delta = delta_n * 2.5
    L = 5.484928  # measured outer length
    gap_ref = (-delta) % L
    return np.array([np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
                     delta_n, v_self, e_lat_n, lane, v_other], np.float32)


def run_rule_vs_rule(mapper, label, n_ep=40, seed0=4000):
    ever_inner = 0
    lead = []
    for k in range(n_ep):
        w = NeutralWorld(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        L = w.L
        prev = 0 if w.delta(0) > 0 else 1
        sw = 0
        inner = False
        while True:
            a = np.stack([baseline_action_ot(mapper(obs[0], L)),
                          baseline_action_ot(mapper(obs[1], L))])
            obs, _, done = w.step(a)
            if max(w.lane) == 1:
                inner = True
            now = 0 if w.delta(0) > 0 else 1
            if now != prev:
                sw += 1
                prev = now
            if done:
                break
        lead.append(sw)
        if inner:
            ever_inner += 1
    print(f"  [{label:26s}] ever-inner {ever_inner}/{n_ep}  "
          f"lead-change mean={np.mean(lead):.2f} zero={sum(1 for x in lead if x==0)}/{n_ep}")


def hardcoded(o, L):
    return selfplay_to_frozen(o)


def T6():
    print("=" * 78)
    print("T6: rule vs rule -- adapter gap fidelity matters?")
    print("=" * 78)
    run_rule_vs_rule(hardcoded, "hardcoded gap=1.0 (as used)")
    run_rule_vs_rule(true_gap_layout, "TRUE gap (corrected)")


def T7():
    print()
    print("=" * 78)
    print("T7: checkpoint as car1, car0=do-nothing -- adapter fidelity")
    print("=" * 78)
    ck = PPO.load(CKPT)

    def plan_adapter(o, L):
        delta_n, v_self, v_other, e_lat_n, lane, clear_n = o
        delta = delta_n * 2.5
        gap_ref = (L / 2.0 - delta) % L
        return np.array([np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
                         delta / 2.5, v_self, e_lat_n, lane,
                         v_other / 0.5 * 1.3], np.float32)

    def true_adapter(o, L):
        delta_n, v_self, v_other, e_lat_n, lane, clear_n = o
        delta = delta_n * 2.5
        gap_ref = (-delta) % L
        return np.array([np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
                         delta / 2.5, v_self, e_lat_n, lane,
                         v_other / 0.5 * 1.3], np.float32)

    for label, mp in [("plan gap (L/2 off)", plan_adapter),
                      ("true gap", true_adapter)]:
        coll = ahead = inner = 0
        lead = []
        v1 = []
        for k in range(40):
            w = NeutralWorld(*build_paths())
            obs = w.reset(np.random.default_rng(12000 + k))
            prev = 0 if w.delta(0) > 0 else 1
            sw = 0
            c1in = False
            while True:
                a0 = np.zeros(2)
                a1, _ = ck.predict(mp(obs[1], w.L), deterministic=True)
                obs, _, done = w.step(np.stack([a0, a1]))
                if w.lane[1] == 1:
                    c1in = True
                now = 0 if w.delta(0) > 0 else 1
                if now != prev:
                    sw += 1
                    prev = now
                if done:
                    break
            if w.reason == "collision":
                coll += 1
            if w.delta(0) < -0.05:
                ahead += 1
            if c1in:
                inner += 1
            lead.append(sw)
            v1.append(np.mean(w.cars[1].v))
        print(f"  [{label:20s}] coll {coll}/40  car1 ahead@end {ahead}/40  "
              f"car1 inner {inner}/40  lead-chg {np.sum(lead)}  v {np.mean(v1):.3f}")


if __name__ == "__main__":
    T6()
    T7()
