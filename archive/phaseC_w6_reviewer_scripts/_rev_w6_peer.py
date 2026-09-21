"""Reviewer final: peer test with the CORRECTED adapter.

Q: is the checkpoint genuinely a "bad peer" (justifying 'produce from scratch'),
   or was the 'not racing' verdict an adapter artifact?
   Test checkpoint-vs-checkpoint (true peers) with corrected gap.

Also: reproduce the plan's §1.2 deadlock number with BOTH adapters.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from stable_baselines3 import PPO
from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld
from _ckpt_as_opponent import frozen_layout

CKPT = "/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip"


def corrected(o, L):
    delta_n, v_self, v_other, e_lat_n, lane, clear_n = o
    delta = delta_n * 2.5
    gap = (-delta) % L
    return np.array([np.clip((gap - 0.2) / 0.5, -1.0, 6.0),
                     delta / 2.5, v_self, e_lat_n, lane,
                     v_other / 0.5 * 1.3], np.float32)


def peer(label, mp, n_ep=40, seed0=12000):
    ck = PPO.load(CKPT)
    coll = 0
    lead = 0
    inner0 = inner1 = 0
    for k in range(n_ep):
        w = NeutralWorld(*build_paths())
        obs = w.reset(np.random.default_rng(seed0 + k))
        prev = 0 if w.delta(0) > 0 else 1
        c0 = c1 = False
        while True:
            a0, _ = ck.predict(mp(obs[0], w.L), deterministic=True)
            a1, _ = ck.predict(mp(obs[1], w.L), deterministic=True)
            obs, _, done = w.step(np.stack([a0, a1]))
            if w.lane[0] == 1:
                c0 = True
            if w.lane[1] == 1:
                c1 = True
            now = 0 if w.delta(0) > 0 else 1
            if now != prev:
                lead += 1
                prev = now
            if done:
                break
        if w.reason == "collision":
            coll += 1
        if c0:
            inner0 += 1
        if c1:
            inner1 += 1
    print(f"  [{label:22s}] coll {coll}/{n_ep}  lead-changes {lead}  "
          f"inner car0 {inner0}/{n_ep}  car1 {inner1}/{n_ep}")


if __name__ == "__main__":
    print("=" * 78)
    print("PEER TEST: checkpoint vs checkpoint")
    print("=" * 78)
    peer("plan adapter", frozen_layout)
    peer("corrected adapter", corrected)
