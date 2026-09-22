"""D0 reconciliation probe: does the 1.0 m/s "reversal" survive the repo's
conventional settling convention?

WHY THIS EXISTS
---------------
Two unit/aggregation conventions are in play and nobody declared them:

  * repo convention (train_ppo.py:90, robustness_sweep.py:75) --
    mean|e| computed over the SETTLED window only: e[int(len(e)*0.2):]
    (the first 20% of the episode is a catch-up transient, discarded);
  * D0 scripts (_verify_r4_grid.py, _d0_step0_wrapper_probe.py) --
    mean|e| over the WHOLE episode.

If the D0 scripts fold the transient into the mean, then every D0 number
(including the headline "RL 5.63 vs P+FF 5.09 reversal") is inflated by an
amount that has nothing to do with controller quality, and the D0 numbers are
NOT comparable to the published 1.00 cm / 1.93 cm figures.

This probe reports both, on the same episodes, with the same policy objects.
It changes NO frozen file: D_DES is overridden at runtime through the wrapper
path validated in _d0_step0_wrapper_probe.py, then restored.
"""
import hashlib
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe


class DDESWrapper:
    """Runtime D_DES override (frozen file untouched)."""

    def __init__(self, d_des, v_set, behavior):
        self.d_des, self.v_set, self.behavior = d_des, v_set, behavior

    def make(self):
        self.env = fe.FollowEnv(domain_randomize=False)
        return self.env

    def reset(self, seed):
        fe.D_DES = self.d_des
        obs, info = self.env.reset(seed=seed)
        self.env.v_set = self.v_set
        self.env.behavior = self.behavior
        return obs, info

    def step(self, a):
        return self.env.step(a)

    def close(self):
        fe.D_DES = 0.20


def run(policy, d_des, v_set, behavior, n_ep=30, seed0=2000):
    """Return (whole_episode_true_cm, settled_true_cm, coll_rate, reasons)."""
    w = DDESWrapper(d_des, v_set, behavior)
    w.make()
    whole, settled, coll = [], [], 0
    reasons = {}
    for k in range(n_ep):
        obs, _ = w.reset(seed0 + k)
        while True:
            obs, r, term, trunc, _ = w.step(policy(w.env, obs))
            if term or trunc:
                break
        e = np.abs(np.array(w.env.log["gap"]) - d_des)
        whole.append(e.mean())
        settled.append(e[int(len(e) * 0.2):].mean())
        reasons[w.env.term_reason] = reasons.get(w.env.term_reason, 0) + 1
        if w.env.term_reason == "collision":
            coll += 1
    w.close()
    return (float(np.mean(whole)) * 100, float(np.mean(settled)) * 100,
            coll / n_ep, reasons)


def main():
    from stable_baselines3 import PPO
    model = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")

    rl = lambda env, obs: model.predict(obs, deterministic=True)[0]

    def pff(env, obs):
        return fe.baseline_action(obs, use_ff=True)

    h0 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]
    print("=" * 100)
    print("D0 CONVENTION RECONCILIATION  (true-cm = 100*|gap-d_des|)")
    print(f"  follow_env.py sha256[:16] before: {h0}")
    print("  whole = mean over full episode   settled = mean over e[20%:] (repo convention)")
    print("=" * 100)
    print(f"  {'grid':>26s} {'RLwhole':>8s} {'RLsett':>8s} {'PFFwhole':>9s} "
          f"{'PFFsett':>8s} {'winner(sett)':>13s} {'coll':>8s}")

    grids = [
        (0.20, 0.30, "constant"),   # as-trained
        (0.20, 0.50, "constant"),
        (0.50, 0.55, "constant"),
        (0.50, 1.00, "constant"),   # AGV-relevant, the claimed reversal
        (0.50, 1.00, "sinusoid"),
        (0.50, 1.00, "brake"),
        (1.00, 1.00, "constant"),
    ]
    for d, v, b in grids:
        rw, rs, rc, rr = run(rl, d, v, b)
        pw, ps, pc, pr = run(pff, d, v, b)
        win = "RL" if rs < ps else "P+FF"
        print(f"  {f'd={d} v={v} {b[:6]}':>26s} {rw:>8.2f} {rs:>8.2f} "
              f"{pw:>9.2f} {ps:>8.2f} {win:>13s} {rc:.0%}/{pc:.0%}")
        print(f"  {'':>26s} reasons RL={rr} PFF={pr}")

    h1 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]
    print()
    print(f"  follow_env.py sha256[:16] after:  {h1}   untouched={h0 == h1}")


if __name__ == "__main__":
    main()
