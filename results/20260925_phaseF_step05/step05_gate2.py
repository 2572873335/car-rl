"""Step 0.5 gate 2 (re-run): completion-rate check on the trained defender.

The training run completed and gate 1 passed (std max 1.433 < 2.0). The gate-2
evaluation then crashed on a typo -- `ev.t_` instead of `ev.t` (the env stores
elapsed time as self.t). The checkpoint was saved before that, so this re-runs
only the evaluation half against it.

Gate 2 asks whether episodes actually finish, because a bounded std can also be
achieved by a policy that collapses into always-terminating states.
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import gymnasium as gym
import follow_env as fe

OUT = f"{REPO}/results/20260925_phaseF_step05"


class RandomWaveLeader(fe.FollowEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        r = self.rng
        kind = r.integers(0, 4)
        if kind == 0:
            period = r.uniform(0.2, 0.8)
            self._wf = lambda t: 0.95 if int(t / period) % 2 == 0 else 0.05
        elif kind == 1:
            f = r.uniform(0.1, 0.8)
            ph = r.uniform(0, 2 * np.pi)
            self._wf = lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * f * t + ph)
        elif kind == 2:
            seg = r.uniform(0.2, 0.8)
            rr = np.random.default_rng(int(r.uniform(0, 1000)))
            vals = rr.uniform(0.05, 1.0, 200)
            self._wf = lambda t: vals[int(t / seg) % 200]
        else:
            self._wf = lambda t: 0.5 + 0.45 * np.sin(
                2 * np.pi * (0.1 + 0.15 * min(t, 4.0)) * t)
        return obs, info

    def _leader_v(self):
        fn = getattr(self, "_wf", None)
        if fn is None:
            return super()._leader_v()
        return float(np.clip(fn(self.t), 0.05, 1.0))


if __name__ == "__main__":
    from stable_baselines3 import PPO
    model = PPO.load(f"{OUT}/step05_defender.zip")
    ev = RandomWaveLeader(domain_randomize=False)

    N = 40
    early, coll, lens = 0, 0, []
    for k in range(N):
        obs, _ = ev.reset(seed=60_000 + k)
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = ev.step(a)
            if term or trunc:
                break
        lens.append(ev.t)
        if ev.t < ev.t_max - 1e-6:
            early += 1
        if ev.term_reason == "collision":
            coll += 1

    rate = early / N
    print("=" * 74)
    print("STEP 0.5 -- gate 2 (completion) on the trained defender")
    print("=" * 74)
    print(f"  episodes                : {N}")
    print(f"  mean episode length [s] : {np.mean(lens):.2f}  (t_max={ev.t_max})")
    print(f"  early terminations      : {rate:.1%} ({early}/{N})")
    print(f"  collisions              : {coll}/{N}")
    g2 = rate < 0.5
    print(f"  gate 2 (early-term < 50%): {'PASS' if g2 else 'FAIL'}")
    print()
    print("  NB: gate 1 already passed during training (std max 1.433 < 2.0),")
    print(f"  so STEP 0.5 = {'PASS' if g2 else 'FAIL'}")
