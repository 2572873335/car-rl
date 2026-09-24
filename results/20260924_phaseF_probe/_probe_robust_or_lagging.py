"""Is the learned policy genuinely robust to bang-bang, or just failing safe?

The search found: bang-bang 0.2s crashes P+FF 20/20 at d=0.20, but the v1 RL
policy 0/20. Before calling that "RL is robust", rule out the degenerate
explanation: a policy that simply backs off and lags far behind also never
crashes. That is failing safe, not tracking well -- and per the project's
degenerate-case discipline (DATA_MANAGEMENT 10.4/10.5) that distinction decides
whether this is a finding or an artifact.

So: log the settled gap error alongside the crash count. Robust = low crash AND
low tracking error. Failing safe = low crash AND large tracking error.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe


class ScriptedLeader(fe.FollowEnv):
    def set_fn(self, fn, base=None):
        self._fn = fn
        self._base = base

    def _leader_v(self):
        return float(np.clip(self._fn(self.t), 0.05, 1.0))


def run(policy, fn, d_des, n_ep=20, seed0=2000):
    fe.D_DES = d_des
    env = ScriptedLeader(domain_randomize=False)
    try:
        coll, errs, gaps = 0, [], []
        for k in range(n_ep):
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            g = np.array(env.log["gap"])
            settled = g[int(len(g) * 0.2):]
            errs.append(np.abs(settled - d_des).mean() * 200)   # obs-cm
            gaps.append(settled.mean())
            if env.term_reason == "collision":
                coll += 1
        return coll, float(np.mean(errs)), float(np.mean(gaps))
    finally:
        fe.D_DES = 0.20


if __name__ == "__main__":
    from stable_baselines3 import PPO
    m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
    pff = lambda o: fe.baseline_action(o, use_ff=True)
    rl = lambda o: m.predict(obs_dummy(o), deterministic=True)[0] if False else m.predict(o, deterministic=True)[0]

    bb = lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
    osc = lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 0.5 * t)
    nom = lambda t: 0.5

    print("=" * 92)
    print("ROBUSTNESS vs FAILING-SAFE  (crash count AND settled tracking error)")
    print("=" * 92)
    print(f"  {'attack':>12s} {'d_des':>6s} {'controller':>12s} "
          f"{'coll/20':>8s} {'settled err (obs-cm)':>21s} {'mean gap (m)':>13s}")
    for name, fn in [("bang-bang", bb), ("osc f=0.5", osc), ("nominal", nom)]:
        for d in [0.20, 0.50]:
            for cname, pol in [("P+FF", pff), ("v1 RL", rl)]:
                c, e, g = run(pol, fn, d, n_ep=20)
                print(f"  {name:>12s} {d:>6.2f} {cname:>12s} "
                      f"{c:>4d}/20 {e:>20.2f} {g:>13.3f}")
    print()
    print("  READ: low crash + LOW error  = genuinely robust")
    print("        low crash + HIGH error = failing safe (lagging), not robust")
