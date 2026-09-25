"""Independently verify review3's F1: at d=0.50 / 1.00 the frozen untrained
baseline (v1) passes the six-clause conjunction by itself.

That claim, if true, means the criterion does not measure "gain" at those
spacings -- the exact failure mode (10.5c) the last two rounds were about, in a
new dimension.

Measures, per spacing, the B* pooled settled error and crash rate for:
  pole, v1 (untrained baseline), zero action, P+FF
Uses the reviewer's judgement domain: B1-B5 plus B6 (step-down 0.1/0.2/0.4 s),
20 episodes/item, env seeds 60000+k, attack seeds 50000+k.

HARNESS NOTE (plan 2.3.1(iv)): the pole is stateful (prev_v), so reset() is
called before EVERY episode, for every policy, on the same harness.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import importlib.util
import numpy as np
import follow_env as fe

BASE = REPO + "/results/20260925_phaseF_positive"


class ScriptedLeader(fe.FollowEnv):
    def set_fn(self, fn):
        self._fn = fn

    def _leader_v(self):
        return float(np.clip(self._fn(self.t), 0.05, 1.0))


B = {
    "B1": lambda t: 0.95 if int(t / 0.3) % 2 == 0 else 0.05,
    "B2": lambda t: 0.5 + 0.4 * np.sin(2 * np.pi * 0.3 * t),
    "B3": lambda t: 0.5 + 0.4 * np.sin(2 * np.pi * (0.1 + 0.125 * min(t, 4.0)) * t),
    "B4": lambda t: [0.3, 0.9, 0.15, 0.7, 0.4][int(t / 0.4) % 5],
    "B5": lambda t: 0.9 if (t % 0.3) < 0.1 else 0.1,
    "B6-0.1": lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05,
    "B6-0.2": lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05,
    "B6-0.4": lambda t: 0.95 if int(t / 0.4) % 2 == 0 else 0.05,
}


def evaluate(policy, d_des, n_ep=20):
    fe.D_DES = d_des
    try:
        env = ScriptedLeader(domain_randomize=False)
        crash, errs = 0, []
        for name, fn in B.items():
            for k in range(n_ep):
                if hasattr(policy, "reset"):
                    policy.reset()
                obs, _ = env.reset(seed=60000 + k)
                env.set_fn(fn)
                while True:
                    obs, r, term, trunc, _ = env.step(policy(obs))
                    if term or trunc:
                        break
                g = np.array(env.log["gap"])
                s = g[int(len(g) * 0.2):]
                errs.append(np.abs(s - d_des).mean() * 200)
                if env.term_reason == "collision":
                    crash += 1
        return crash, float(np.mean(errs)), 20 * len(B)
    finally:
        fe.D_DES = 0.20


if __name__ == "__main__":
    from stable_baselines3 import PPO
    spec = importlib.util.spec_from_file_location("pole", BASE + "/positive_pole.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)

    v1 = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")

    pols = {
        "pole (anchor)": pm.PositivePole(),
        "v1 (UNTRAINED baseline)": (lambda o: v1.predict(o, deterministic=True)[0]),
        "zero action": (lambda o: np.zeros(1, dtype=np.float32)),
        "P+FF": (lambda o: fe.baseline_action(o, use_ff=True)),
    }

    print("=" * 92)
    print("REVIEW3 F1 VERIFICATION -- does the untrained baseline self-pass?")
    print("=" * 92)
    for d in [0.20, 0.50, 1.00]:
        print(f"\n  d_des = {d}")
        print(f"    {'policy':>24s} {'crash':>9s} {'errB*':>8s}")
        for name, pol in pols.items():
            c, e, n = evaluate(pol, d)
            print(f"    {name:>24s} {c:>5d}/{n} {e:>8.2f}")
