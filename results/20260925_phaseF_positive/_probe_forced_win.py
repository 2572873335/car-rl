"""Is the step-down attack a FORCED WIN, or does it just beat MY controllers?

All four hand-coded candidates crashed 14-20/20 on step-down. Before concluding
anything, test the physics.

Both cars have the SAME a_max (nominal 1.5). The follower's prior is
v_cmd = clip(v_l + 0.8*a, 0, 1.3), and v_l is the leader's COMMANDED speed, which
drops instantaneously to 0.05. So with a=0 the follower is commanded to 0.05 too,
and both slew down at the same rate -> the gap should be preserved.

If zero action survives, the crashes are NOT a braking-authority limit; they are
the controllers over-committing during the catch-up transient (initial gap is
random in [0.20, gap_max], so the follower often starts far behind and is pushing
forward when the step lands).

Decisive tests:
  T1 zero action, default gap_max
  T2 P+FF with gap_max = 0.20 (no catch-up at all)
  T3 zero action with gap_max = 0.20
  T4 trace one episode: gap, v_l, v_f over time under step-down
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe


class SL(fe.FollowEnv):
    def set_fn(self, fn):
        self._fn = fn

    def _leader_v(self):
        return float(np.clip(self._fn(self.t), 0.05, 1.0))


def run(policy, fn, d_des, gap_max=1.2, n_ep=20, seed0=2000, trace=False):
    fe.D_DES = d_des
    env = SL(domain_randomize=False, gap_max=gap_max)
    try:
        coll, gaps0 = 0, []
        for k in range(n_ep):
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            gaps0.append(env.gap0)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            if env.term_reason == "collision":
                coll += 1
                if trace and coll == 1:
                    g = np.array(env.log["gap"])
                    vl = np.array(env.log["v_l"])
                    vf = np.array(env.log["v_f"])
                    idx = np.linspace(0, len(g) - 1, 12).astype(int)
                    print("      t     gap     v_l     v_f")
                    for i in idx:
                        print(f"   {env.log['t'][i]:6.2f} {g[i]:7.3f} "
                              f"{vl[i]:7.3f} {vf[i]:7.3f}")
        return coll, float(np.mean(gaps0))
    finally:
        fe.D_DES = 0.20


if __name__ == "__main__":
    pff = lambda o: fe.baseline_action(o, use_ff=True)
    zero = lambda o: np.zeros(1, dtype=np.float32)
    step = lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05

    print("=" * 80)
    print("FORCED-WIN TEST: step-down 0.1s, setpoint d=0.20")
    print("=" * 80)
    for label, pol, gm in [
        ("T1 zero action, gap_max=1.2", zero, 1.2),
        ("T2 P+FF,        gap_max=1.2", pff, 1.2),
        ("T3 zero action, gap_max=0.20", zero, 0.20),
        ("T4 P+FF,        gap_max=0.20", pff, 0.20),
    ]:
        c, g0 = run(pol, step, 0.20, gap_max=gm, n_ep=20)
        print(f"  {label:32s}  coll {c:>2d}/20   mean gap0 {g0:.3f}")

    print("\n  T5 trace of the first colliding episode (P+FF, gap_max=1.2):")
    run(pff, step, 0.20, gap_max=1.2, n_ep=20, trace=True)
