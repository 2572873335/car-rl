"""Positive pole v3: bound the CLOSING RATE, not the command.

v2 failed because it capped the commanded closing rate while the ACHIEVED
closing rate lagged behind -- once the follower has built up speed during the
initial catch-up, lowering the command does not shed that speed instantly
(deceleration is bounded by a_max).

The key realisation from the forced-win trace: the error variable's derivative
already carries the closing rate. With e = gap - D_DES,

    de/dt = d(gap)/dt = v_leader - v_follower = -c

so obs[1] = de/2.0 gives c = -2*obs[1] directly -- no extra state needed.

Correct policy:
  1. compute the arrestable envelope at the CURRENT gap
        c_max = sqrt(2 * a_max_lo * (gap - COLLISION_GAP - margin))
  2. if the ACHIEVED closing rate already exceeds it, brake at full authority
     (shed it now -- this is the branch v2 was missing)
  3. otherwise close at the P-commanded rate, but never beyond c_max
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

ACT = fe.ACT_GAIN


class RateLimitedFollower3:
    def __init__(self, kp=0.8, a_max_lo=1.0, margin=0.05, safety=0.7,
                 yield_g=0.25, yield_max=0.10, recover=0.999):
        self.kp = kp
        self.a_max_lo = a_max_lo
        self.margin = margin
        self.safety = safety      # fraction of the theoretical envelope used
        self.yield_g = yield_g
        self.yield_max = yield_max
        self.recover = recover
        self.extra = 0.0
        self.prev_v = None

    def reset(self):
        self.extra = 0.0
        self.prev_v = None

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        v_self = float(obs[2]) * 1.3
        c = -de                                     # closing rate (>0 = closing)

        # pre-emptive yield on leader deceleration (uses the follower's own
        # speed change as a proxy -- the true leader speed is not in obs)
        if self.prev_v is not None:
            dv = self.prev_v - v_self
            if dv > 0:
                self.extra = min(self.extra + self.yield_g * dv, self.yield_max)
        self.extra *= self.recover
        self.prev_v = v_self

        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - self.margin, 0.0)
        c_max = self.safety * np.sqrt(2.0 * self.a_max_lo * room)

        if c > c_max:
            # already closing too fast -- shed it at full authority
            return np.array([-1.0], dtype=np.float32)

        a_p = self.kp * (e - self.extra) / ACT
        a = float(np.clip(a_p, -1.0, min(1.0, c_max / ACT)))
        return np.array([a], dtype=np.float32)


if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "search", f"{REPO}/results/20260925_phaseF_positive/positive_pole_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = mod.run

    probes = {
        "step-down 0.1s": lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05,
        "step-down 0.2s": lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05,
        "step-down 0.4s": lambda t: 0.95 if int(t / 0.4) % 2 == 0 else 0.05,
        "osc 0.5Hz": lambda t: 0.5 + 0.45 * np.sin(np.pi * t),
        "osc 0.3Hz": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 0.3 * t),
        "chirp": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * (0.1 + 0.25 * min(t, 4.0)) * t),
        "nominal": lambda t: 0.5,
    }

    print("=" * 88)
    print("RATE-LIMITED FOLLOWER v3 -- positive-pole candidate")
    print("=" * 88)
    for d in [0.20, 0.50]:
        pol = RateLimitedFollower3()
        print(f"\n  d_des = {d}")
        tot, errs = 0, []
        for aname, fn in probes.items():
            c, e = run(pol, fn, d, n_ep=20)
            tot += c
            errs.append(e)
            print(f"    {aname:>16s}  coll {c:>2d}/20   settled err {e:>6.2f}")
        print(f"    {'TOTAL':>16s}  coll {tot:>2d}/140   "
              f"mean err {np.mean(errs):.2f}  max err {np.max(errs):.2f}")
