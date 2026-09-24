"""Positive pole v2: a rate-limited "anticipate and yield" follower.

What the forced-win probe established (results/20260925_phaseF_positive/
_probe_forced_win.py):

  * ZERO ACTION never crashes under step-down -- the V2V prior preserves the gap.
  * P+FF crashes 20/20 because it aggressively closes a large initial gap and
    cannot shed that closing rate before the leader steps down.
  * The trace shows the gap monotonically decreasing while v_f stays above v_l:
    the loss is the CLOSING RATE, not braking authority.

So the attack punishes aggression, not caution. A viable positive pole must bound
its own closing rate to what can still be arrested inside the remaining gap:

    v_close <= sqrt(2 * a_max * max(gap - COLLISION_GAP - margin, 0))

with a_max taken at the pessimistic end (1.0, the DR lower bound).

Design: a P controller on e, but with the COMMANDED speed clamped so the
predicted closing rate stays inside the safe envelope, plus a yield term that
opens the gap when the leader brakes hard. Minimal, interpretable, hand-coded --
which is what an anchor should be.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

ACT = fe.ACT_GAIN


class RateLimitedFollower:
    """P control with a closing-rate envelope and a brake-yield term.

    params:
      kp        proportional gain on gap error (same units as P+FF's 0.8)
      a_max_lo  pessimistic deceleration used for the envelope (DR floor)
      margin    extra standoff kept below the envelope
      yield_g   gain on leader deceleration (opens the gap pre-emptively)
      yield_max cap on the pre-emptive opening (m)
      recover   per-step decay of the pre-emptive opening
    """

    def __init__(self, kp=0.8, a_max_lo=1.0, margin=0.04,
                 yield_g=0.30, yield_max=0.12, recover=0.999):
        self.kp = kp
        self.a_max_lo = a_max_lo
        self.margin = margin
        self.yield_g = yield_g
        self.yield_max = yield_max
        self.recover = recover
        self.extra = 0.0
        self.prev_vl = None

    def reset(self):
        self.extra = 0.0
        self.prev_vl = None

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        v_self = obs[2] * 1.3

        # --- pre-emptive yield: leader decelerating -> open the gap now ---
        v_l_cmd = v_self + ACT * 0.0     # v_self already reflects the prior
        if self.prev_vl is not None:
            dv = self.prev_vl - v_self
            if dv > 0:
                self.extra = min(self.extra + self.yield_g * dv, self.yield_max)
        self.extra *= self.recover
        self.prev_vl = v_self

        # --- desired gap and the resulting P command ---
        e_eff = e - self.extra
        a_cmd = self.kp * e_eff / ACT

        # --- closing-rate envelope -------------------------------------------
        # remaining room before the collision shell, from the CURRENT gap
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - self.margin, 0.0)
        v_close_safe = np.sqrt(2.0 * self.a_max_lo * room)
        # do not command a closing speed beyond what can be arrested in time
        a_cap = v_close_safe / ACT          # as an action value
        a = float(np.clip(a_cmd, -1.0, min(1.0, a_cap)))
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
    print("RATE-LIMITED FOLLOWER -- positive-pole candidate v2")
    print("=" * 88)
    for d in [0.20, 0.50]:
        print(f"\n  d_des = {d}")
        pol = RateLimitedFollower()
        tot = 0
        for aname, fn in probes.items():
            c, e = run(pol, fn, d, n_ep=20)
            tot += c
            print(f"    {aname:>16s}  coll {c:>2d}/20   settled err {e:>6.2f}")
        print(f"    {'TOTAL':>16s}  coll {tot:>2d}/140")
