"""Round 13 probe C -- three supporting measurements for review3.

C1  THE RESET TRAP (plan sec 2.3.1(iv)): the pole is STATEFUL (prev_v).  Show that
    the same policy, same attacks, same harness differ wildly if policy.reset() is
    NOT called at the episode boundary.  This is the trap the owner flagged; it is
    demonstrated here so that the zoo numbers (which DO reset) can be trusted.

C2  ACTUATOR-LIMITED CELLS at d=0.20 (plan sec 2.3.1(i) generalised).  The leader's
    own actuator (a_max*DT = 0.03 m/s/step) cannot follow a 0.1 s square wave, so
    the DELIVERED attack is much weaker than the commanded one.  If B6-0.1 is such
    a cell, its crash column and its error column both measure the LEADER, not the
    defender -- and B6's discriminating power may rest on 0.2/0.4 s only.

C3  EXTRA NO-LOGIC SHAPES not covered by the zoo: always-brake, always-throttle,
    freeze-on-rate-only, and a P-cap with a hard negative cap.
"""
import sys
import importlib.util

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/results/20260925_phaseF_positive")
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
import _review3_core as C

pp = C.pp
ACT = fe.ACT_GAIN
CLIP = C.CLIP


# ---------------------------------------------------------------- C1: reset ---
def run_b_nors(policy, items=("B1", "B2", "B6-0.2"), n_ep=20, env0=60000, atk0=50000, d=0.20):
    """Same as C.evaluate but the policy is built ONCE and reset() is NEVER called."""
    fe.D_DES = d
    crash, err = {}, {}
    try:
        pol = policy()                      # ONE instance for the whole run
        for kind in items:
            env = C.ScriptedLeader(domain_randomize=False, gap_max=1.2)
            c, es = 0, []
            for k in range(n_ep):
                obs, _ = env.reset(seed=env0 + k)
                env.set_fn(C.attack_fn(kind, atk0 + k))
                while True:
                    obs, r, term, trunc, _ = env.step(pol(obs))
                    if term or trunc:
                        break
                g = np.array(env.log["gap"])
                s = g[int(len(g) * 0.2):]
                es.append(float(np.abs(s - d).mean() * 200))
                c += int(env.term_reason == "collision")
            crash[kind], err[kind] = c, float(np.mean(es))
    finally:
        fe.D_DES = 0.20
    return crash, err


print("=" * 100)
print("C1  THE RESET TRAP -- pole with per-episode reset() vs without")
print("=" * 100)
print("  (positive pole and zero-action both measured under the two harnesses)")
print("\n  policy            harness              B1     B2   B6-0.2   errB1   errB2  errB6-0.2")
c1, e1, _, _ = C.evaluate(lambda: C.Pole(), items=("B1", "B2", "B6-0.2"), n_ep=20)
print(f"  {'pole':>16s}  {'reset every ep':>18s} " +
      "".join(f"{c1[k]:>4d}/20" for k in ("B1", "B2", "B6-0.2")) +
      "".join(f"{e1[k]:>8.2f}" for k in ("B1", "B2", "B6-0.2")))
c2, e2 = run_b_nors(lambda: C.Pole(), items=("B1", "B2", "B6-0.2"), n_ep=20)
print(f"  {'pole':>16s}  {'NO reset':>18s} " +
      "".join(f"{c2[k]:>4d}/20" for k in ("B1", "B2", "B6-0.2")) +
      "".join(f"{e2[k]:>8.2f}" for k in ("B1", "B2", "B6-0.2")))
c3b, e3b, _, _ = C.evaluate(lambda: C.PFF(0.8), items=("B1", "B2", "B6-0.2"), n_ep=20)
print(f"  {'P+FF':>16s}  {'reset every ep':>18s} " +
      "".join(f"{c3b[k]:>4d}/20" for k in ("B1", "B2", "B6-0.2")) +
      "".join(f"{e3b[k]:>8.2f}" for k in ("B1", "B2", "B6-0.2")))
c4b, e4b = run_b_nors(lambda: C.PFF(0.8), items=("B1", "B2", "B6-0.2"), n_ep=20)
print(f"  {'P+FF':>16s}  {'NO reset':>18s} " +
      "".join(f"{c4b[k]:>4d}/20" for k in ("B1", "B2", "B6-0.2")) +
      "".join(f"{e4b[k]:>8.2f}" for k in ("B1", "B2", "B6-0.2")))

# ------------------------------------------------- C2: delivered attack strength ---
print("\n" + "=" * 100)
print("C2  DELIVERED vs COMMANDED attack strength at d=0.20 (plan 2.3.1(i) generalised)")
print("=" * 100)
print(f"  {'item':>8s} {'cmd mean':>9s} {'cmd pp':>8s} {'ACTUAL mean':>12s} {'ACTUAL pp':>10s}")
for kind in C.ITEMS:
    fe.D_DES = 0.20
    env = C.ScriptedLeader(domain_randomize=False, gap_max=1.2)
    cmd, act, clip0 = [], [], 0
    for k in range(20):
        pol = C.Pole()
        pol.reset()
        obs, _ = env.reset(seed=60000 + k)
        fn = C.attack_fn(kind, 50000 + k)
        env.set_fn(fn)
        while True:
            obs, r, term, trunc, _ = env.step(pol(obs))
            if term or trunc:
                break
        cmd += [fn(t) for t in env.log["t"]]
        act += list(env.log["v_l"])
    cmd, act = np.array(cmd), np.array(act)
    print(f"  {kind:>8s} {cmd.mean():>9.3f} {cmd.max()-cmd.min():>8.3f} "
          f"{act.mean():>12.3f} {act.max()-act.min():>10.3f}")
fe.D_DES = 0.20

# ---------------------------------------------------------- C3: extra shapes ---
class AlwaysBrake:
    def reset(self):
        pass

    def __call__(self, obs):
        return np.array([-1.0], dtype=np.float32)


class AlwaysThrottle:
    def reset(self):
        pass

    def __call__(self, obs):
        return np.array([1.0], dtype=np.float32)


class FreezeOnRate:
    """No gap term at all: a == 0 unless the closing rate is large, then brake hard.
    Contains a reaction, but ONLY to the rate -- no tracking at all."""

    def __init__(self, c_th=0.20):
        self.c_th = c_th

    def reset(self):
        pass

    def __call__(self, obs):
        de = obs[1] * 2.0
        if -de > self.c_th:
            return np.array([-1.0], dtype=np.float32)
        return np.zeros(1, dtype=np.float32)


class PCapSigned:
    """P+FF with BOTH signs capped: closing AND opening rate bounded."""

    def __init__(self, amax=0.3):
        self.amax = amax

    def reset(self):
        pass

    def __call__(self, obs):
        return np.array([float(np.clip(0.8 * (obs[0] * 0.5) / ACT, -self.amax,
                                       self.amax))], dtype=np.float32)


class GainSweep:
    """P+FF with a large gain (stiff tracking, no disturbance logic)."""

    def __init__(self, kp):
        self.kp = kp

    def reset(self):
        pass

    def __call__(self, obs):
        return CLIP(self.kp * (obs[0] * 0.5) / ACT)


print("\n" + "=" * 100)
print("C3  EXTRA NO-LOGIC / LOW-LOGIC SHAPES on B*")
print("=" * 100)
pole_c, pole_e, _, _ = C.evaluate(lambda: C.Pole(), n_ep=20)
gate_b = 0.10 * 160
gate_c = 0.5 * 42
gate_e = 1.5 * C.pooled(pole_e, C.ITEMS)
print(f"  gates: (b)<=16/160  (c)<=21/160  (e')<={gate_e:.2f}")
print(f"  {'policy':>30s} {'crash':>8s} {'errB15':>7s} {'errB6':>7s} {'errB*':>7s} "
      f"{'nom':>6s}  (b)(c)(e')  VERDICT")
for nm, f in (("always brake a=-1", lambda: AlwaysBrake()),
              ("always throttle a=+1", lambda: AlwaysThrottle()),
              ("freezeOnRate c>.10", lambda: FreezeOnRate(0.10)),
              ("freezeOnRate c>.20", lambda: FreezeOnRate(0.20)),
              ("freezeOnRate c>.35", lambda: FreezeOnRate(0.35)),
              ("PcapSigned +-0.10", lambda: PCapSigned(0.10)),
              ("PcapSigned +-0.30", lambda: PCapSigned(0.30)),
              ("PcapSigned +-0.60", lambda: PCapSigned(0.60)),
              ("P+FF kp=4.0", lambda: GainSweep(4.0)),
              ("P+FF kp=8.0", lambda: GainSweep(8.0))):
    crash, err, mg, tr = C.evaluate(f, n_ep=20)
    tot = sum(crash.values())
    e15 = C.pooled(err, C.B15)
    e6 = C.pooled(err, C.B6)
    eall = C.pooled(err, C.ITEMS)
    ncc, nerr = C.evaluate_nominal(f, n_ep=20)
    b = tot <= gate_b
    c = tot <= gate_c
    e = eall <= gate_e
    print(f"  {nm:>30s} {tot:>5d}/160 {e15:>7.2f} {e6:>7.2f} {eall:>7.2f} {nerr:>6.2f}  "
          f"{'Y' if b else 'n'} {'Y' if c else 'n'} {'Y' if e else 'n':>3s}   "
          f"{'PASS' if (b and c and e) else 'fail'}")

# ------------------------------------------- C4: (f) counterfactual on B1-5 only ---
print("\n" + "=" * 100)
print("C4  COUNTERFACTUAL: (b)(c)(e') graded on B1-B5 ONLY (no B6) -- review2's channel")
print("=" * 100)
v1c, v1e, _, _ = C.evaluate(lambda: np.array([0.0], dtype=np.float32), n_ep=20)
print("  (v1 baseline already measured in the zoo run: crash B15 = "
      "1+12+20+5+0 = 38/100, pole B15 pooled err 3.99 -> gate 5.98)")
print(f"  {'policy':>30s} {'crash15':>8s} {'errB15':>7s}  (b15<=10) (c15<=19) (e15<=5.98)  VERDICT")
for nm, f in (("pole (anchor)", lambda: C.Pole()),
              ("Pcap a+<=0.30", lambda: C.PCap(0.8, 0.30)),
              ("Pcap a+<=0.50", lambda: C.PCap(0.8, 0.50)),
              ("P+FF kp=0.60", lambda: C.PFF(0.60)),
              ("freezeGap g<.35 c>.20", lambda: C.FreezeGap(0.35, 0.20)),
              ("zero action", lambda: C.Zero()),
              ("brake only", lambda: C.BrakeOnly())):
    crash, err, mg, tr = C.evaluate(f, items=C.B15, n_ep=20)
    tot = sum(crash.values())
    e15 = C.pooled(err, C.B15)
    b = tot <= 10
    c = tot <= 19
    e = e15 <= 5.98
    print(f"  {nm:>30s} {tot:>5d}/100 {e15:>7.2f}   {'Y' if b else 'n':>6s} "
          f"{'Y' if c else 'n':>8s} {'Y' if e else 'n':>10s}   "
          f"{'PASS' if (b and c and e) else 'fail'}")

print("\nDONE")
