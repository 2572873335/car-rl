"""Round 13 probe F -- THE SHARPEST REMAINING NO-LOGIC FAMILY.

The zoo showed every "react to the disturbance" -free policy fails, because any
policy that CATCHES UP (gap0 ~ U(0.2, 1.2) forces an approach) builds a closing
rate it cannot shed when the leader steps down.  The one family the zoo did not
combine is:

      "target a LARGER gap"  x  "approach it GENTLY"  x  "freeze on arrival"

i.e. never approach fast (so nothing has to be shed), never sit at d_des (so the
absolute gap is bigger when the step arrives), and stop acting once settled (so
the V2V prior holds whatever gap was reached).  None of these three ingredients
is a reaction to a disturbance: the policy has no branch that reads "the attack
started".

If such a combination passes (b)(c)(e') on B*, the conjunction still has a
channel even with B6 in the domain -- that is exactly the claim under test.

Also swept: asymmetric gains and a slew-rate limit on the action.
"""
import sys
import numpy as np

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/results/20260925_phaseF_positive")
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import follow_env as fe
import _review3_core as C

ACT = fe.ACT_GAIN


class GapCap:
    """Aim at d_des + off, with the POSITIVE (closing) action capped at apos.
    No disturbance branch; no gap-rate term at all in the decision."""

    def __init__(self, off=0.30, apos=0.15, kp=0.8):
        self.off, self.apos, self.kp = off, apos, kp

    def reset(self):
        pass

    def __call__(self, obs):
        e = obs[0] * 0.5 - self.off
        return np.array([float(np.clip(self.kp * e / ACT, -1.0,
                                       min(1.0, self.apos)))], dtype=np.float32)


class GapCapFreeze(GapCap):
    """GapCap that latches a=0 once |e - off| is small (the V2V prior then holds)."""

    def __init__(self, off=0.30, apos=0.15, kp=0.8, eps=0.03):
        super().__init__(off, apos, kp)
        self.eps = eps

    def reset(self):
        self.frozen = False
        self.n = self.nf = 0

    def __call__(self, obs):
        e = obs[0] * 0.5 - self.off
        self.n += 1
        if abs(e) < self.eps:
            self.frozen = True
        if self.frozen:
            self.nf += 1
            return np.zeros(1, dtype=np.float32)
        return np.array([float(np.clip(self.kp * e / ACT, -1.0,
                                       min(1.0, self.apos)))], dtype=np.float32)


class AsymGain:
    """Asymmetric P+FF: gentle when closing (a>0), full authority when opening."""

    def __init__(self, apos=0.15, kp=0.8):
        self.apos, self.kp = apos, kp

    def reset(self):
        pass

    def __call__(self, obs):
        return np.array([float(np.clip(self.kp * (obs[0] * 0.5) / ACT, -1.0,
                                       min(1.0, self.apos)))], dtype=np.float32)


class SlewLimit:
    """P+FF with a slew-rate limit on the action: |a_t - a_{t-1}| <= da."""

    def __init__(self, da=0.05, kp=0.8):
        self.da, self.kp = da, kp

    def reset(self):
        self.a = 0.0

    def __call__(self, obs):
        t = float(np.clip(self.kp * (obs[0] * 0.5) / ACT, -1.0, 1.0))
        self.a += float(np.clip(t - self.a, -self.da, self.da))
        return np.array([self.a], dtype=np.float32)


ZOO_F = []
for off in (0.20, 0.30, 0.40, 0.50):
    for apos in (0.05, 0.10, 0.20, 0.30):
        ZOO_F.append((f"gapCap off{off:.2f} a+{apos:.2f}",
                      lambda off=off, apos=apos: GapCap(off, apos), "NONE"))
for off in (0.20, 0.30, 0.40, 0.50):
    for apos in (0.10, 0.20, 0.30):
        ZOO_F.append((f"gapCapFr off{off:.2f} a+{apos:.2f}",
                      lambda off=off, apos=apos: GapCapFreeze(off, apos), "NONE"))
ZOO_F += [
    ("asym a+<=0.05", lambda: AsymGain(0.05), "NONE"),
    ("asym a+<=0.10", lambda: AsymGain(0.10), "NONE"),
    ("asym a+<=0.20", lambda: AsymGain(0.20), "NONE"),
    ("slew da=0.02", lambda: SlewLimit(0.02), "NONE"),
    ("slew da=0.05", lambda: SlewLimit(0.05), "NONE"),
    ("slew da=0.10", lambda: SlewLimit(0.10), "NONE"),
]

print("=" * 118)
print("F  DISTANCE x GENTLE APPROACH x FREEZE -- no disturbance branch anywhere")
print("=" * 118)

pole_c, pole_e, _, _ = C.evaluate(lambda: C.Pole(), n_ep=20)
v1_c, v1_e, _, _ = C.evaluate(lambda: C.v1_loader(), n_ep=20) if hasattr(C, "v1_loader") else (None, None, None, None)
pole_all = C.pooled(pole_e, C.ITEMS)
pole15 = C.pooled(pole_e, C.B15)
gate_b = 0.10 * 160
gate_e = 1.5 * pole_all
v1_tot = 42
gate_c = 0.5 * v1_tot
print(f"  gate (b) <= {gate_b:.0f}/160   (c) <= {gate_c:.0f}/160   "
      f"(e') <= {gate_e:.2f}   [pole B* pooled {pole_all:.2f}, B15 {pole15:.2f}]")
print(f"\n  {'policy':>24s} {'crash':>9s} {'B15':>7s} {'B6':>6s} {'errB15':>7s} "
      f"{'errB6':>7s} {'errB*':>7s} {'nom':>6s}  (b)(c)(e')  VERDICT")
best = []
for nm, f, lg in ZOO_F:
    crash, err, mg, tr = C.evaluate(f, n_ep=20)
    tot = sum(crash.values())
    t15 = sum(crash[k] for k in C.B15)
    t6 = sum(crash[k] for k in C.B6)
    e15 = C.pooled(err, C.B15)
    e6 = C.pooled(err, C.B6)
    eall = C.pooled(err, C.ITEMS)
    ncc, ne = C.evaluate_nominal(f, n_ep=20)
    b = tot <= gate_b
    c = tot <= gate_c
    e = eall <= gate_e
    ok = b and c and e
    if ok:
        best.append(nm)
    print(f"  {nm:>24s} {tot:>5d}/160 {t15:>4d}/100 {t6:>3d}/60 {e15:>7.2f} "
          f"{e6:>7.2f} {eall:>7.2f} {ne:>6.2f}  {'Y' if b else 'n'} {'Y' if c else 'n'} "
          f"{'Y' if e else 'n':>3s}   {'PASS' if ok else 'fail'}")
    sys.stdout.flush()

print(f"\n  PASSERS: {best if best else 'NONE'}")
print("\nDONE")
