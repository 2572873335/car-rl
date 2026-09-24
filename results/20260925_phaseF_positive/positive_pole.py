"""Positive pole -- FINAL. Locked configuration and full validation.

Locked (from the D2 grid, results/20260925_phaseF_positive/_out_pole_v4.txt):

    margin = 0.05, safety = 0.60, kp = 0.8, a_max_lo = 1.0,
    ythresh = 0.20, yg = 0.25, ymax = 0.06, recover = 0.998

Result at d=0.20: 0/140 collisions, nominal settled error 2.24 obs-cm
(better than the P+FF reference's own 2.54).

Why it works, stated precisely -- the earlier forced-win probe settled this:
  ZERO ACTION never crashes under any of these attacks (the V2V prior tracks the
  leader's commanded speed, so the gap is preserved). The crashes come from
  AGGRESSIVE GAP CLOSING: the follower builds a closing rate to catch up from the
  random initial gap, and cannot shed it when the leader steps down. The binding
  constraint is therefore the CLOSING RATE, not braking authority:

      c_max = safety * sqrt(2 * a_max_lo * (gap - COLLISION_GAP - margin))

  The follower brakes at full authority whenever the achieved closing rate
  exceeds c_max, and otherwise behaves as a plain P controller. Because the
  envelope only binds during aggression, nominal precision is untouched.

Acceptance (owner Q6):
  A. zero collisions vs both named probes (step-down 0.1s, osc 0.5Hz)
  B. zero collisions across the whole probe family, both spacings
  C. nominal tracking within 1.5x the P+FF baseline (H-A4) and not fakeable by
     "crawl far away"
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

ACT = fe.ACT_GAIN

KP = 0.8
A_MAX_LO = 1.0
MARGIN = 0.05
SAFETY = 0.60
YTHRESH = 0.20
YG = 0.25
YMAX = 0.06
RECOVER = 0.998


class PositivePole:
    """Hand-coded follower -- the Phase F threshold anchor.

    MECHANISM, corrected per review1 R2 (an earlier description claimed
    "anticipate and yield"; the anticipate half is DEAD CODE):

        a = -1                    if the ACHIEVED closing rate c = -d(gap)/dt
                                   exceeds the gap-dependent envelope c_max
        a = clip(kp*e/ACT, -1, 1) otherwise

    i.e. plain P control, plus one full-authority brake rule that fires on
    2.4-8.1% of steps. Ablation (review1): clamping the commanded closing rate
    ALONE crashes 60/220; the full-authority brake ALONE passes 0/220. What
    carries the behaviour is braking on the ACHIEVED rate, not anticipation.

    DEAD PARAMETERS: YTHRESH, YG, YMAX, RECOVER never bind -- YTHRESH=0.20
    against a per-step speed-change ceiling of a_max*DT <= 0.04 (unreachable;
    measured 0 triggers across 11 families x 3 spacings). The command clamp
    (c_max/ACT) also never binds (0/1001 nominal steps). Kept for lineage; do
    not cite them as mechanism.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.extra = 0.0
        self.prev_v = None

    def __call__(self, obs):
        e = float(obs[0]) * 0.5          # gap error [m], e = gap - D_DES
        de = float(obs[1]) * 2.0         # gap rate [m/s]
        v = float(obs[2]) * 1.3          # own speed [m/s]
        c = -de                          # closing rate (>0 = losing gap)

        # pre-emptive yield on abrupt deceleration (this is the "anticipate")
        if self.prev_v is not None:
            dv = self.prev_v - v
            if dv > YTHRESH:
                self.extra = min(self.extra + YG * dv, YMAX)
        self.extra *= RECOVER
        self.prev_v = v

        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - MARGIN, 0.0)
        c_max = SAFETY * np.sqrt(2.0 * A_MAX_LO * room)

        if c > c_max:                    # yield hard: shed the closing rate now
            return np.array([-1.0], dtype=np.float32)

        a = KP * (e - self.extra) / ACT
        return np.array([float(np.clip(a, -1.0, min(1.0, c_max / ACT)))],
                        dtype=np.float32)


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
        "step-down 0.8s": lambda t: 0.95 if int(t / 0.8) % 2 == 0 else 0.05,
        "osc 0.5Hz": lambda t: 0.5 + 0.45 * np.sin(np.pi * t),
        "osc 0.3Hz": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 0.3 * t),
        "osc 0.8Hz": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 0.8 * t),
        "chirp": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * (0.1 + 0.25 * min(t, 4.0)) * t),
        "pw-const": lambda t: [0.3, 0.9, 0.15, 0.7, 0.4][int(t / 0.4) % 5],
        "nominal": lambda t: 0.5,
        "crawl": lambda t: 0.10,
    }

    print("=" * 92)
    print("POSITIVE POLE -- FINAL VALIDATION")
    print(f"  locked: margin={MARGIN} safety={SAFETY} kp={KP} yth={YTHRESH} "
          f"yg={YG} ymax={YMAX}")
    print("=" * 92)

    all_ok = True
    ref = {}
    for d in [0.20, 0.50, 1.00]:
        pol = PositivePole()
        print(f"\n  d_des = {d}")
        tot = 0
        for an, fn in probes.items():
            c, e = run(pol, fn, d, n_ep=20)
            tot += c
            if an == "nominal":
                ref[d] = e
            print(f"    {an:>16s}  coll {c:>2d}/20   settled err {e:>6.2f}")
        print(f"    {'TOTAL':>16s}  coll {tot:>2d}/{20*len(probes)}")
        if tot:
            all_ok = False

    # acceptance C: compare against P+FF on the nominal case
    print()
    print("=" * 92)
    print("ACCEPTANCE")
    print("=" * 92)
    pff_nom = {}
    for d in [0.20, 0.50, 1.00]:
        c, e = run(lambda o: fe.baseline_action(o, use_ff=True),
                   probes["nominal"], d, n_ep=20)
        pff_nom[d] = e
        ratio = ref[d] / e
        ok = ratio <= 1.5
        print(f"  d={d:.2f}  nominal err: pole {ref[d]:.2f} vs P+FF {e:.2f} "
              f"-> ratio {ratio:.2f}x  {'OK' if ok else 'FAIL'} (<=1.5x)")
        all_ok = all_ok and ok
    print()
    print(f"  ALL ACCEPTANCE: {'PASS' if all_ok else 'FAIL'}")
