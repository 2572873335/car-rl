"""Tune the positive pole: keep zero collisions, recover nominal precision.

v3 achieves 0/140 on both spacings but tracks badly in nominal (18.06 obs-cm vs
P+FF's 2.54). H-A4 requires the positive pole to stay within 1.5x the baseline,
so that is a real failure, not a cosmetic one.

Diagnosis: two terms are over-conservative.
  * `safety=0.7` and `margin=0.05` shrink the allowed closing rate even when
    there is no attack at all, so the controller never catches up after the
    random initial gap (gap0 ~ U(0.20, gap_max=1.2)).
  * `yield_g/yield_max` accumulate `extra` from ordinary speed variation, holding
    a standing offset.

Fix direction: keep the emergency branch (c > c_max -> full brake, which is what
actually prevents the crashes) but let the normal branch be a plain P controller
with the envelope as a CEILING rather than a target, and make the yield term
trigger only on genuinely abrupt deceleration.

Sweep a small grid and report the collision/error trade-off rather than picking
one point by eye.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

ACT = fe.ACT_GAIN


class Pole:
    def __init__(self, kp=0.8, a_max_lo=1.0, margin=0.02, safety=0.9,
                 ythresh=0.30, yg=0.15, ymax=0.05, recover=0.99):
        self.kp, self.a_max_lo = kp, a_max_lo
        self.margin, self.safety = margin, safety
        self.ythresh, self.yg, self.ymax = ythresh, yg, ymax
        self.recover = recover
        self.extra, self.prev_v = 0.0, None

    def reset(self):
        self.extra, self.prev_v = 0.0, None

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        v = float(obs[2]) * 1.3
        c = -de

        if self.prev_v is not None:
            dv = self.prev_v - v
            if dv > self.ythresh:            # only abrupt deceleration
                self.extra = min(self.extra + self.yg * dv, self.ymax)
        self.extra *= self.recover
        self.prev_v = v

        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - self.margin, 0.0)
        c_max = self.safety * np.sqrt(2.0 * self.a_max_lo * room)

        if c > c_max:                        # emergency: shed closing rate now
            return np.array([-1.0], dtype=np.float32)
        a_p = self.kp * (e - self.extra) / ACT
        return np.array([float(np.clip(a_p, -1.0, min(1.0, c_max / ACT)))],
                        dtype=np.float32)


if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "search", f"{REPO}/results/20260925_phaseF_positive/positive_pole_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = mod.run

    probes = {
        "sd0.1": lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05,
        "sd0.2": lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05,
        "sd0.4": lambda t: 0.95 if int(t / 0.4) % 2 == 0 else 0.05,
        "osc0.5": lambda t: 0.5 + 0.45 * np.sin(np.pi * t),
        "osc0.3": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 0.3 * t),
        "chirp": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * (0.1 + 0.25 * min(t, 4.0)) * t),
        "nom": lambda t: 0.5,
    }

    print("=" * 100)
    print("TUNING GRID  (d=0.20, 20 eps each; aim: total coll=0, nominal err near P+FF's 2.54)")
    print("=" * 100)
    print(f"  {'margin':>7s} {'safety':>7s} {'yth':>5s} {'ymax':>5s} "
          f"{'coll/140':>9s} {'nom_err':>8s} {'max_err':>8s}")
    best = None
    for margin in [0.01, 0.02, 0.04]:
        for safety in [0.9, 1.0, 1.15]:
            for yth in [0.30, 0.50]:
                for ymax in [0.0, 0.05]:
                    pol = Pole(margin=margin, safety=safety,
                               ythresh=yth, ymax=ymax)
                    tot, nom, mx = 0, None, 0.0
                    for aname, fn in probes.items():
                        c, e = run(pol, fn, 0.20, n_ep=20)
                        tot += c
                        mx = max(mx, e)
                        if aname == "nom":
                            nom = e
                    print(f"  {margin:>7.2f} {safety:>7.2f} {yth:>5.2f} {ymax:>5.2f} "
                          f"{tot:>9d} {nom:>8.2f} {mx:>8.2f}")
                    if tot == 0:
                        if best is None or nom < best[0]:
                            best = (nom, margin, safety, yth, ymax)
    print()
    if best:
        print(f"  BEST zero-collision config: nominal err {best[0]:.2f} "
              f"(margin={best[1]}, safety={best[2]}, yth={best[3]}, ymax={best[4]})")
    else:
        print("  NO zero-collision config in this grid")
