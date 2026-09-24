"""Positive pole v4: minimal design + conservative grid.

What the previous grid established: nominal error is pinned at P+FF's 2.54 for
every config, so the envelope only binds under attack -- the controller IS P+FF
when nothing hostile is happening. The grid then failed to find zero collisions
because it swept LESS conservative settings than v3 (v3's 0/140 used
margin=0.05, safety=0.7; the grid used margin<=0.04, safety>=0.9).

Two things to test:
  D1  The minimal design: plain P+FF, plus a single emergency brake branch that
      fires when the achieved closing rate exceeds what can be arrested.
      Question: does the emergency branch alone prevent the crashes, keeping
      P+FF's nominal precision untouched?
  D2  A conservative grid around v3's working point, to see whether nominal error
      can be tightened while holding zero collisions.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

ACT = fe.ACT_GAIN


def pff_a(obs, kp=0.8):
    e = float(obs[0]) * 0.5
    return float(np.clip(kp * e / ACT, -1.0, 1.0))


class MinimalEmergency:
    """P+FF, plus: if the closing rate cannot be arrested in the remaining gap,
    brake at full authority. Nothing else."""

    def __init__(self, a_max_lo=1.0, margin=0.05):
        self.a_max_lo, self.margin = a_max_lo, margin

    def reset(self):
        pass

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        c = -de
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - self.margin, 0.0)
        c_max = np.sqrt(2.0 * self.a_max_lo * room)
        if c > c_max:
            return np.array([-1.0], dtype=np.float32)
        return np.array([pff_a(obs)], dtype=np.float32)


class Pole4:
    """P control with the closing-rate ceiling and a cumulative yield term."""

    def __init__(self, kp=0.8, a_max_lo=1.0, margin=0.05, safety=0.8,
                 ythresh=0.25, yg=0.25, ymax=0.08, recover=0.998):
        self.kp, self.a_max_lo = kp, a_max_lo
        self.margin, self.safety = margin, safety
        self.ythresh, self.yg, self.ymax, self.recover = ythresh, yg, ymax, recover
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
            if dv > self.ythresh:
                self.extra = min(self.extra + self.yg * dv, self.ymax)
        self.extra *= self.recover
        self.prev_v = v
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - self.margin, 0.0)
        c_max = self.safety * np.sqrt(2.0 * self.a_max_lo * room)
        if c > c_max:
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

    print("=" * 96)
    print("D1: MINIMAL DESIGN (P+FF + emergency brake only), d=0.20")
    print("=" * 96)
    for margin in [0.02, 0.05, 0.10]:
        pol = MinimalEmergency(margin=margin)
        tot, nom, mx = 0, None, 0.0
        for an, fn in probes.items():
            c, e = run(pol, fn, 0.20, n_ep=20)
            tot += c
            mx = max(mx, e)
            if an == "nom":
                nom = e
        print(f"  margin={margin:.2f}  coll {tot:>3d}/140   "
              f"nom_err {nom:.2f}   max_err {mx:.2f}")

    print()
    print("=" * 96)
    print("D2: CONSERVATIVE GRID around v3's working point, d=0.20")
    print("=" * 96)
    print(f"  {'margin':>7s} {'safety':>7s} {'yth':>5s} {'yg':>5s} {'ymax':>5s} "
          f"{'coll/140':>9s} {'nom_err':>8s}")
    for margin in [0.05, 0.08, 0.12]:
        for safety in [0.6, 0.7, 0.8]:
            for yth in [0.2, 0.4]:
                for ymax in [0.06, 0.12]:
                    pol = Pole4(margin=margin, safety=safety,
                                ythresh=yth, ymax=ymax)
                    tot, nom = 0, None
                    for an, fn in probes.items():
                        c, e = run(pol, fn, 0.20, n_ep=20)
                        tot += c
                        if an == "nom":
                            nom = e
                    print(f"  {margin:>7.2f} {safety:>7.2f} {yth:>5.2f} "
                          f"{0.25:>5.2f} {ymax:>5.2f} {tot:>9d} {nom:>8.2f}")
