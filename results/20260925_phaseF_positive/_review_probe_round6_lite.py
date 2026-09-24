"""Review probe round 6 lite (round 11) -- the one decisive number left for the
H-A3 baseline question, plus a compact attack-family search.

  W1  ZERO ACTION on the B group  -> is H-A3 (crash-rate only) satisfiable by
      the laziest possible policy?
  W2  1500-candidate random waveform search against the pole (item C)
  W3  branch occupancy, all 11 families (item A)
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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pp = load("pole_mod", BASE + "/positive_pole.py")
srch = load("search_mod", BASE + "/positive_pole_search.py")
ScriptedLeader = srch.ScriptedLeader
ACT = fe.ACT_GAIN
zero = lambda obs: np.zeros(1, dtype=np.float32)


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


def crash_of(policy, fn, d_des, seed=2000, gap_max=1.2, want_env=False):
    fe.D_DES = d_des
    env = ScriptedLeader(domain_randomize=False, gap_max=gap_max)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        obs, _ = env.reset(seed=seed)
        env.set_fn(fn)
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            if term or trunc:
                break
        c = int(env.term_reason == "collision")
        if want_env:
            g = np.array(env.log["gap"])[int(len(env.log["gap"]) * 0.2):]
            return c, env.term_reason, float(np.abs(g - d_des).mean() * 200), g.mean()
        return c, env.term_reason, None, None
    finally:
        fe.D_DES = 0.20


def b_fn(kind, seed):
    r = np.random.default_rng(seed)
    if kind == "B1":
        per, duty = float(r.choice([0.3, 0.7])), float(r.uniform(0.3, 0.7))
        return lambda t, per=per, duty=duty: 0.95 if (t % per) < duty * per else 0.05
    if kind == "B2":
        f, a, ph = float(r.choice([0.3, 0.8])), float(r.choice([0.3, 0.5])), float(r.uniform(0, 2 * np.pi))
        return lambda t, f=f, a=a, ph=ph: 0.5 + a * np.sin(2 * np.pi * f * t + ph)
    if kind == "B3":
        return lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * (0.1 + 0.125 * min(t, 4.0)) * t)
    if kind == "B4":
        segs, acc = [], 0.0
        while acc < 22.0:
            segs.append(float(r.uniform(0.2, 0.8)))
            acc += segs[-1]
        vals = r.uniform(0.05, 1.0, len(segs))

        def f(t, segs=segs, vals=vals):
            acc = 0.0
            for i, s in enumerate(segs):
                acc += s
                if t < acc:
                    return float(vals[i])
            return float(vals[-1])
        return f
    if kind == "B5":
        sep = float(r.choice([0.2, 0.5]))

        def f(t, sep=sep):
            tt = t % 3.0
            return 0.05 if (tt < 0.1 or sep <= tt < sep + 0.1) else 0.95
        return f


print("=" * 96)
print("W1  ZERO ACTION on the B group (d=0.20, env seeds 60000+k, attack seeds 50000+k)")
print("=" * 96)
print(f"  {'policy':>12s} " + "".join(f"{k:>8s}" for k in ("B1", "B2", "B3", "B4", "B5"))
      + f"{'TOTAL':>9s}  {'nominal err':>12s}")
for lbl, pol in [("zero action", zero), ("P+FF", p_ff), ("pole", pp.PositivePole())]:
    row, tot, errs = "", 0, []
    for kind in ("B1", "B2", "B3", "B4", "B5"):
        c = 0
        for k in range(20):
            fn = b_fn(kind, 50000 + k)
            cc, _, e, mg = crash_of(pol, fn, 0.20, seed=60000 + k, want_env=True)
            c += cc
            errs.append(e)
        tot += c
        row += f"{c:>5d}/20"
    ne, _, neerr, _ = crash_of(pol, lambda t: 0.5, 0.20, seed=2000, want_env=True)
    print(f"  {lbl:>12s} " + row + f"{tot:>6d}/100  {neerr:>12.2f}")

print()
print("=" * 96)
print("W2  RANDOM WAVEFORM SEARCH vs the pole (d=0.20, 1500 candidates)")
print("=" * 96)
rng = np.random.default_rng(20260925)
found = []
for i in range(1500):
    kind = int(rng.integers(0, 6))
    if kind == 0:
        per, duty = rng.uniform(0.05, 1.5), rng.uniform(0.2, 0.8)
        lo, hi = rng.uniform(0.05, 0.5), rng.uniform(0.5, 1.0)
        fn = lambda t, per=per, duty=duty, lo=lo, hi=hi: (hi if (t % per) < duty * per else lo)
    elif kind == 1:
        f, a = rng.uniform(0.1, 3.0), rng.uniform(0.1, 0.5)
        ph, off = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 0.7)
        fn = lambda t, f=f, a=a, ph=ph, off=off: off + a * np.sin(2 * np.pi * f * t + ph)
    elif kind == 2:
        seg = rng.uniform(0.05, 0.8)
        rr = np.random.default_rng(int(rng.integers(0, 10**6)))
        vals = rr.uniform(0.05, 1.0, 500)
        fn = lambda t, seg=seg, vals=vals: float(vals[int(t / seg) % 500])
    elif kind == 3:
        t_up = rng.uniform(0.1, 4.0)
        hi, lo = rng.uniform(0.6, 1.0), rng.uniform(0.05, 0.2)
        fn = lambda t, t_up=t_up, hi=hi, lo=lo: (hi if t < t_up else lo)
    elif kind == 4:
        f0, f1 = rng.uniform(0.05, 0.5), rng.uniform(0.5, 3.0)
        a = rng.uniform(0.2, 0.5)
        fn = lambda t, f0=f0, f1=f1, a=a: 0.5 + a * np.sin(
            2 * np.pi * (f0 + (f1 - f0) / 4.0 * min(t, 4.0)) * t)
    else:
        sep, w, per = rng.uniform(0.1, 1.5), rng.uniform(0.02, 0.3), rng.uniform(1.0, 4.0)
        fn = lambda t, sep=sep, w=w, per=per: (
            0.05 if (t % per) < w or (sep <= (t % per) < sep + w) else 0.95)
    c, _, _, _ = crash_of(pp.PositivePole(), fn, 0.20, seed=int(rng.integers(0, 10**6)))
    if c:
        found.append((i, kind))
print(f"  candidates: 1500   crashes: {len(found)}")
for i, k in found[:8]:
    print(f"    candidate {i} (kind {k})")
if not found:
    print("  => no crossing waveform found")

print()
print("=" * 96)
print("W3  BRANCH OCCUPANCY, 11 families, d=0.20, 20 ep each")
print("=" * 96)


class Occ(pp.PositivePole):
    def reset(self):
        super().reset()
        self.n = self.nb = self.ncl = self.ny = 0

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        v = float(obs[2]) * 1.3
        if self.prev_v is not None and (self.prev_v - v) > pp.YTHRESH:
            self.ny += 1
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        cm = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        if -de > cm:
            self.nb += 1
        elif pp.KP * (e - self.extra) / ACT > min(1.0, cm / ACT):
            self.ncl += 1
        self.n += 1
        return super().__call__(obs)


PROBES = {
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
print(f"  {'attack':>16s} {'steps':>7s} {'brake%':>8s} {'clamp%':>8s} {'yield%':>8s}")
for an, fn in PROBES.items():
    o = Occ()
    for k in range(20):
        crash_of(o, fn, 0.20, seed=2000 + k)
    print(f"  {an:>16s} {o.n:>7d} {100.0*o.nb/o.n:>7.1f}% "
          f"{100.0*o.ncl/o.n:>7.1f}% {100.0*o.ny/o.n:>7.1f}%")
print("\nDONE")
