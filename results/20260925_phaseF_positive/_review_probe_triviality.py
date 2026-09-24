"""Review probe (round 11) -- is the Phase F positive pole's safety mechanism
DISTINCT from a trivial "cap the closing rate / follow slowly" solution?

Sections
  S1  yield-branch liveness: is YTHRESH=0.20 even reachable given a_max*DT?
  S2  catch-up agility, nominal (leader const 0.5, d=0.20, gap0 ~ U(0.2,1.2)):
      pole vs P+FF vs constant-rate-cap vs small-gain P (= "crawl")
  S3  ablations over the hard attack subset at d=0.20

Read-only w.r.t. frozen files: only imports them and monkeypatches fe.D_DES
at run time (restored in a finally block).
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


# ---------------------------------------------------------------- policies --
class CountPole(pp.PositivePole):
    """PositivePole + branch activity counters."""

    def reset(self):
        super().reset()
        self.n = 0
        self.n_brake = 0
        self.n_clamp = 0
        self.n_yield = 0
        self.max_dv = 0.0

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        v = float(obs[2]) * 1.3
        c = -de
        if self.prev_v is not None:
            dv = self.prev_v - v
            self.max_dv = max(self.max_dv, dv)
            if dv > pp.YTHRESH:
                self.n_yield += 1
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        c_max = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        if c > c_max:
            self.n_brake += 1
        elif pp.KP * (e - self.extra) / ACT > min(1.0, c_max / ACT):
            self.n_clamp += 1
        self.n += 1
        return super().__call__(obs)


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


def make_const_cap(cap):
    def f(obs):
        e = obs[0] * 0.5
        return np.array([np.clip(0.8 * e / ACT, -1.0, cap / ACT)], dtype=np.float32)
    return f


def make_slowp(kp):
    def f(obs):
        e = obs[0] * 0.5
        return np.array([np.clip(kp * e / ACT, -1.0, 1.0)], dtype=np.float32)
    return f


def zero(obs):
    return np.zeros(1, dtype=np.float32)


# ---------------------------------------------------------------- harness ---
def run2(policy, fn, d_des, n_ep=20, seed0=2000, gap_max=1.2):
    fe.D_DES = d_des
    env = ScriptedLeader(domain_randomize=False, gap_max=gap_max)
    recs = []
    try:
        coll = 0
        for k in range(n_ep):
            if hasattr(policy, "reset"):
                policy.reset()
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            g = np.array(env.log["gap"])
            t = np.array(env.log["t"])
            s = g[int(len(g) * 0.2):]
            err = float(np.abs(s - d_des).mean() * 200)
            idx = np.where(g <= d_des + 0.05)[0]
            tc = float(t[idx[0]]) if len(idx) else float("nan")
            isc = env.term_reason == "collision"
            recs.append(dict(gap0=env.gap0, coll=isc, err=err, t_catch=tc,
                             min_gap=float(g.min()),
                             peak_close=float(np.max(-np.diff(g) / fe.DT)
                                              if len(g) > 1 else 0.0)))
            coll += int(isc)
        return coll, recs
    finally:
        fe.D_DES = 0.20


PROBES = {
    "step-down 0.1s": lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05,
    "step-down 0.2s": lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05,
    "step-down 0.4s": lambda t: 0.95 if int(t / 0.4) % 2 == 0 else 0.05,
    "osc 0.5Hz": lambda t: 0.5 + 0.45 * np.sin(np.pi * t),
    "osc 0.8Hz": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 0.8 * t),
    "chirp": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * (0.1 + 0.25 * min(t, 4.0)) * t),
    "pw-const": lambda t: [0.3, 0.9, 0.15, 0.7, 0.4][int(t / 0.4) % 5],
    "nominal": lambda t: 0.5,
}
NOMINAL = PROBES["nominal"]

print("=" * 100)
print("S1  YIELD-BRANCH LIVENESS  (YTHRESH=%.2f, a_max*DT=%.3f m/s per step)"
      % (pp.YTHRESH, 1.5 * fe.DT))
print("=" * 100)
for aname in ("nominal", "step-down 0.1s"):
    pol = CountPole()
    c, _ = run2(pol, PROBES[aname], 0.20, n_ep=20)
    print(f"  {aname:>16s}  steps {pol.n:>6d}  yield-branch fired {pol.n_yield:>6d} "
          f"({100.0*pol.n_yield/pol.n:5.1f}%)  brake {pol.n_brake:>6d} "
          f"({100.0*pol.n_brake/pol.n:5.1f}%)  clamp {pol.n_clamp:>6d} "
          f"({100.0*pol.n_clamp/pol.n:5.1f}%)  max dv {pol.max_dv:.4f}")

print()
print("=" * 100)
print("S2  CATCH-UP AGILITY, nominal, d=0.20 (leader const 0.5, gap0 ~ U(0.2,1.2))")
print("=" * 100)
POLICIES = {
    "pole (final)": CountPole(),
    "P+FF (ref)": p_ff,
    "const-cap 0.15 (crawl)": make_const_cap(0.15),
    "const-cap 0.30": make_const_cap(0.30),
    "slowP kp=0.15 (crawl)": make_slowp(0.15),
    "zero action": zero,
}
rows = {}
for name, pol in POLICIES.items():
    c, recs = run2(pol, NOMINAL, 0.20, n_ep=20)
    rows[name] = recs
    tc = np.array([r["t_catch"] for r in recs])
    err = np.array([r["err"] for r in recs])
    pk = np.array([r["peak_close"] for r in recs])
    print(f"  {name:>24s}  coll {c:>2d}/20  t_catch mean {np.nanmean(tc):5.2f}s "
          f"max {np.nanmax(tc):5.2f}s  settled err {err.mean():6.2f}  "
          f"peak closing {pk.mean():5.2f} m/s")

print()
print("  per-episode (sorted by initial gap), t_catch [s] and settled err [obs-cm]:")
hdr = "    gap0 " + "".join(f"{k[:18]:>21s}" for k in POLICIES)
print(hdr)
order = np.argsort([r["gap0"] for r in rows["pole (final)"]])
for i in order:
    line = f"    {rows['pole (final)'][i]['gap0']:.3f}"
    for k in POLICIES:
        r = rows[k][i]
        line += f"  t{r['t_catch']:5.2f}/e{r['err']:5.2f}"
    print(line)

print()
print("=" * 100)
print("S3  ABLATIONS on the hard subset, d=0.20")
print("=" * 100)


class NoYield(pp.PositivePole):
    """Envelope intact, 'anticipate' term disabled."""

    def reset(self):
        super().reset()
        self._orig = pp.YTHRESH

    def __call__(self, obs):
        save = pp.YTHRESH
        pp.YTHRESH = 1e9
        out = super().__call__(obs)
        pp.YTHRESH = save
        return out


class NoEnvelope(pp.PositivePole):
    """'anticipate' term intact, closing-rate envelope disabled -> pure P+FF."""

    def __call__(self, obs):
        save = pp.SAFETY
        pp.SAFETY = 1e9
        out = super().__call__(obs)
        pp.SAFETY = save
        return out


ABL = {
    "pole (final)": pp.PositivePole(),
    "pole, no yield": NoYield(),
    "pole, no envelope (=P+FF)": NoEnvelope(),
    "const-cap 0.15 (crawl)": make_const_cap(0.15),
    "slowP kp=0.15 (crawl)": make_slowp(0.15),
}
print(f"  {'policy':>26s} " + "".join(f"{k.split()[0][:9]:>11s}" for k in PROBES) +
      f"{'TOTAL':>8s}")
for name, pol in ABL.items():
    tot = 0
    out = ""
    for pname, fn in PROBES.items():
        c, _ = run2(pol, fn, 0.20, n_ep=20)
        tot += c
        out += f"{c:>8d}/20"
    print(f"  {name:>26s} " + out + f"{tot:>6d}/160")
print("\nDONE")
