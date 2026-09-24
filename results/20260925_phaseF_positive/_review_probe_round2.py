"""Review probe round 2 (round 11).

  S4  forced-win reproduction: zero-action vs P+FF on step-down
  S5  is the pole at d=1.0 / step-down 0.2s actually TRACKING?  (trace)
  S6  sharpened ablation: which half of the envelope is load-bearing?
  S7  attacks NOT in the 11-family: random phase/freq, continuous small steps
  S8  B-group (held-out) overlap, per reviews/20260925_phaseF_holdout_B.md
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
            vl = np.array(env.log["v_l"])
            vf = np.array(env.log["v_f"])
            s = g[int(len(g) * 0.2):]
            ss = slice(int(len(g) * 0.2), len(g))
            recs.append(dict(
                coll=env.term_reason == "collision", gap0=env.gap0,
                err=float(np.abs(s - d_des).mean() * 200),
                err_signed=float((s - d_des).mean() * 200),
                frac_far=float(np.mean(np.abs(s - d_des) > 0.05)),
                mean_gap=float(s.mean()),
                mean_closing=float(np.mean(vl[ss] - vf[ss])),
                min_gap=float(g.min()), t_end=float(t[-1]),
                g=g, t=t, vl=vl, vf=vf))
            coll += int(recs[-1]["coll"])
        return coll, recs
    finally:
        fe.D_DES = 0.20


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


def zero(obs):
    return np.zeros(1, dtype=np.float32)


print("=" * 96)
print("S4  FORCED-WIN REPRODUCTION (independent re-run of _probe_forced_win.py)")
print("=" * 96)
step = lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05
for label, pol, gm in [("zero action, gap_max=1.2", zero, 1.2),
                       ("P+FF,        gap_max=1.2", p_ff, 1.2),
                       ("zero action, gap_max=0.20", zero, 0.20),
                       ("P+FF,        gap_max=0.20", p_ff, 0.20)]:
    c, _ = run2(pol, step, 0.20, n_ep=20, gap_max=gm)
    print(f"  {label:32s}  coll {c:>2d}/20")

print()
print("=" * 96)
print("S5  IS THE POLE TRACKING at d=1.0 / step-down 0.2s?  (worst cell, 78.28 obs-cm)")
print("=" * 96)
sd2 = lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
c, recs = run2(pp.PositivePole(), sd2, 1.00, n_ep=20)
print(f"  coll {c}/20   mean settled err {np.mean([r['err'] for r in recs]):.2f} "
      f"obs-cm   mean gap {np.mean([r['mean_gap'] for r in recs]):.3f} m "
      f"(target 1.00)   mean(mean closing rate) "
      f"{np.mean([r['mean_closing'] for r in recs]):+.3f} m/s")
print(f"  frac of settled time with |gap-1.0| > 5cm: "
      f"{np.mean([r['frac_far'] for r in recs]):.2f}")
r = recs[0]
print("  trace of episode 0 (gap target 1.0, leader 0.95<->0.05 every 0.2 s):")
print("      t     gap    v_l    v_f   gap-1.0")
idx = np.linspace(0, len(r["g"]) - 1, 20).astype(int)
for i in idx:
    print(f"   {r['t'][i]:6.2f} {r['g'][i]:7.3f} {r['vl'][i]:6.3f} {r['vf'][i]:6.3f}"
          f" {r['g'][i]-1.0:+8.3f}")

print()
print("=" * 96)
print("S6  SHARPENED ABLATION: which half of the envelope is load-bearing?")
print("=" * 96)


class ClampOnly(pp.PositivePole):
    """Gap-dependent command clamp kept; the full-brake branch REMOVED."""

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        v = float(obs[2]) * 1.3
        if self.prev_v is not None:
            dv = self.prev_v - v
            if dv > pp.YTHRESH:
                self.extra = min(self.extra + pp.YG * dv, pp.YMAX)
        self.extra *= pp.RECOVER
        self.prev_v = v
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        c_max = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        a = pp.KP * (e - self.extra) / ACT
        return np.array([float(np.clip(a, -1.0, min(1.0, c_max / ACT)))],
                        dtype=np.float32)


class BrakeOnly(pp.PositivePole):
    """Full-brake branch kept; the command clamp REMOVED."""

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        v = float(obs[2]) * 1.3
        c = -de
        if self.prev_v is not None:
            dv = self.prev_v - v
            if dv > pp.YTHRESH:
                self.extra = min(self.extra + pp.YG * dv, pp.YMAX)
        self.extra *= pp.RECOVER
        self.prev_v = v
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        c_max = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        if c > c_max:
            return np.array([-1.0], dtype=np.float32)
        a = pp.KP * (e - self.extra) / ACT
        return np.array([float(np.clip(a, -1.0, 1.0))], dtype=np.float32)


class HardGapBrake(pp.PositivePole):
    """Trivial 1-line alternative: P+FF, but full brake whenever gap < X."""

    def __init__(self, x):
        self.x = x
        super().__init__()

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        gap = e + fe.D_DES
        if gap < self.x:
            return np.array([-1.0], dtype=np.float32)
        a = pp.KP * e / ACT
        return np.array([float(np.clip(a, -1.0, 1.0))], dtype=np.float32)


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
ABL = {
    "pole (final)": pp.PositivePole(),
    "clamp only (no brake)": ClampOnly(),
    "brake only (no clamp)": BrakeOnly(),
    "P+FF + hard brake g<0.20": HardGapBrake(0.20),
    "P+FF + hard brake g<0.30": HardGapBrake(0.30),
    "P+FF, no envelope": p_ff,
}
for d in (0.20, 1.00):
    print(f"\n  --- d = {d} ---")
    print(f"  {'policy':>26s} " + "".join(f"{k.split()[0][:8]:>10s}" for k in PROBES)
          + f"{'TOT':>7s}")
    for name, pol in ABL.items():
        tot, out = 0, ""
        for pname, fn in PROBES.items():
            c, _ = run2(pol, fn, d, n_ep=20)
            tot += c
            out += f"{c:>7d}/20"
        print(f"  {name:>26s} " + out + f"{tot:>4d}/220")

print()
print("=" * 96)
print("S7  ATTACKS OUTSIDE THE 11-FAMILY: random phase/freq, continuous small steps")
print("=" * 96)
rng = np.random.default_rng(7777)


def rand_sine(t):
    return 0.5 + rng.uniform(0.30, 0.50) * np.sin(
        2 * np.pi * rng.uniform(0.2, 1.0) * t + rng.uniform(0, 2 * np.pi))


def rand_sine_fn(seed):
    r = np.random.default_rng(seed)
    f = r.uniform(0.2, 1.0)
    a = r.uniform(0.30, 0.50)
    ph = r.uniform(0, 2 * np.pi)
    return lambda t: 0.5 + a * np.sin(2 * np.pi * f * t + ph)


def small_steps_fn(seed, step=0.15, hold=0.15):
    r = np.random.default_rng(seed)
    n_seg = int(20.0 / hold) + 2
    vals = np.clip(0.5 + np.cumsum(r.uniform(-step, step, n_seg)), 0.05, 0.95)
    return lambda t: float(vals[min(int(t / hold), n_seg - 1)])


def tiny_steps_fn(seed, step=0.08, hold=0.10):
    return small_steps_fn(seed, step, hold)


def rand_hold_fn(seed, lo=0.2, hi=0.8):
    return srch.sched_gap  # placeholder, replaced below


print(f"  {'attack':>34s}  {'pole':>10s}  {'P+FF':>10s}")
for name, s0 in [("rand sine f~U(.2,1.0) a~U(.3,.5) ph~U(0,2pi)", 30000),
                 ("small steps +-0.15 / 0.15s", 31000),
                 ("tiny steps +-0.08 / 0.10s", 32000),
                 ("small steps +-0.25 / 0.25s", 33000)]:
    if "sine" in name:
        fns = [rand_sine_fn(s0 + k) for k in range(20)]
    elif "tiny" in name:
        fns = [tiny_steps_fn(s0 + k) for k in range(20)]
    elif "0.25" in name:
        fns = [small_steps_fn(s0 + k, 0.25, 0.25) for k in range(20)]
    else:
        fns = [small_steps_fn(s0 + k) for k in range(20)]
    for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
        tot = 0
        for k in range(20):
            c, _ = run2(pol, fns[k], 0.20, n_ep=1, seed0=40000 + k)
            tot += c
        print(f"  {name if lbl=='pole' else '':>34s}  {tot:>3d}/20   " if False else "", end="")
    # cleaner: accumulate both then print
    res = {}
    for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
        tot = 0
        for k in range(20):
            c, _ = run2(pol, fns[k], 0.20, n_ep=1, seed0=40000 + k)
            tot += c
        res[lbl] = tot
    print(f"  {name:>34s}  {res['pole']:>3d}/20      {res['P+FF']:>3d}/20")

print()
print("=" * 96)
print("S8  HELD-OUT B-GROUP OVERLAP (as defined in reviews/20260925_phaseF_holdout_B.md)")
print("=" * 96)


def b_fn(kind, seed):
    r = np.random.default_rng(seed)
    if kind == "B1":                       # square, period {0.3,0.7}, duty [0.3,0.7]
        per = float(r.choice([0.3, 0.7]))
        duty = float(r.uniform(0.3, 0.7))
        return lambda t, per=per, duty=duty: 0.95 if (t % per) < duty * per else 0.05
    if kind == "B2":                       # sine, f {0.3,0.8}, amp {0.3,0.5}, rand phase
        f = float(r.choice([0.3, 0.8]))
        a = float(r.choice([0.3, 0.5]))
        ph = float(r.uniform(0, 2 * np.pi))
        return lambda t, f=f, a=a, ph=ph: 0.5 + a * np.sin(2 * np.pi * f * t + ph)
    if kind == "B3":                       # chirp 0.1 -> 0.6 Hz
        return lambda t: 0.5 + 0.45 * np.sin(
            2 * np.pi * (0.1 + (0.5 / 4.0) * min(t, 4.0)) * t)
    if kind == "B4":                       # random hold, seg 0.2-0.8s, val [0.05,1.0]
        segs, t_acc = [], 0.0
        while t_acc < 22.0:
            segs.append(float(r.uniform(0.2, 0.8)))
            t_acc += segs[-1]
        vals = r.uniform(0.05, 1.0, len(segs))

        def f(t, segs=segs, vals=vals):
            acc = 0.0
            for i, s in enumerate(segs):
                acc += s
                if t < acc:
                    return float(vals[i])
            return float(vals[-1])
        return f
    if kind == "B5":                       # double pulse, sep {0.2,0.5}
        sep = float(r.choice([0.2, 0.5]))

        def f(t, sep=sep):
            per = 3.0
            tt = t % per
            return 0.05 if (tt < 0.1 or sep <= tt < sep + 0.1) else 0.95
        return f
    raise ValueError(kind)


print(f"  {'item':>6s}  {'pole coll':>10s}  {'pole err':>9s}  {'P+FF coll':>10s}")
for kind in ("B1", "B2", "B3", "B4", "B5"):
    for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
        tot, errs = 0, []
        for k in range(20):
            fn = b_fn(kind, 50000 + k)
            c, recs = run2(pol, fn, 0.20, n_ep=1, seed0=60000 + k)
            tot += c
            errs.append(recs[0]["err"])
        if lbl == "pole":
            print(f"  {kind:>6s}  {tot:>8d}/20  {np.mean(errs):>7.2f}   ", end="")
        else:
            print(f"{tot:>8d}/20")

print("\nDONE")
