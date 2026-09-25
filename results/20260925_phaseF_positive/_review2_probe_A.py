"""Review probe round 12 (Phase F positive review 2), item A -- THE core question.

Does H-A3'(e) still leave a degenerate channel?

H-A3' (e) is:  B-group settled error <= 1.5x the pole's own value, quoted in the
plan as "pole d=0.2 = 2.19 obs-cm => gate <= 3.28".

Two things have to be true for that clause to do the job it is claimed to do
("the ONLY clause that excludes 'precise nominally + freeze under attack'"):
  (1) the pole ITSELF must pass it on the B group -- i.e. the pole's B-group
      settled error must be <= 3.28, else the gate is anchored to a number
      (the NOMINAL 2.19) that its own anchor cannot meet under attack;
  (2) the freeze degenerates must FAIL it, i.e. their B-group settled error must
      exceed the pole's by more than the 1.5x band.

So this probe measures the B-group settled error of:
  pole / zero-action / P+FF / freeze-on-attack family / crawl-family degenerates
using exactly the B-group instance definition of _review_probe_round6_lite.py
(env seed 60000+k, attack seed 50000+k, 20 episodes per item).
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


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


zero = lambda obs: np.zeros(1, dtype=np.float32)


class FreezeGap:
    """Nominal: P+FF.  Under threat: freeze (a=0 -> v_cmd = v_leader, speed match).

    Threat detector is GAP-KEYED so that it does NOT fire during the legitimate
    catch-up phase (large gap): freeze only when the gap is already small and a
    closing rate has built up.  This is the most favourable form of the
    "precise nominally, freeze under attack" degenerate -- the one review1 said
    H-A4 could not exclude.
    """

    def __init__(self, g_thresh=0.40, c_thresh=0.20):
        self.gt, self.ct = g_thresh, c_thresh

    def reset(self):
        pass

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        gap = e + fe.D_DES
        c = -de
        if gap < self.gt and c > self.ct:
            return np.zeros(1, dtype=np.float32)
        return p_ff(obs)


class FreezeTime:
    """Same idea, but time-keyed: freeze for `hold` seconds once a sharp closing
    rate is seen.  Tests whether memory (rather than instantaneous gate) changes
    the picture."""

    def __init__(self, c_thresh=0.35, hold=0.4):
        self.ct, self.hold = c_thresh, hold

    def reset(self):
        self.t_left = 0.0

    def __call__(self, obs):
        de = obs[1] * 2.0
        if -de > self.ct:
            self.t_left = self.hold
        if self.t_left > 0.0:
            self.t_left -= fe.DT
            return np.zeros(1, dtype=np.float32)
        return p_ff(obs)


class LowGain:
    def __init__(self, kp=0.15):
        self.kp = kp

    def reset(self):
        pass

    def __call__(self, obs):
        e = obs[0] * 0.5
        return np.array([np.clip(self.kp * e / ACT, -1, 1)], dtype=np.float32)


class BigGap:
    """'Always keep a large gap' -- target 0.5 m instead of 0.2 m."""

    def __init__(self, offset=0.30):
        self.off = offset

    def reset(self):
        pass

    def __call__(self, obs):
        e = obs[0] * 0.5 - self.off
        return np.array([np.clip(0.8 * e / ACT, -1, 1)], dtype=np.float32)


class ClampOnly:
    """The pole's envelope applied as a COMMAND clamp only (no full brake) --
    review1's 'clamp only' ablation, which crashed 60/220 on the hard subset."""

    def reset(self):
        pass

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        cm = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        return np.array([float(np.clip(pp.KP * e / ACT, -1.0, min(1.0, cm / ACT)))],
                        dtype=np.float32)


class BrakeOnly:
    """The pole's envelope with the full-authority brake, no clamp --
    review1's 'brake only' ablation: 0/220."""

    def reset(self):
        pass

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        cm = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        if -de > cm:
            return np.array([-1.0], dtype=np.float32)
        return np.array([float(np.clip(pp.KP * e / ACT, -1.0, 1.0))],
                        dtype=np.float32)


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


def run_b(policy, d=0.20, n_ep=20, attack_seed0=50000, env_seed0=60000):
    """Return (per_item_crash, per_item_err, per_item_mean_gap)."""
    fe.D_DES = d
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    try:
        crash, err, mgap = {}, {}, {}
        for kind in ("B1", "B2", "B3", "B4", "B5"):
            c, es, gs = 0, [], []
            for k in range(n_ep):
                if hasattr(policy, "reset"):
                    policy.reset()
                obs, _ = env.reset(seed=env_seed0 + k)
                env.set_fn(b_fn(kind, attack_seed0 + k))
                while True:
                    obs, r, term, trunc, _ = env.step(policy(obs))
                    if term or trunc:
                        break
                g = np.array(env.log["gap"])
                s = g[int(len(g) * 0.2):]
                es.append(float(np.abs(s - d).mean() * 200))
                gs.append(float(s.mean()))
                c += int(env.term_reason == "collision")
            crash[kind], err[kind], mgap[kind] = c, float(np.mean(es)), float(np.mean(gs))
        return crash, err, mgap
    finally:
        fe.D_DES = 0.20


def run_nom(policy, d=0.20, n_ep=20, seed0=2000):
    fe.D_DES = d
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    try:
        es, c = [], 0
        for k in range(n_ep):
            if hasattr(policy, "reset"):
                policy.reset()
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(lambda t: 0.5)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            g = np.array(env.log["gap"])
            s = g[int(len(g) * 0.2):]
            es.append(float(np.abs(s - d).mean() * 200))
            c += int(env.term_reason == "collision")
        return c, float(np.mean(es))
    finally:
        fe.D_DES = 0.20


POLICIES = [
    ("pole (anchor)", lambda: pp.PositivePole()),
    ("zero action", lambda: zero),
    ("P+FF", lambda: p_ff),
    ("freeze gap<0.40 c>0.20", lambda: FreezeGap(0.40, 0.20)),
    ("freeze gap<0.30 c>0.15", lambda: FreezeGap(0.30, 0.15)),
    ("freeze gap<0.50 c>0.25", lambda: FreezeGap(0.50, 0.25)),
    ("freeze time c>0.35 hold0.4", lambda: FreezeTime(0.35, 0.4)),
    ("lowgain kp=0.15", lambda: LowGain(0.15)),
    ("biggap target 0.50", lambda: BigGap(0.30)),
    ("clamp only", lambda: ClampOnly()),
    ("brake only", lambda: BrakeOnly()),
]

print("=" * 104)
print("A  H-A3'(e) SEPARATION TEST -- B-group settled error, d=0.20")
print("   gate as written: B-group settled err <= 1.5 x pole's quoted 2.19 = 3.28")
print("=" * 104)
hdr = f"  {'policy':>27s} {'B1':>6s} {'B2':>6s} {'B3':>6s} {'B4':>6s} {'B5':>6s} " \
      f"{'TOTAL':>8s} {'errB1':>6s} {'errB2':>6s} {'errB3':>6s} {'errB4':>6s} {'errB5':>6s} " \
      f"{'errB_pooled':>11s} {'nom_err':>8s}"
print(hdr)
res = {}
for lbl, mk in POLICIES:
    c, e, m = run_b(mk())
    tot = sum(c.values())
    pooled = float(np.mean([e[k] for k in ("B1", "B2", "B3", "B4", "B5")]))
    nc, ne = run_nom(mk())
    res[lbl] = dict(crash=c, err=e, pooled=pooled, nom=ne, mgap=m)
    row = f"  {lbl:>27s} " + "".join(f"{c[k]:>4d}/20" for k in ("B1", "B2", "B3", "B4", "B5"))
    row += f" {tot:>5d}/100 " + "".join(f"{e[k]:>6.2f}" for k in ("B1", "B2", "B3", "B4", "B5"))
    row += f" {pooled:>11.2f} {ne:>8.2f}"
    print(row)

print()
print("=" * 104)
print("A2  VERDICT LINES (per-item gate, gate value 3.28)")
print("=" * 104)
pole_pool = res["pole (anchor)"]["pooled"]
pole_nom = res["pole (anchor)"]["nom"]
print(f"  pole nominal err (the number 2.19 came from)   : {pole_nom:.2f}")
print(f"  pole B-group POOLED settled err                : {pole_pool:.2f}")
print(f"  gate value in the plan (1.5 x nominal)         : {1.5*pole_nom:.2f}")
print(f"  gate value if anchored to the pole's B group   : {1.5*pole_pool:.2f}")
print()
print(f"  {'policy':>27s} {'B-group pooled':>15s}  pass@3.28  pass@1.5xPoleB")
for lbl, _ in POLICIES:
    v = res[lbl]["pooled"]
    print(f"  {lbl:>27s} {v:>15.2f}   "
          f"{'PASS' if v <= 1.5*pole_nom else 'fail':>5s}      "
          f"{'PASS' if v <= 1.5*pole_pool else 'fail':>5s}")
print()
print("  per-item: does the pole pass its OWN gate on every B item?")
for k in ("B1", "B2", "B3", "B4", "B5"):
    v = res["pole (anchor)"]["err"][k]
    print(f"    {k}: pole {v:>6.2f}  vs gate 3.28 -> "
          f"{'PASS' if v <= 3.28 else 'FAIL'}")
print()
print("  per-item: freeze degenerates vs gate 3.28")
for lbl in ("freeze gap<0.40 c>0.20", "freeze gap<0.30 c>0.15", "freeze time c>0.35 hold0.4"):
    print(f"    {lbl}")
    for k in ("B1", "B2", "B3", "B4", "B5"):
        v = res[lbl]["err"][k]
        print(f"      {k}: {v:>6.2f}  vs 3.28 -> {'PASS' if v <= 3.28 else 'FAIL'}")

print()
print("=" * 104)
print("A3  B-group mean gap (are these policies actually tracking, or parked?)")
print("=" * 104)
for lbl, _ in POLICIES:
    m = res[lbl]["mgap"]
    print(f"  {lbl:>27s} " + "".join(f"{k}:{m[k]:.3f} " for k in ("B1", "B2", "B3", "B4", "B5")))
print("\nDONE")
