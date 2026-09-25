"""Review probe round 12, parts C+D.

C  reproduce the H-A3'(c) published noise band (frozen v1 RL on B: 36 -> 40)
D  the SHARPEST freeze degenerate: "catch up nominally, then stop trying".
   Review1's shape was "nominal P+FF, freeze under attack".  The version that
   maximises the chance of slipping through (b)(c)(e) is one that reaches the
   setpoint using the normal controller and then simply stops acting -- the V2V
   prior then matches the leader speed, so the gap is HELD.  Does the five-clause
   conjunction admit it?
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import importlib.util
import numpy as np
import follow_env as fe
from stable_baselines3 import PPO

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


class CatchThenFreeze:
    """Plain P+FF until |e| < eps, then a == 0 forever.

    Justification as a "degenerate": it never does anything to REJECT a
    disturbance -- once at the setpoint it simply stops, and the environment's
    V2V speed-matching prior holds the gap whatever the leader does.  It has no
    braking logic at all.  If this passes (a)-(e), the conjunction measures the
    prior, not the defender.
    """

    def __init__(self, eps=0.02):
        self.eps = eps

    def reset(self):
        self.frozen = False
        self.n_frozen = self.n = 0

    def __call__(self, obs):
        e = obs[0] * 0.5
        self.n += 1
        if abs(e) < self.eps:
            self.frozen = True
        if self.frozen:
            self.n_frozen += 1
            return np.zeros(1, dtype=np.float32)
        return p_ff(obs)


class CatchThenFreezeGap:
    def __init__(self, g=0.25):
        self.g = g

    def reset(self):
        self.frozen = False
        self.n_frozen = self.n = 0

    def __call__(self, obs):
        e = obs[0] * 0.5
        self.n += 1
        if e + fe.D_DES < self.g:
            self.frozen = True
        if self.frozen:
            self.n_frozen += 1
            return np.zeros(1, dtype=np.float32)
        return p_ff(obs)


class FreezeEnvelope:
    def __init__(self):
        self.f = load("pole_mod2", BASE + "/positive_pole.py")

    def reset(self):
        self.fires = self.steps = 0

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        cm = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        self.steps += 1
        if -de > cm:
            self.fires += 1
            return np.zeros(1, dtype=np.float32)
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


def run_b(policy, seed0=50000, env0=60000, n_ep=20, d=0.20):
    fe.D_DES = d
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    try:
        crash, err, mg = {}, {}, {}
        for kind in ("B1", "B2", "B3", "B4", "B5"):
            c, es, gs = 0, [], []
            for k in range(n_ep):
                if hasattr(policy, "reset"):
                    policy.reset()
                obs, _ = env.reset(seed=env0 + k)
                env.set_fn(b_fn(kind, seed0 + k))
                while True:
                    obs, r, term, trunc, _ = env.step(policy(obs))
                    if term or trunc:
                        break
                g = np.array(env.log["gap"])
                s = g[int(len(g) * 0.2):]
                es.append(float(np.abs(s - d).mean() * 200))
                gs.append(float(s.mean()))
                c += int(env.term_reason == "collision")
            crash[kind], err[kind], mg[kind] = c, float(np.mean(es)), float(np.mean(gs))
        return crash, err, mg
    finally:
        fe.D_DES = 0.20


def run_nom(policy, d=0.20, n_ep=20, seed0=2000):
    fe.D_DES = d
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    try:
        es = []
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
        return float(np.mean(es))
    finally:
        fe.D_DES = 0.20


print("=" * 104)
print("D  SHARP FREEZE DEGENERATES: 'catch up nominally, then stop acting'")
print("=" * 104)
print(f"  {'policy':>30s} " + "".join(f"{k:>7s}" for k in ("B1", "B2", "B3", "B4", "B5"))
      + f" {'TOTAL':>8s} {'errB_pool':>9s} {'nom_err':>8s} {'frozen%':>8s} {'meanGap':>8s}")
for lbl, mk in [("pole (anchor)", lambda: pp.PositivePole()),
                ("P+FF", lambda: p_ff),
                ("catch-then-freeze |e|<.02", lambda: CatchThenFreeze(0.02)),
                ("catch-then-freeze gap<.25", lambda: CatchThenFreezeGap(0.25)),
                ("freeze-envelope (brake->a0)", lambda: FreezeEnvelope()),
                ("zero action", lambda: zero)]:
    pol = mk()
    crash, err, mg = run_b(pol)
    ne = run_nom(mk())
    pooled = float(np.mean(list(err.values())))
    fz = ""
    if hasattr(pol, "n_frozen"):
        fz = f"{100.0*pol.n_frozen/max(pol.n,1):.1f}%"
    elif hasattr(pol, "steps"):
        fz = f"{100.0*pol.fires/max(pol.steps,1):.1f}%"
    row = f"  {lbl:>30s} " + "".join(f"{crash[k]:>5d}/20" for k in ("B1", "B2", "B3", "B4", "B5"))
    row += f" {sum(crash.values()):>5d}/100 {pooled:>9.2f} {ne:>8.2f} {fz:>8s}"
    row += f" {np.mean(list(mg.values())):>8.3f}"
    print(row)

print()
print("=" * 104)
print("D2  THE FIVE-CLAUSE CHECK (baseline = frozen v1 RL, 36/100; gate on (e) =")
print("    1.5 x pole's OWN B-group pooled err = 5.98, i.e. the correctly anchored form)")
print("=" * 104)
pole_c, pole_e, _ = run_b(pp.PositivePole())
pole_pool = float(np.mean(list(pole_e.values())))
gate = 1.5 * pole_pool
print(f"  pole B-group pooled err = {pole_pool:.2f}  => correctly anchored (e) gate = {gate:.2f}")
print(f"  (the plan's written gate was 3.28, anchored to the NOMINAL 2.19)")
print()
print(f"  {'policy':>30s} {'(a)':>5s} {'(b)<=10%':>9s} {'(c)>=50%':>9s} {'(e)<=5.98':>10s} {'VERDICT':>9s}")
for lbl, mk in [("pole (anchor)", lambda: pp.PositivePole()),
                ("catch-then-freeze |e|<.02", lambda: CatchThenFreeze(0.02)),
                ("catch-then-freeze gap<.25", lambda: CatchThenFreezeGap(0.25)),
                ("freeze-envelope (brake->a0)", lambda: FreezeEnvelope()),
                ("zero action", lambda: zero),
                ("P+FF", lambda: p_ff)]:
    crash, err, _ = run_b(mk())
    tot = sum(crash.values())
    pooled = float(np.mean(list(err.values())))
    b = tot <= 10
    c = tot <= 0.5 * 36
    e = pooled <= gate
    verdict = "PASS" if (b and c and e) else "fail"
    print(f"  {lbl:>30s} {'OK':>5s} {str(tot)+'/100':>9s} {str(b):>9s} {str(c):>9s} "
          f"{pooled:>10.2f} {verdict:>9s}")

print()
print("=" * 80)
print("C  FROZEN v1 RL on the B group -- reproduce the published 36 -> 40 band")
print("=" * 80)
V1 = REPO + "/ckpt/follow_stage2_final_v1.zip"
model = PPO.load(V1)


def v1_policy(obs):
    a, _ = model.predict(obs, deterministic=True)
    return np.asarray(a, dtype=np.float32)


for s0 in (50000, 77000, 91000):
    crash, err, _ = run_b(v1_policy, seed0=s0)
    print(f"  attack seeds {s0}+:  per-item "
          + " ".join(f"{k}:{crash[k]:>2d}/20" for k in crash)
          + f"   TOTAL {sum(crash.values()):>3d}/100   pooled err "
          f"{np.mean(list(err.values())):.2f}")
print("  review1 published: 36/100 (50000+) and 40/100 (77000+)")
print("\nDONE")
