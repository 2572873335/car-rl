"""Review probe round 12, part B -- supporting measurements for item A.

  B1  reproduce review1's exact numbers (single-episode nominal conventions)
  B2  branch occupancy of the freeze degenerates  -- is "freeze" real, or a
      relabelling of P+FF?
  B3  seed band of the pole's B-group settled error  -- is a 6% gap resolvable?
  B4  the most dangerous degenerate: the pole's OWN envelope rule with the
      full-authority brake replaced by a=0 (freeze).  Does the channel widen?
  B5  step-down hard subset for the freeze variants (the pole's 0/220 claim)
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


class FreezeGap:
    def __init__(self, g_thresh=0.40, c_thresh=0.20):
        self.gt, self.ct = g_thresh, c_thresh

    def reset(self):
        self.fires = self.steps = 0

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        gap, c = e + fe.D_DES, -de
        self.steps += 1
        if gap < self.gt and c > self.ct:
            self.fires += 1
            return np.zeros(1, dtype=np.float32)
        return p_ff(obs)


class FreezeEnvelope:
    """The pole's envelope, but the full-authority brake replaced by a=0."""

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


class SmartFreeze:
    """'Attack-gated freeze': stay exactly P+FF while the leader looks nominal,
    freeze (a=0) whenever the leader's speed departs from its own recent mean by
    more than a threshold.  This is the strongest causal form of
    'precise nominally + freeze under attack' avail-ABLE from the 4-dim obs."""

    def __init__(self, thresh=0.10, hold=0.3):
        self.th, self.hold = thresh, hold

    def reset(self):
        self.v_prev = None
        self.t_left = 0.0
        self.fires = self.steps = 0

    def __call__(self, obs):
        v = float(obs[2]) * 1.3          # own speed ~ leader speed at a=0
        self.steps += 1
        if self.v_prev is not None and abs(v - self.v_prev) > self.th:
            self.t_left = self.hold
        self.v_prev = v
        if self.t_left > 0.0:
            self.t_left -= fe.DT
            self.fires += 1
            return np.zeros(1, dtype=np.float32)
        return p_ff(obs)


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
        crash, err = {}, {}
        for kind in ("B1", "B2", "B3", "B4", "B5"):
            c, es = 0, []
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
                c += int(env.term_reason == "collision")
            crash[kind], err[kind] = c, float(np.mean(es))
        return crash, err
    finally:
        fe.D_DES = 0.20


def one_ep(policy, fn, d=0.20, seed=2000):
    fe.D_DES = d
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        obs, _ = env.reset(seed=seed)
        env.set_fn(fn)
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            if term or trunc:
                break
        g = np.array(env.log["gap"])
        s = g[int(len(g) * 0.2):]
        return int(env.term_reason == "collision"), float(np.abs(s - d).mean() * 200)
    finally:
        fe.D_DES = 0.20


HARD = {
    "step-down 0.1s": lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05,
    "step-down 0.2s": lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05,
    "step-down 0.4s": lambda t: 0.95 if int(t / 0.4) % 2 == 0 else 0.05,
}

print("=" * 100)
print("B1  REPRODUCE review1's conventions: single-episode (seed 2000) nominal")
print("=" * 100)
for lbl, pol in [("pole", pp.PositivePole()), ("zero action", zero), ("P+FF", p_ff)]:
    c, e = one_ep(pol, lambda t: 0.5)
    print(f"  {lbl:>12s}  coll {c}  nominal err {e:>7.2f}   (review1 quoted: "
          f"{'2.19' if lbl=='pole' else ('83.22' if lbl=='zero action' else '2.49')})")

print()
print("=" * 100)
print("B2  FREEZE BRANCH OCCUPANCY in the B group  (does 'freeze' actually fire?)")
print("=" * 100)
for lbl, pol in [("freeze gap<0.40 c>0.20", FreezeGap(0.40, 0.20)),
                 ("freeze gap<0.30 c>0.15", FreezeGap(0.30, 0.15)),
                 ("freeze envelope a=0", FreezeEnvelope()),
                 ("smart freeze dv>0.10 h0.3", SmartFreeze(0.10, 0.3))]:
    crash, err = run_b(pol)
    occ = pol.fires / max(pol.steps, 1)
    print(f"  {lbl:>26s}  freeze fires {pol.fires}/{pol.steps} = {100*occ:5.2f}%   "
          f"crash {sum(crash.values()):>3d}/100   errB pooled "
          f"{np.mean(list(err.values())):.2f}")

print()
print("=" * 100)
print("B3  SEED BAND: pole B-group settled error on three attack-seed blocks")
print("=" * 100)
for s0 in (50000, 77000, 91000):
    crash, err = run_b(pp.PositivePole(), seed0=s0)
    print(f"  attack seeds {s0}+ :  crash {sum(crash.values()):>3d}/100  "
          f"per-item " + " ".join(f"{k}:{err[k]:.2f}" for k in ("B1", "B2", "B3", "B4", "B5"))
          + f"   pooled {np.mean(list(err.values())):.2f}")

print()
print("=" * 100)
print("B4  DEGENERATE SEPARATION: pole envelope vs freeze-envelope, B group")
print("=" * 100)
cr_p, e_p = run_b(pp.PositivePole())
cr_f, e_f = run_b(FreezeEnvelope())
cr_s, e_s = run_b(SmartFreeze(0.10, 0.3))
for lbl, cr, e in [("pole (brake a=-1)", cr_p, e_p),
                   ("freeze envelope (a=0)", cr_f, e_f),
                   ("smart freeze", cr_s, e_s)]:
    print(f"  {lbl:>24s}  crash {sum(cr.values()):>3d}/100  pooled err "
          f"{np.mean(list(e.values())):.2f}   per-item "
          + " ".join(f"{k}:{e[k]:.2f}" for k in ("B1", "B2", "B3", "B4", "B5")))

print()
print("=" * 100)
print("B5  HARD SUBSET (step-down) -- pole 0/220 claim, freeze variants")
print("=" * 100)
for lbl, mk in [("pole", lambda: pp.PositivePole()),
                ("freeze envelope (a=0)", lambda: FreezeEnvelope()),
                ("smart freeze dv>0.10", lambda: SmartFreeze(0.10, 0.3)),
                ("P+FF", lambda: p_ff)]:
    tot = 0
    row = ""
    for an, fn in HARD.items():
        c, e = 0, []
        for k in range(20):
            cc, ee = one_ep(mk(), fn, d=0.20, seed=2000 + k)
            c += cc
            e.append(ee)
        tot += c
        row += f"  {an}: {c:>2d}/20 (err {np.mean(e):.2f})"
    print(f"  {lbl:>22s} total {tot:>3d}/60" + row)
print("\nDONE")
