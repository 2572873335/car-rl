"""Review probe round 13 (Phase F positive review 3) -- CRITERION ATTACK.

CORE METHOD OF THIS ROUND: attack the criterion BEFORE judging the positive pole.
Exhaustively enumerate degenerate strategy families and evaluate each against the
six-clause conjunction H-A3'(a)-(f) on the judgement domain

    B*  =  B1..B5 (scripted families, reviews/20260925_phaseF_holdout_B.md)
           + B6 (hard step-down family: step-down 0.1 / 0.2 / 0.4 s, added by
                 review2 F2 as the ONLY separating family between the pole and P+FF)

Central question: with B6 in the domain, is there STILL a strategy that passes
the whole conjunction and contains NO anti-disturbance logic?
(review1's answer was "zero action"; review2's was "P+FF" -- this round's answer
is what this probe measures.)

Harness discipline (plan sec 2.3.1(iv)): EVERY policy is driven through the SAME
loop, and `policy.reset()` is called at the start of EVERY episode whenever the
policy exposes it -- including the stateless ones (given a no-op reset) and the
frozen v1 RL reference.  The positive pole is STATEFUL (prev_v); without the
per-episode reset it is fed a false dv at the episode boundary, which has been
measured to flip 0/20 into 20/20 (review2 sec 2.3.1(iv)).
"""
import sys
import time
import importlib.util

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

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


def CLIP(a, hi=1.0):
    return np.array([float(np.clip(a, -1.0, hi))], dtype=np.float32)


# ------------------------------------------------------------------ attacks ---
def attack_fn(kind, seed):
    """B-group attack families.  B1..B5 verbatim from _review2_probe_A.py;
    B6 = the hard step-down family, split into its three published cells."""
    r = np.random.default_rng(seed)
    if kind == "B1":
        per, duty = float(r.choice([0.3, 0.7])), float(r.uniform(0.3, 0.7))
        return lambda t, per=per, duty=duty: 0.95 if (t % per) < duty * per else 0.05
    if kind == "B2":
        f, a, ph = (float(r.choice([0.3, 0.8])), float(r.choice([0.3, 0.5])),
                    float(r.uniform(0, 2 * np.pi)))
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
    if kind == "B6-0.1":
        return lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05
    if kind == "B6-0.2":
        return lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
    if kind == "B6-0.4":
        return lambda t: 0.95 if int(t / 0.4) % 2 == 0 else 0.05
    raise KeyError(kind)


ITEMS = ("B1", "B2", "B3", "B4", "B5", "B6-0.1", "B6-0.2", "B6-0.4")
B15 = ("B1", "B2", "B3", "B4", "B5")
B6 = ("B6-0.1", "B6-0.2", "B6-0.4")


# ------------------------------------------------------------------ policies ---
class PFF:
    """Plain P + speed feedforward (v_cmd = v_l + kp*e).  No disturbance logic.
    kp=0.8 is the codebase's own baseline_action(use_ff=True)."""

    def __init__(self, kp=0.8):
        self.kp = kp

    def reset(self):
        pass

    def __call__(self, obs):
        return CLIP(self.kp * (obs[0] * 0.5) / ACT)


class PCap:
    """P+FF with the POSITIVE action capped at `apos` -> the commanded closing
    rate is bounded by 0.8*apos.  Still no disturbance logic: it never reacts to
    a disturbance, it is simply never aggressive.  Braking authority is intact."""

    def __init__(self, kp=0.8, apos=0.3):
        self.kp, self.apos = kp, apos

    def reset(self):
        pass

    def __call__(self, obs):
        return np.array([float(np.clip(self.kp * (obs[0] * 0.5) / ACT, -1.0,
                                       min(1.0, self.apos)))], dtype=np.float32)


class POnly:
    """Pure P on the speed prior (no feedforward): v_cmd = v_f + kp*e."""

    def __init__(self, kp=0.8):
        self.kp = kp

    def reset(self):
        pass

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        return CLIP((self.kp * e - de) / ACT)


class BigGap:
    """Aim for d_des + offset: the classic "keep a big gap" safety degenerate."""

    def __init__(self, offset=0.30, kp=0.8):
        self.off, self.kp = offset, kp

    def reset(self):
        pass

    def __call__(self, obs):
        return CLIP(self.kp * ((obs[0] * 0.5) - self.off) / ACT)


class Crawl:
    """Never exceed a fixed speed cap (const-cap controller)."""

    def __init__(self, cap=0.15):
        self.cap = cap

    def reset(self):
        pass

    def __call__(self, obs):
        return CLIP((self.cap - obs[2] * 1.3) / ACT)


class Zero:
    def reset(self):
        pass

    def __call__(self, obs):
        return np.zeros(1, dtype=np.float32)


class RandAct:
    def __init__(self, seed=12345):
        self.seed = seed

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, obs):
        return np.array([float(self.rng.uniform(-1, 1))], dtype=np.float32)


class FreezeGap:
    """Nominal P+FF; freeze (a=0 -> speed match) when the gap is small AND a
    closing rate has built up.  Gap-keyed so it does not fire during catch-up."""

    def __init__(self, g_thresh=0.40, c_thresh=0.20):
        self.gt, self.ct = g_thresh, c_thresh

    def reset(self):
        self.fires = self.steps = 0

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        self.steps += 1
        if (e + fe.D_DES) < self.gt and (-de) > self.ct:
            self.fires += 1
            return np.zeros(1, dtype=np.float32)
        return CLIP(0.8 * e / ACT)


class FreezeTime:
    """Time-keyed freeze: hold a=0 for `hold` s once a sharp closing rate is seen."""

    def __init__(self, c_thresh=0.35, hold=0.4):
        self.ct, self.hold = c_thresh, hold

    def reset(self):
        self.t_left = 0.0
        self.fires = self.steps = 0

    def __call__(self, obs):
        de = obs[1] * 2.0
        self.steps += 1
        if -de > self.ct:
            self.t_left = self.hold
        if self.t_left > 0.0:
            self.t_left -= fe.DT
            self.fires += 1
            return np.zeros(1, dtype=np.float32)
        return CLIP(0.8 * (obs[0] * 0.5) / ACT)


class CatchThenFreeze:
    """P+FF until the error has converged, then a == 0 forever.  The V2V prior
    then holds the gap whatever the leader does.  NO braking logic whatsoever:
    the policy has no notion of a disturbance, it just stops once it is happy.

    latch: |e| < eps            (rate-blind)
    latch2: |e| < eps and |de| < dc   (needs a genuine settle, still no
                                      disturbance logic -- de is its own state)
    """

    def __init__(self, eps=0.02, dc=None):
        self.eps, self.dc = eps, dc

    def reset(self):
        self.frozen = False
        self.n_frozen = self.n = 0

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        self.n += 1
        if abs(e) < self.eps and (self.dc is None or abs(de) < self.dc):
            self.frozen = True
        if self.frozen:
            self.n_frozen += 1
            return np.zeros(1, dtype=np.float32)
        return CLIP(0.8 * e / ACT)


class CatchThenFreezeGap:
    """P+FF until the gap first drops below `g`, then a == 0 forever."""

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
        return CLIP(0.8 * e / ACT)


class CapThenFreeze:
    """Bounded-closing catch-up (PCap) that latches to a=0 once converged.
    Combines the two ideas above and is, on paper, the sharpest no-logic
    candidate: converge gently (so nothing has to be shed), then stop."""

    def __init__(self, apos=0.3, eps=0.03, dc=None):
        self.apos, self.eps, self.dc = apos, eps, dc

    def reset(self):
        self.frozen = False
        self.n_frozen = self.n = 0

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        self.n += 1
        if abs(e) < self.eps and (self.dc is None or abs(de) < self.dc):
            self.frozen = True
        if self.frozen:
            self.n_frozen += 1
            return np.zeros(1, dtype=np.float32)
        return np.array([float(np.clip(0.8 * e / ACT, -1.0,
                                       min(1.0, self.apos)))], dtype=np.float32)


class ClampOnly:
    """Pole ablation 1: the envelope applied as a COMMAND clamp only."""

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
    """Pole ablation 2: the envelope with the full-authority brake, no clamp."""

    def reset(self):
        pass

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        cm = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        if -de > cm:
            return np.array([-1.0], dtype=np.float32)
        return CLIP(pp.KP * e / ACT)


class FreezeEnvelope:
    """Pole with the full brake replaced by a=0 (brake authority given up)."""

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
        return CLIP(pp.KP * e / ACT)


class ThreshBrake:
    """One-line alternative: P+FF plus a= -1 whenever gap < g (fixed threshold,
    not an envelope).  This IS a crude anti-disturbance rule, included as the
    'obvious one-liner' control."""

    def __init__(self, g=0.25):
        self.g = g

    def reset(self):
        self.fires = self.steps = 0

    def __call__(self, obs):
        e = obs[0] * 0.5
        self.steps += 1
        if e + fe.D_DES < self.g:
            self.fires += 1
            return np.array([-1.0], dtype=np.float32)
        return CLIP(0.8 * e / ACT)


class Pole:
    def __init__(self):
        self.p = pp.PositivePole()

    def reset(self):
        self.p.reset()

    def __call__(self, obs):
        return self.p(obs)


def P(nm, f, logic, note=""):
    return dict(name=nm, factory=f, logic=logic, note=note)


# logic: "NONE" = no anti-disturbance logic at all (pure tracking / freeze / crawl)
#        "YES"  = contains an explicit reaction to a disturbance
ZOO = [
    P("pole (anchor)", lambda: Pole(), "YES", "envelope full brake, 2.4-8.1% of steps"),
    P("P+FF kp=0.8", lambda: PFF(0.8), "NONE", "the codebase baseline_action(use_ff=True)"),
    P("zero action", lambda: Zero(), "NONE", "v_cmd = v_l exactly (V2V prior)"),
    P("random action", lambda: RandAct(12345), "NONE", "10.5c random-perturbation class"),

    P("P+FF kp=0.15", lambda: PFF(0.15), "NONE", ""),
    P("P+FF kp=0.25", lambda: PFF(0.25), "NONE", ""),
    P("P+FF kp=0.40", lambda: PFF(0.40), "NONE", ""),
    P("P+FF kp=0.60", lambda: PFF(0.60), "NONE", ""),
    P("P+FF kp=1.20", lambda: PFF(1.20), "NONE", ""),
    P("P+FF kp=2.00", lambda: PFF(2.00), "NONE", ""),
    P("P only kp=0.8", lambda: POnly(0.8), "NONE", "no speed feedforward"),

    P("Pcap a+<=0.05", lambda: PCap(0.8, 0.05), "NONE", "closing rate <= 0.04 m/s"),
    P("Pcap a+<=0.10", lambda: PCap(0.8, 0.10), "NONE", "closing rate <= 0.08 m/s"),
    P("Pcap a+<=0.15", lambda: PCap(0.8, 0.15), "NONE", "closing rate <= 0.12 m/s"),
    P("Pcap a+<=0.20", lambda: PCap(0.8, 0.20), "NONE", "closing rate <= 0.16 m/s"),
    P("Pcap a+<=0.30", lambda: PCap(0.8, 0.30), "NONE", "closing rate <= 0.24 m/s"),
    P("Pcap a+<=0.50", lambda: PCap(0.8, 0.50), "NONE", "closing rate <= 0.40 m/s"),

    P("biggapplus 0.05", lambda: BigGap(0.05), "NONE", ""),
    P("biggapplus 0.10", lambda: BigGap(0.10), "NONE", ""),
    P("biggapplus 0.20", lambda: BigGap(0.20), "NONE", ""),
    P("biggapplus 0.30", lambda: BigGap(0.30), "NONE", ""),
    P("biggapplus 0.50", lambda: BigGap(0.50), "NONE", ""),
    P("crawl cap=0.15", lambda: Crawl(0.15), "NONE", "const speed cap"),

    P("freezeGap g<.25 c>.05", lambda: FreezeGap(0.25, 0.05), "NONE", ""),
    P("freezeGap g<.25 c>.20", lambda: FreezeGap(0.25, 0.20), "NONE", ""),
    P("freezeGap g<.35 c>.10", lambda: FreezeGap(0.35, 0.10), "NONE", ""),
    P("freezeGap g<.35 c>.20", lambda: FreezeGap(0.35, 0.20), "NONE", ""),
    P("freezeGap g<.45 c>.10", lambda: FreezeGap(0.45, 0.10), "NONE", ""),
    P("freezeGap g<.45 c>.20", lambda: FreezeGap(0.45, 0.20), "NONE", ""),
    P("freezeGap g<.60 c>.20", lambda: FreezeGap(0.60, 0.20), "NONE", ""),
    P("freezeTime c>.20 hold.2", lambda: FreezeTime(0.20, 0.2), "NONE", ""),
    P("freezeTime c>.35 hold.4", lambda: FreezeTime(0.35, 0.4), "NONE", ""),
    P("freezeTime c>.50 hold.8", lambda: FreezeTime(0.50, 0.8), "NONE", ""),

    P("catchFreeze |e|<.02", lambda: CatchThenFreeze(0.02), "NONE", ""),
    P("catchFreeze |e|<.05", lambda: CatchThenFreeze(0.05), "NONE", ""),
    P("catchFreeze |e|<.10", lambda: CatchThenFreeze(0.10), "NONE", ""),
    P("catchFreeze |e|<.02 &|de|<.02", lambda: CatchThenFreeze(0.02, 0.02), "NONE", ""),
    P("catchFreeze |e|<.05 &|de|<.05", lambda: CatchThenFreeze(0.05, 0.05), "NONE", ""),
    P("catchFreeze gap<.22", lambda: CatchThenFreezeGap(0.22), "NONE", ""),
    P("catchFreeze gap<.25", lambda: CatchThenFreezeGap(0.25), "NONE", ""),
    P("catchFreeze gap<.30", lambda: CatchThenFreezeGap(0.30), "NONE", ""),

    P("capFreeze a+<=0.10 |e|<.03", lambda: CapThenFreeze(0.10, 0.03), "NONE", ""),
    P("capFreeze a+<=0.20 |e|<.03", lambda: CapThenFreeze(0.20, 0.03), "NONE", ""),
    P("capFreeze a+<=0.30 |e|<.03", lambda: CapThenFreeze(0.30, 0.03), "NONE", ""),
    P("capFreeze a+<=0.30 |e|<.05", lambda: CapThenFreeze(0.30, 0.05), "NONE", ""),
    P("capFreeze a+<=0.50 |e|<.03", lambda: CapThenFreeze(0.50, 0.03), "NONE", ""),

    P("clamp only (pole abl.)", lambda: ClampOnly(), "NO(broken)", "review1: 60/220"),
    P("brake only (pole abl.)", lambda: BrakeOnly(), "YES", "review1: 0/220"),
    P("freeze-envelope", lambda: FreezeEnvelope(), "YES(brake->0)", "review2 leak"),
    P("threshBrake g<0.20", lambda: ThreshBrake(0.20), "YES", "one-line alternative"),
    P("threshBrake g<0.25", lambda: ThreshBrake(0.25), "YES", "one-line alternative"),
    P("threshBrake g<0.30", lambda: ThreshBrake(0.30), "YES", "one-line alternative"),
]


# ------------------------------------------------------------------ harness ---
def evaluate(factory, items=ITEMS, n_ep=10, env0=60000, atk0=50000, d=0.20):
    """One harness for everything.  policy.reset() is called before EVERY episode
    when the policy exposes it (stateless policies get a no-op)."""
    fe.D_DES = d
    crash, err, mgap, term = {}, {}, {}, {}
    try:
        for kind in items:
            env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
            c, es, gs, tr = 0, [], [], {}
            for k in range(n_ep):
                pol = factory()
                if hasattr(pol, "reset"):
                    pol.reset()
                obs, _ = env.reset(seed=env0 + k)
                env.set_fn(attack_fn(kind, atk0 + k))
                while True:
                    obs, r, term_, trunc, _ = env.step(pol(obs))
                    if term_ or trunc:
                        break
                g = np.array(env.log["gap"])
                s = g[int(len(g) * 0.2):]
                es.append(float(np.abs(s - d).mean() * 200))
                gs.append(float(s.mean()))
                c += int(env.term_reason == "collision")
                tr[env.term_reason] = tr.get(env.term_reason, 0) + 1
            crash[kind], err[kind] = c, float(np.mean(es))
            mgap[kind], term[kind] = float(np.mean(gs)), tr
    finally:
        fe.D_DES = 0.20
    return crash, err, mgap, term


def evaluate_nominal(factory, n_ep=20, seed0=2000, d=0.20):
    fe.D_DES = d
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    try:
        es, c = [], 0
        for k in range(n_ep):
            pol = factory()
            if hasattr(pol, "reset"):
                pol.reset()
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(lambda t: 0.5)
            while True:
                obs, r, term, trunc, _ = env.step(pol(obs))
                if term or trunc:
                    break
            g = np.array(env.log["gap"])
            s = g[int(len(g) * 0.2):]
            es.append(float(np.abs(s - d).mean() * 200))
            c += int(env.term_reason == "collision")
        return c, float(np.mean(es))
    finally:
        fe.D_DES = 0.20


def pooled(err, items):
    return float(np.mean([err[k] for k in items]))
