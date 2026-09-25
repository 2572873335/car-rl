"""Review probe round 12, part F -- confirm the load-bearing claim across seeds.

The claim to confirm: on the B group, the "envelope rule with the full-authority
brake replaced by a=0" (a policy with NO braking ability at all) has a pooled
settled error BETTER than the pole's, and passes (b) (<=10% crash).  If that
holds across attack-seed blocks, then H-A3'(e) ranks the no-brake degenerate
ABOVE its own anchor -- an inversion of the kind 10.5c names.
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


class FreezeEnvelope:
    def reset(self):
        pass

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        cm = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
        if -de > cm:
            return np.zeros(1, dtype=np.float32)
        return np.array([float(np.clip(pp.KP * e / ACT, -1.0, 1.0))],
                        dtype=np.float32)


class BrakeOnlyEnvelope:
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


def run_b(policy, seed0, env0=60000, n_ep=20, d=0.20):
    fe.D_DES = d
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    try:
        crash, err, per = {}, {}, {}
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
            per[kind] = (c, float(np.mean(es)))
        return crash, err
    finally:
        fe.D_DES = 0.20


POL = [("pole (a=-1 brake)", lambda: pp.PositivePole()),
       ("freeze-envelope (a=0)", lambda: FreezeEnvelope()),
       ("brakeonly-envelope", lambda: BrakeOnlyEnvelope()),
       ("P+FF", lambda: p_ff)]

print("=" * 100)
print("F  RANKING INVERSION CHECK -- three attack-seed blocks, B group")
print("=" * 100)
for s0 in (50000, 77000, 91000):
    print(f"\n  attack seeds {s0}+")
    print(f"    {'policy':>22s} {'crash':>8s} {'pooled err':>11s}  per-item err")
    for lbl, mk in POL:
        crash, err = run_b(mk(), s0)
        tot = sum(crash.values())
        pooled = float(np.mean(list(err.values())))
        print(f"    {lbl:>22s} {str(tot)+'/100':>8s} {pooled:>11.2f}  "
              + " ".join(f"{k}:{err[k]:.2f}" for k in ("B1", "B2", "B3", "B4", "B5")))
print()
print("  pole's own B-group pooled err on these blocks is the correct anchor for (e).")
print("  If freeze-envelope's pooled err is <= the pole's on all three blocks, then")
print("  (e) -- correctly anchored -- prefers the no-brake degenerate.  RANKING INVERTED.")
print("\nDONE")
