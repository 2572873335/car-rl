"""Review probe round 3 (round 11).

  S9  TERMINATION AUDIT for the pole (all five classes, not just collisions)
  S10 the DECISIVE number for H-A3: the frozen v1 RL defender's B-group crash rate
  S11 does the acceptance discriminate?  nominal cost of the simple alternative
  S12 domain randomization ON (the final validation used domain_randomize=False)
  S13 seed sensitivity: hard attacks, more episodes
  S14 why the pole stalls at ~0.6 m in (d=1.0, step-down 0.2s)
  S15 extra attack classes beyond the 11-family
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


def run3(policy, fn, d_des, n_ep=20, seed0=2000, gap_max=1.2, dr=False):
    fe.D_DES = d_des
    env = ScriptedLeader(domain_randomize=dr, gap_max=gap_max)
    recs = []
    try:
        reasons = {}
        for k in range(n_ep):
            if hasattr(policy, "reset"):
                policy.reset()
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            reasons[env.term_reason] = reasons.get(env.term_reason, 0) + 1
            g = np.array(env.log["gap"])
            s = g[int(len(g) * 0.2):]
            recs.append(dict(coll=env.term_reason == "collision",
                             reason=env.term_reason,
                             err=float(np.abs(s - d_des).mean() * 200),
                             mean_gap=float(s.mean()),
                             min_gap=float(g.min()), gap0=env.gap0,
                             g=g, t=np.array(env.log["t"]),
                             vl=np.array(env.log["v_l"]),
                             vf=np.array(env.log["v_f"])))
        return reasons, recs
    finally:
        fe.D_DES = 0.20


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


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
        segs = []
        acc = 0.0
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
print("S9  TERMINATION AUDIT (all five classes) -- pole + P+FF, d=0.20")
print("=" * 96)
print(f"  {'attack':>16s}  {'pole term reasons':>46s}   {'P+FF':>20s}")
for an, fn in PROBES.items():
    rp, _ = run3(pp.PositivePole(), fn, 0.20, n_ep=20)
    rf, _ = run3(p_ff, fn, 0.20, n_ep=20)
    print(f"  {an:>16s}  {str(rp):>46s}   {str(rf):>46s}")

print()
print("=" * 96)
print("S10 ★ DECISIVE for H-A3: frozen v1 RL defender's crash rate on the B group")
print("=" * 96)
from stable_baselines3 import PPO
import os
V1 = REPO + "/ckpt/follow_stage2_final_v1.zip"
print(f"  ckpt {V1}  exists={os.path.exists(V1)}")
model = PPO.load(V1)


def v1_policy(obs):
    a, _ = model.predict(obs, deterministic=True)
    return np.asarray(a, dtype=np.float32)


tot_v1 = tot_p = 0
print(f"  {'item':>6s}  {'v1 RL crash':>13s} {'v1 err':>9s}  {'pole crash':>11s} "
      f"{'pole err':>9s}  {'P+FF crash':>11s} {'P+FF err':>9s}")
for kind in ("B1", "B2", "B3", "B4", "B5"):
    v1c = pc = fc = 0
    v1e = pe = fee = []
    for k in range(20):
        fn = b_fn(kind, 50000 + k)
        r, recs = run3(v1_policy, fn, 0.20, n_ep=1, seed0=60000 + k)
        v1c += sum(1 for x in recs if x["coll"])
        v1e.append(recs[0]["err"])
        r, recs = run3(pp.PositivePole(), fn, 0.20, n_ep=1, seed0=60000 + k)
        pc += sum(1 for x in recs if x["coll"])
        pe.append(recs[0]["err"])
        r, recs = run3(p_ff, fn, 0.20, n_ep=1, seed0=60000 + k)
        fc += sum(1 for x in recs if x["coll"])
        fee.append(recs[0]["err"])
    tot_v1 += v1c
    tot_p += pc
    print(f"  {kind:>6s}  {v1c:>10d}/20  {np.mean(v1e):>7.2f}  {pc:>8d}/20 "
          f"{np.mean(pe):>7.2f}  {fc:>8d}/20 {np.mean(fee):>7.2f}")
print(f"  {'TOTAL':>6s}  {tot_v1:>10d}/100 {'':>9s}  {tot_p:>8d}/100 {'':>9s}")
print(f"\n  H-A3 threshold is a RELATIVE reduction >=50% vs the UNTRAINED defender.")
print(f"  If baseline = v1 RL : need <= {0.5*tot_v1:.1f}/100  (from {tot_v1}/100)")
print(f"  If baseline = P+FF  : need <= 1.5/100  (from 3/100)")
print(f"  If baseline = pole  : need <= 0.0/100  (from 0/100) -- UNSATISFIABLE")

print()
print("=" * 96)
print("S11 does the acceptance DISCRIMINATE?  nominal cost of simple alternatives")
print("=" * 96)


class HardGapBrake(pp.PositivePole):
    def __init__(self, x):
        self.x = x
        super().__init__()

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        if e + fe.D_DES < self.x:
            return np.array([-1.0], dtype=np.float32)
        return np.array([float(np.clip(pp.KP * e / ACT, -1.0, 1.0))],
                        dtype=np.float32)


def make_const_cap(cap):
    def f(obs):
        e = obs[0] * 0.5
        return np.array([np.clip(0.8 * e / ACT, -1.0, cap / ACT)], dtype=np.float32)
    return f


print(f"  {'policy':>28s}  {'hard-subset coll':>16s}  {'nominal err':>11s}  "
      f"{'t_catch':>8s}  {'peak closing':>12s}")
for name, pol in [("pole (final)", pp.PositivePole()),
                  ("P+FF + hard brake g<0.30", HardGapBrake(0.30)),
                  ("P+FF + hard brake g<0.25", HardGapBrake(0.25)),
                  ("const-cap 0.30", make_const_cap(0.30)),
                  ("P+FF (ref)", p_ff)]:
    tot = 0
    for an in ("step-down 0.1s", "step-down 0.2s", "step-down 0.4s"):
        r, _ = run3(pol, PROBES[an], 0.20, n_ep=20)
        tot += sum(1 for x in _ if x["coll"])
    r, recs = run3(pol, PROBES["nominal"], 0.20, n_ep=20)
    err = np.mean([x["err"] for x in recs])
    pk = np.mean([float(np.max(np.append(-np.diff(x["g"]) / fe.DT, 0))) for x in recs])
    print(f"  {name:>28s}  {tot:>13d}/60  {err:>9.2f}  {'':>8s}  {pk:>10.2f}")

print()
print("=" * 96)
print("S12 DOMAIN RANDOMIZATION ON (final validation used dr=False)")
print("=" * 96)
for dr in (False, True):
    tot = 0
    for an, fn in PROBES.items():
        r, _ = run3(pp.PositivePole(), fn, 0.20, n_ep=20, dr=dr)
        tot += sum(1 for x in _ if x["coll"])
    print(f"  domain_randomize={dr!s:>5s}  pole collisions over 11 families: {tot}/220")

print()
print("=" * 96)
print("S13 SEED SENSITIVITY: hard attacks, 60 episodes, two seed blocks")
print("=" * 96)
for an in ("step-down 0.1s", "step-down 0.2s", "step-down 0.4s"):
    for s0 in (2000, 90000):
        r, recs = run3(pp.PositivePole(), PROBES[an], 0.20, n_ep=60, seed0=s0)
        c = sum(1 for x in recs if x["coll"])
        print(f"  {an:>16s}  seed0={s0:>6d}  coll {c:>2d}/60  "
              f"mean settled err {np.mean([x['err'] for x in recs]):6.2f}")

print()
print("=" * 96)
print("S14 WHY the pole stalls at ~0.6 m in (d=1.0, step-down 0.2s)")
print("=" * 96)
sd2 = lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
    r, recs = run3(pol, sd2, 1.00, n_ep=20)
    print(f"  {lbl:>5s}  mean settled gap {np.mean([x['mean_gap'] for x in recs]):.3f} "
          f"(target 1.00)   mean err {np.mean([x['err'] for x in recs]):.2f}")
# nominal at d=1.0 for both
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
    r, recs = run3(pol, PROBES["nominal"], 1.00, n_ep=20)
    print(f"  {lbl:>5s}  NOMINAL d=1.0: mean settled gap "
          f"{np.mean([x['mean_gap'] for x in recs]):.3f}   err "
          f"{np.mean([x['err'] for x in recs]):.2f}")
x = recs[0]
print("  P+FF trace at d=1.0, step-down 0.2s (episode 0):")
r, recs = run3(p_ff, sd2, 1.00, n_ep=1, seed0=2000)
x = recs[0]
print(f"    gap0 {x['gap0']:.3f}  final gap {x['g'][-1]:.3f}  "
      f"min {x['min_gap']:.3f}")
idx = np.linspace(0, len(x["g"]) - 1, 12).astype(int)
print("      t     gap    v_l    v_f")
for i in idx:
    print(f"   {x['t'][i]:6.2f} {x['g'][i]:7.3f} {x['vl'][i]:6.3f} {x['vf'][i]:6.3f}")

print()
print("=" * 96)
print("S15 EXTRA attack classes beyond the 11-family, d=0.20")
print("=" * 96)
EXTRA = {
    "square 0.05s (20Hz)": lambda t: 0.95 if int(t / 0.05) % 2 == 0 else 0.05,
    "step 1.0->0.05 once": lambda t: 1.0 if t < 2.0 else 0.05,
    "staircase down 5x": lambda t: max(0.05, 1.0 - 0.2 * int(t / 1.0)),
    "random phase 0.5Hz": None,
    "big osc 0.5+-0.45 1.2Hz": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 1.2 * t),
    "full-range hold 0.05/1.0 0.1s": lambda t: 1.0 if int(t / 0.1) % 2 == 0 else 0.05,
}
for an, fn in EXTRA.items():
    if fn is None:
        fns = []
        for k in range(20):
            rr = np.random.default_rng(80000 + k)
            ph = rr.uniform(0, 2 * np.pi)
            fns.append(lambda t, ph=ph: 0.5 + 0.45 * np.sin(np.pi * t + ph))
        res = {}
        for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
            tot = sum(sum(1 for y in run3(pol, fns[k], 0.20, n_ep=1,
                                          seed0=95000 + k)[1] if y["coll"])
                      for k in range(20))
            res[lbl] = tot
        print(f"  {an:>30s}  pole {res['pole']:>2d}/20   P+FF {res['P+FF']:>2d}/20")
        continue
    res = {}
    for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
        r, recs = run3(pol, fn, 0.20, n_ep=20)
        res[lbl] = sum(1 for y in recs if y["coll"])
    print(f"  {an:>30s}  pole {res['pole']:>2d}/20   P+FF {res['P+FF']:>2d}/20")

print("\nDONE")
