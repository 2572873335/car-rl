"""Review probe round 4 (round 11) -- resolving two anomalies in round 3.

  T1  why were S10's per-policy 'err' columns IDENTICAL to 2 dp?
  T2  is the (d=1.0, step-down 0.2s) settled err=78.28 a PATH-WRAP artifact?
      (_measure returns (s_l - s_f) % path.length -- a lapped follower and a
       follower 0.6 m behind report the SAME gap)
  T3  v1 RL B-group crash rate: seed and N sensitivity (the H-A3 baseline)
  T4  closing-rate distribution: pole vs P+FF vs v1 (item A)
  T5  does the pole reproduce P+FF whenever the envelope is slack?
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
LAP = fe._paths().length
print(f"lap length = {LAP:.4f} m")

from stable_baselines3 import PPO
model = PPO.load(REPO + "/ckpt/follow_stage2_final_v1.zip")


def v1_policy(obs):
    a, _ = model.predict(obs, deterministic=True)
    return np.asarray(a, dtype=np.float32)


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


class Traced(ScriptedLeader):
    """Records BOTH the wrapped gap (env's own definition) and the UNWRAPPED
    longitudinal separation, plus the actuation."""

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        s_l = self.path.nearest(self.leader.pos)[0]
        s_f = self.path.nearest(self.follower.pos)[0]
        raw = (s_l - s_f) % self.path.length
        self._raw.append(raw)
        self._act.append(float(np.asarray(action).ravel()[0]))
        self._cmd.append(self.log["v_l"][-1] + ACT * float(np.asarray(action).ravel()[0]))
        # unwrapped: integrate (v_l_cmd - v_f) is wrong (actuator lag); instead
        # track cumulative arc-length difference properly
        self._sl.append(s_l)
        self._sf.append(s_f)
        return obs, r, term, trunc, info

    def reset(self, **kw):
        self._raw, self._act, self._cmd, self._sl, self._sf = [], [], [], [], []
        return super().reset(**kw)


def unwrap(sl, sf, L):
    """Unwrapped signed separation leader-minus-follower, in metres."""
    out = []
    for i in range(len(sl)):
        if i == 0:
            out.append((sl[0] - sf[0]) % L)
        else:
            d = (sl[i] - sf[i]) - out[-1]
            d = (d + L / 2) % L - L / 2          # nearest representative
            out.append(out[-1] + d)
    return np.array(out)


def trace_run(policy, fn, d_des, seed, n_steps=None):
    fe.D_DES = d_des
    env = Traced(domain_randomize=False)
    try:
        obs, _ = env.reset(seed=seed)
        env.set_fn(fn)
        while True:
            obs, r, term, trunc, _ = env.step(policy(obs))
            if term or trunc:
                break
        return dict(reason=env.term_reason, t=np.array(env.log["t"]),
                    g=np.array(env.log["gap"]), vl=np.array(env.log["v_l"]),
                    vf=np.array(env.log["v_f"]), raw=np.array(env._raw),
                    act=np.array(env._act), cmd=np.array(env._cmd),
                    unw=unwrap(env._sl, env._sf, LAP))
    finally:
        fe.D_DES = 0.20


print()
print("=" * 96)
print("T1  WHY identical 'err' columns?  B3 chirp, k=0, seed 60000, d=0.20")
print("=" * 96)
b3 = lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * (0.1 + 0.125 * min(t, 4.0)) * t)
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff), ("v1 RL", v1_policy)]:
    r = trace_run(pol, b3, 0.20, 60000)
    s = r["g"][int(len(r["g"]) * 0.2):]
    print(f"  {lbl:>6s}  reason {r['reason']:>10s}  steps {len(r['g']):>5d}  "
          f"t_end {r['t'][-1]:5.2f}  min_gap {r['g'].min():.3f}  "
          f"mean_gap(20pct+) {s.mean():.3f}  err {np.abs(s-0.20).mean()*200:6.2f}  "
          f"unw[0,-1] {r['unw'][0]:.2f}->{r['unw'][-1]:.2f}")

print()
print("=" * 96)
print("T2  IS (d=1.0, step-down 0.2s) err=78.28 A WRAP ARTIFACT?")
print("=" * 96)
sd2 = lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
    r = trace_run(pol, sd2, 1.00, 2000)
    s = r["g"][int(len(r["g"]) * 0.2):]
    su = r["unw"][int(len(r["unw"]) * 0.2):]
    print(f"  {lbl:>5s}  wrapped gap: mean {s.mean():.3f}  "
          f"min {s.min():.3f}  max {s.max():.3f}   -> err {np.abs(s-1.0).mean()*200:6.2f}")
    print(f"        UNWRAPPED separation: mean {su.mean():.3f}  "
          f"min {su.min():.3f}  max {su.max():.3f}   "
          f"-> drift {(su[-1]-su[0]):+.3f} m over {r['t'][-1]-r['t'][0]:.1f} s")
    print(f"        wrapped-gap swing within episode: {s.max()-s.min():.3f} m "
          f"(a genuine tracker would hold it small)")
r = trace_run(pp.PositivePole(), sd2, 1.00, 2000)
print("  pole trace, unwrapped separation (m) and wrapped gap (m):")
idx = np.linspace(0, len(r["g"]) - 1, 15).astype(int)
print("      t   wrapped   unwrapped")
for i in idx:
    print(f"   {r['t'][i]:6.2f}   {r['g'][i]:7.3f}   {r['unw'][i]:9.3f}")
# nominal d=1.0 for contrast
r = trace_run(pp.PositivePole(), lambda t: 0.5, 1.00, 2000)
s = r["g"][int(len(r["g"]) * 0.2):]
su = r["unw"][int(len(r["unw"]) * 0.2):]
print(f"  CONTROL nominal d=1.0: wrapped mean {s.mean():.3f}  "
      f"unwrapped mean {su.mean():.3f}  err {np.abs(s-1.0).mean()*200:.2f}")

print()
print("=" * 96)
print("T3  v1 RL B-group crash rate -- seed and N sensitivity (H-A3 baseline)")
print("=" * 96)


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


def crash_count(policy, fns, seed0):
    tot = 0
    for k, fn in enumerate(fns):
        fe.D_DES = 0.20
        env = ScriptedLeader(domain_randomize=False)
        try:
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            tot += int(env.term_reason == "collision")
        finally:
            fe.D_DES = 0.20
    return tot


for cs in (50000, 77000):
    print(f"  attack seeds {cs}..{cs+19}")
    for lbl, pol in [("v1 RL", v1_policy), ("P+FF", p_ff),
                     ("pole", pp.PositivePole())]:
        fns = [b_fn(kind, cs + j) for kind in ("B1", "B2", "B3", "B4", "B5")
               for j in range(20)]
        c = crash_count(pol, fns, 60000)
        print(f"    {lbl:>6s}  {c:>3d}/100")
print("  (env seeds held at 60000+k throughout, matching S10)")

print()
print("=" * 96)
print("T4  CLOSING-RATE and CONTROL distribution: pole vs P+FF vs v1 (item A)")
print("=" * 96)


def stats(policy, fn, d_des, n_ep=20, seed0=2000):
    fe.D_DES = d_des
    env = Traced(domain_randomize=False)
    try:
        cr, cm = [], []
        for k in range(n_ep):
            if hasattr(policy, "reset"):
                policy.reset()
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            vl = np.array(env.log["v_l"])
            vf = np.array(env.log["v_f"])
            cl = np.maximum(vl - vf, 0)
            cr.append(cl)
            cm.append(cl.mean())
        return np.concatenate(cr), float(np.mean(cm))
    finally:
        fe.D_DES = 0.20


for an in ("nominal", "step-down 0.1s"):
    fn = (lambda t: 0.5) if an == "nominal" else (
        lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05)
    print(f"  --- {an} (d=0.20) ---")
    for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff), ("v1 RL", v1_policy)]:
        c, mc = stats(pol, fn, 0.20)
        q = np.percentile(c, [50, 90, 99, 100])
        print(f"    {lbl:>6s}  mean closing {mc:+.3f} m/s | p50 {q[0]:.3f} "
              f"p90 {q[1]:.3f} p99 {q[2]:.3f} max {q[3]:.3f}")
print()
print("  T4b NOMINAL precision (the H-A4 gate) -- pole vs P+FF, d=0.20:")
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
    fe.D_DES = 0.20
    env = ScriptedLeader(domain_randomize=False)
    try:
        errs = []
        for k in range(20):
            pol.reset() if hasattr(pol, "reset") else None
            obs, _ = env.reset(seed=2000 + k)
            env.set_fn(lambda t: 0.5)
            while True:
                obs, r, term, trunc, _ = env.step(pol(obs))
                if term or trunc:
                    break
            g = np.array(env.log["gap"])[int(len(env.log["gap"]) * 0.2):]
            errs.append(np.abs(g - 0.20).mean() * 200)
        print(f"    {lbl:>6s}  nominal err {np.mean(errs):.2f} obs-cm "
              f"(max {np.max(errs):.2f})")
    finally:
        fe.D_DES = 0.20

print()
print("=" * 96)
print("T5  IS THE POLE JUST P+FF WHEN THE ENVELOPE IS SLACK?")
print("=" * 96)
for an, fn in [("nominal", lambda t: 0.5),
               ("step-down 0.8s", lambda t: 0.95 if int(t / 0.8) % 2 == 0 else 0.05),
               ("crawl", lambda t: 0.10)]:
    a = trace_run(pp.PositivePole(), fn, 0.20, 2000)
    b = trace_run(p_ff, fn, 0.20, 2000)
    n = min(len(a["act"]), len(b["act"]))
    same = int(np.sum(np.abs(a["act"][:n] - b["act"][:n]) > 1e-6))
    print(f"  {an:>16s}  steps {n:>5d}  differing action steps {same:>5d} "
          f"({100.0*same/n:5.1f}%)  mean|dact| "
          f"{np.mean(np.abs(a['act'][:n]-b['act'][:n])):.4f}  "
          f"pole err {np.abs(a['g'][int(n*0.2):]-0.2).mean()*200:6.2f} "
          f"P+FF err {np.abs(b['g'][int(n*0.2):]-0.2).mean()*200:6.2f}")

print("\nDONE")
