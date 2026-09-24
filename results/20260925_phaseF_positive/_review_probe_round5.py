"""Review probe round 5 (round 11) -- closing the remaining questions.

  U1  DEAD YIELD BRANCH: what is the maximum achievable |dv| per step?
      (YTHRESH=0.20 vs a_max*DT; with/without domain randomization)
  U2  TRUE closing rate (from -d(gap)/dt), not the command-minus-actual proxy
  U3  the (d=1.0, step-down 0.2s) stall: leader ACTUAL speed vs command
  U4  pole vs P+FF per-step action difference at nominal -- resolved
  U5  load-bearing branch identified by trace at step-down 0.1s
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
ACT = fe.ACT_GAIN
DT = fe.DT
print(f"a_max*DT = 1.5*{DT} = {1.5*DT} m/s per step "
      f"(DR range a_max in [1.0,2.0] -> [0.020,0.040])")
print(f"YTHRESH = {pp.YTHRESH}  ->  ratio {pp.YTHRESH/(1.5*DT):.1f}x the nominal step")


class Traced(srch.ScriptedLeader):
    def step(self, action):
        out = super().step(action)
        s_l = self.path.nearest(self.leader.pos)[0]
        s_f = self.path.nearest(self.follower.pos)[0]
        self._sl.append(s_l)
        self._sf.append(s_f)
        self._vl_act.append(self.leader.v)      # TRUE leader speed
        self._act.append(float(np.asarray(action).ravel()[0]))
        self._obs_hist.append(np.array(out[0], dtype=float))
        return out

    def reset(self, **kw):
        self._sl, self._sf, self._vl_act, self._act, self._obs_hist = [], [], [], [], []
        return super().reset(**kw)


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


def go(policy, fn, d_des, seed=2000, dr=False):
    fe.D_DES = d_des
    env = Traced(domain_randomize=dr)
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
        return dict(reason=env.term_reason, t=np.array(env.log["t"]), g=g,
                    vl_cmd=np.array(env.log["v_l"]),
                    vl=np.array(env._vl_act), vf=np.array(env.log["v_f"]),
                    act=np.array(env._act), obs=np.array(env._obs_hist))
    finally:
        fe.D_DES = 0.20


print()
print("=" * 96)
print("U1  MAXIMUM ACHIEVABLE |dv| per step -- is YTHRESH=0.20 reachable?")
print("=" * 96)
worst = 0.0
worst_where = None
for dr in (False, True):
    for an, fn in [("nominal", lambda t: 0.5),
                   ("step-down 0.1s", lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05),
                   ("full-range 0.05s",
                    lambda t: 1.0 if int(t / 0.05) % 2 == 0 else 0.05)]:
        r = go(pp.PositivePole(), fn, 0.20, dr=dr)
        dv = np.abs(np.diff(r["vf"]))
        mx = float(dv.max()) if len(dv) else 0.0
        if mx > worst:
            worst, worst_where = mx, (dr, an)
        print(f"  dr={dr!s:>5s} {an:>18s}  max|dv_follower| per step = {mx:.4f} m/s "
              f"(obs[2]-derived {np.abs(np.diff(r['obs'][:,2])).max()*1.3:.4f})")
print(f"  WORST over all: {worst:.4f} m/s at {worst_where}  "
      f"vs YTHRESH {pp.YTHRESH}  ->  branch is "
      f"{'REACHABLE' if worst > pp.YTHRESH else 'UNREACHABLE (dead code)'}")

print()
print("=" * 96)
print("U2  TRUE closing rate c = -d(gap)/dt, d=0.20")
print("=" * 96)
for an, fn in [("nominal", lambda t: 0.5),
               ("step-down 0.1s", lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05)]:
    print(f"  --- {an} ---")
    for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
        c = []
        for k in range(20):
            r = go(pol, fn, 0.20, seed=2000 + k)
            cc = -np.diff(r["g"]) / DT
            c.append(cc[int(len(cc) * 0.2):])
        c = np.concatenate(c)
        q = np.percentile(c, [50, 90, 99, 100])
        print(f"    {lbl:>5s}  mean {c.mean():+.3f}  p50 {q[0]:+.3f} p90 {q[1]:+.3f} "
              f"p99 {q[2]:+.3f} max {q[3]:+.3f} m/s")

print()
print("=" * 96)
print("U3  THE (d=1.0, step-down 0.2s) STALL: why does the gap never reach 1.0?")
print("=" * 96)
sd2 = lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
    r = go(pol, sd2, 1.00, seed=2000)
    n = len(r["g"])
    sl = slice(int(n * 0.2), n)
    print(f"  {lbl:>5s}  mean actual leader v {r['vl'][sl].mean():.3f}  "
          f"mean follower v {r['vf'][sl].mean():.3f}  "
          f"mean commander-mean {r['vl_cmd'][sl].mean():.3f}  "
          f"mean gap {r['g'][sl].mean():.3f}  gap0 {r['g'][0]:.3f}")
    print(f"        mean action {r['act'][sl].mean():+.3f}  "
          f"frac of steps with v_cmd clipped at 0 "
          f"{np.mean(r['vl_cmd'][sl] + ACT*r['act'][sl] <= 0.001):.2f}")
print("  CONTRAST: nominal leader at d=1.0")
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
    r = go(pol, lambda t: 0.5, 1.00, seed=2000)
    n = len(r["g"])
    sl = slice(int(n * 0.2), n)
    print(f"  {lbl:>5s}  mean leader v {r['vl'][sl].mean():.3f}  "
          f"mean follower v {r['vf'][sl].mean():.3f}  gap {r['g'][sl].mean():.3f}  "
          f"action {r['act'][sl].mean():+.3f}")
print("  d=1.0 with a 0.2 s square wave: leader's own actuator cannot track, so its")
print("  ACTUAL speed oscillates near 0.5; the follower's command is clipped at 0")
print("  during the low half, so its mean speed cannot get BELOW the leader's ->")
print("  the gap can never open.  Both policies are identical here by construction.")

print()
print("=" * 96)
print("U4  pole vs P+FF per-step action difference at nominal, d=0.20, seed 2000")
print("=" * 96)
a = go(pp.PositivePole(), lambda t: 0.5, 0.20, seed=2000)
b = go(p_ff, lambda t: 0.5, 0.20, seed=2000)
n = min(len(a["act"]), len(b["act"]))
d = a["act"][:n] - b["act"][:n]
big = np.abs(d) > 1e-6
print(f"  steps {n}  differing {big.sum()} ({100.0*big.sum()/n:.1f}%)  "
      f"mean|dact| {np.abs(d).mean():.4f}  max|dact| {np.abs(d).max():.4f}")
print(f"  differing steps are concentrated in the catch-up phase: "
      f"first differing step index {np.argmax(big)}, "
      f"frac of the first 200 steps differing "
      f"{big[:200].mean():.2f}, frac of the last 200 {big[-200:].mean():.2f}")
print(f"  gap at first difference {a['g'][np.argmax(big)]:.3f} m; "
      f"gap range in the first 200 steps "
      f"[{a['g'][:200].min():.3f}, {a['g'][:200].max():.3f}]")
# where does the clamp bind?
gap = a["g"]
e = gap - 0.20
room = np.maximum(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
cmax = pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room)
print(f"  clamp cap c_max/ACT: at gap 0.20 -> {cmax[0]/ACT:.3f}, "
      f"at gap 0.61 -> {pp.SAFETY*np.sqrt(2*pp.A_MAX_LO*(0.61-0.17))/ACT:.3f}")
nz = np.sum((0.8 * e / ACT) > (cmax / ACT))
print(f"  steps where the P demand exceeds the clamp: {nz}/{n}")

print()
print("=" * 96)
print("U5  LOAD-BEARING BRANCH at step-down 0.1s, d=0.20, seed 2000")
print("=" * 96)


class CountBoth(pp.PositivePole):
    def reset(self):
        super().reset()
        self.nb = self.ny = 0
        self.n = 0

    def __call__(self, obs):
        e = float(obs[0]) * 0.5
        de = float(obs[1]) * 2.0
        v = float(obs[2]) * 1.3
        c = -de
        if self.prev_v is not None and (self.prev_v - v) > pp.YTHRESH:
            self.ny += 1
        gap = e + fe.D_DES
        room = max(gap - fe.COLLISION_GAP - pp.MARGIN, 0.0)
        if c > pp.SAFETY * np.sqrt(2.0 * pp.A_MAX_LO * room):
            self.nb += 1
        self.n += 1
        return super().__call__(obs)


pol = CountBoth()
r = go(pol, lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05, 0.20, seed=2000)
print(f"  steps {pol.n}  brake branch {pol.nb} ({100.0*pol.nb/pol.n:.1f}%)  "
      f"yield branch {pol.ny} ({100.0*pol.ny/pol.n:.1f}%)")
print(f"  reason {r['reason']}  min gap {r['g'].min():.3f} "
      f"(collision at {fe.COLLISION_GAP})")
idx = np.linspace(0, len(r["g"]) - 1, 16).astype(int)
print("      t    gap    v_l_act   v_f    action")
for i in idx:
    print(f"   {r['t'][i]:6.2f} {r['g'][i]:6.3f} {r['vl'][i]:8.3f} "
          f"{r['vf'][i]:6.3f} {r['act'][i]:+8.3f}")
print("\nDONE")
