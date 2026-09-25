"""Review probe round 12, part E -- number traceability.

  E1  v1 RL B-group crash rate under the CONTINUOUS env-seed convention
      (round4 T3 style) -- does it reproduce the published 36/100 and 40/100?
  E2  d=1.0 / step-down 0.2s: pole vs P+FF 20-episode mean settled error --
      the plan says "P+FF 给出逐位相同的 78.65" against the pole's 78.28.
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


def p_ff(obs):
    return np.array([np.clip(fe.baseline_action(obs, use_ff=True)[0], -1, 1)],
                    dtype=np.float32)


V1 = REPO + "/ckpt/follow_stage2_final_v1.zip"
model = PPO.load(V1)


def v1_policy(obs):
    a, _ = model.predict(obs, deterministic=True)
    return np.asarray(a, dtype=np.float32)


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


print("=" * 96)
print("E1  v1 RL on B -- CONTINUOUS env seeds 60000..60099 (round4 T3 convention)")
print("=" * 96)
for s0 in (50000, 77000):
    fe.D_DES = 0.20
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    tot, cnt = 0, 0
    per = {}
    try:
        for kind in ("B1", "B2", "B3", "B4", "B5"):
            c = 0
            for k in range(20):
                obs, _ = env.reset(seed=60000 + cnt)
                env.set_fn(b_fn(kind, s0 + k))
                cnt += 1
                while True:
                    obs, r, term, trunc, _ = env.step(v1_policy(obs))
                    if term or trunc:
                        break
                c += int(env.term_reason == "collision")
            per[kind] = c
            tot += c
    finally:
        fe.D_DES = 0.20
    print(f"  attack seeds {s0}+: " + " ".join(f"{k}:{per[k]:>2d}/20" for k in per)
          + f"   TOTAL {tot:>3d}/100")
print("  review1 published (T3, continuous convention): 36/100 (50000+) and 40/100 (77000+)")

print()
print("=" * 96)
print("E2  d=1.0 / step-down 0.2s -- pole vs P+FF, 20-episode mean")
print("=" * 96)
sd2 = lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05
for lbl, pol in [("pole", pp.PositivePole()), ("P+FF", p_ff)]:
    fe.D_DES = 1.00
    env = ScriptedLeader(domain_randomize=False, gap_max=1.2)
    es, mg = [], []
    try:
        for k in range(20):
            if hasattr(pol, "reset"):
                pol.reset()
            obs, _ = env.reset(seed=2000 + k)
            env.set_fn(sd2)
            while True:
                obs, r, term, trunc, _ = env.step(pol(obs))
                if term or trunc:
                    break
            g = np.array(env.log["gap"])
            s = g[int(len(g) * 0.2):]
            es.append(float(np.abs(s - 1.0).mean() * 200))
            mg.append(float(s.mean()))
    finally:
        fe.D_DES = 0.20
    print(f"  {lbl:>5s}  20-ep mean err {np.mean(es):>7.2f}   mean gap {np.mean(mg):.3f}"
          f"   (archive _out_pole_final.txt: pole 78.28)")
print("\nDONE")
