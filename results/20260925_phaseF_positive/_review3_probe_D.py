"""Round 13 probe D -- C4 counterfactual + a targeted reset-trap test."""
import sys
import numpy as np

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/results/20260925_phaseF_positive")
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import follow_env as fe
import _review3_core as C
from stable_baselines3 import PPO

V1 = REPO + "/ckpt/follow_stage2_final_v1.zip"
model = PPO.load(V1, device="cpu")


def v1_policy(obs):
    a, _ = model.predict(obs, deterministic=True)
    return np.asarray(a, dtype=np.float32)


print("=" * 100)
print("C4  COUNTERFACTUAL: (b)(c)(e') graded on B1-B5 ONLY (no B6) -- review2's channel")
print("=" * 100)
pole_c, pole_e, _, _ = C.evaluate(lambda: C.Pole(), items=C.B15, n_ep=20)
v1_c, v1_e, _, _ = C.evaluate(lambda: v1_policy, items=C.B15, n_ep=20)
pole15 = C.pooled(pole_e, C.B15)
v1tot15 = sum(v1_c.values())
print(f"  v1 baseline on B1-B5: crash {v1tot15}/100 (per-item "
      + " ".join(f"{k}:{v1_c[k]}" for k in C.B15) + f")  pooled err {C.pooled(v1_e, C.B15):.2f}")
print(f"  pole on B1-B5:       crash {sum(pole_c.values())}/100  pooled err {pole15:.2f}")
gate_b15 = 10.0
gate_c15 = 0.5 * v1tot15
gate_e15 = 1.5 * pole15
print(f"  gates: (b15)<=10/100  (c15)<={gate_c15:.1f}/100  (e15')<={gate_e15:.2f}")
print(f"\n  {'policy':>30s} {'crash15':>8s} {'errB15':>7s} {'nom':>6s}  "
      f"(b15)(c15)(e15')  VERDICT")
for nm, f in (("pole (anchor)", lambda: C.Pole()),
              ("Pcap a+<=0.30", lambda: C.PCap(0.8, 0.30)),
              ("Pcap a+<=0.50", lambda: C.PCap(0.8, 0.50)),
              ("P+FF kp=0.60", lambda: C.PFF(0.60)),
              ("P+FF kp=0.40", lambda: C.PFF(0.40)),
              ("freezeGap g<.35 c>.20", lambda: C.FreezeGap(0.35, 0.20)),
              ("clamp only", lambda: C.ClampOnly()),
              ("zero action", lambda: C.Zero()),
              ("brake only", lambda: C.BrakeOnly())):
    crash, err, mg, tr = C.evaluate(f, items=C.B15, n_ep=20)
    tot = sum(crash.values())
    e15 = C.pooled(err, C.B15)
    ncc, ne = C.evaluate_nominal(f, n_ep=20)
    b = tot <= gate_b15
    c = tot <= gate_c15
    e = e15 <= gate_e15
    print(f"  {nm:>30s} {tot:>5d}/100 {e15:>7.2f} {ne:>6.2f}  "
          f"{'Y' if b else 'n':>4s} {'Y' if c else 'n':>4s} {'Y' if e else 'n':>5s}   "
          f"{'PASS' if (b and c and e) else 'fail'}")

print("\n" + "=" * 100)
print("D2  TARGETED RESET TRAP: pole reused across episodes without reset, B6 family")
print("=" * 100)


def trap(items, n_ep=20, reset=True):
    fe.D_DES = 0.20
    pol = C.Pole()
    if reset:
        pass
    out = {}
    try:
        for kind in items:
            env = C.ScriptedLeader(domain_randomize=False, gap_max=1.2)
            c, es = 0, []
            for k in range(n_ep):
                if reset:
                    pol.reset()
                obs, _ = env.reset(seed=60000 + k)
                env.set_fn(C.attack_fn(kind, 50000 + k))
                while True:
                    obs, r, term, trunc, _ = env.step(pol(obs))
                    if term or trunc:
                        break
                g = np.array(env.log["gap"])
                s = g[int(len(g) * 0.2):]
                es.append(float(np.abs(s - d).mean() * 200) if False else
                          float(np.abs(s - 0.20).mean() * 200))
                c += int(env.term_reason == "collision")
            out[kind] = (c, float(np.mean(es)))
    finally:
        fe.D_DES = 0.20
    return out


for reset in (True, False):
    r = trap(C.ITEMS, reset=reset)
    print(f"  reset={reset}: " + "  ".join(f"{k}:{r[k][0]}/20(e{r[k][1]:.1f})"
                                           for k in C.ITEMS))
print("\n  NOTE: shows the reset-dependence of a STATEFUL policy's readings.")
print("DONE")
