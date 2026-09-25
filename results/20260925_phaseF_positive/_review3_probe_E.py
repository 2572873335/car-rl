"""Round 13 probe E -- THE SPACING HOLE.

H-A3'(a)-(f) never pins d_des.  B's parameterisation (holdout_B.md) does not
mention d_des either; review1/review2 and the pole's own archive table all run
THREE spacings (0.20 / 0.50 / 1.00).

plan sec 2.3.1(i) already established that at d=1.0 the step-down cells are
ACTUATOR-LIMITED DEAD CELLS: the LEADER's own actuator cannot track the 0.1-0.4 s
square wave, the gap parks at ~0.607 m whatever the follower does, and "0 crash"
is therefore empty credit (zero action is also 0/20).

If (f) forces the hard family into the judged domain, and the hard family is
exactly the family that is dead at d=1.0, then the conjunction may collapse back
to the review1 failure mode AT THAT SPACING -- independently of B6 existing.

This probe measures {pole, zero action, P+FF, Pcap0.30} on B* at d = 0.20/0.50/1.00
and grades (b)(c)(e') on each spacing separately.
"""
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


POLS = [("pole (anchor)", lambda: C.Pole(), "YES"),
        ("zero action", lambda: C.Zero(), "NONE"),
        ("P+FF kp=0.8", lambda: C.PFF(0.8), "NONE"),
        ("Pcap a+<=0.30", lambda: C.PCap(0.8, 0.30), "NONE"),
        ("brake only", lambda: C.BrakeOnly(), "YES"),
        ("v1 RL (baseline)", lambda: v1_policy, "RL")]

print("=" * 118)
print("E  SPACING SENSITIVITY of H-A3' on B* -- does the hard family stay alive at d=1.0?")
print("=" * 118)

store = {}
for d in (0.20, 0.50, 1.00):
    print(f"\n---- d_des = {d:.2f} ----")
    print(f"  {'policy':>17s} {'crash':>9s} {'B15':>8s} {'B6':>7s} {'errB15':>8s} "
          f"{'errB6':>8s} {'errB*':>8s} {'gapB6':>7s} {'nom':>6s}")
    for nm, f, lg in POLS:
        crash, err, mg, tr = C.evaluate(f, n_ep=20, d=d)
        tot = sum(crash.values())
        t15 = sum(crash[k] for k in C.B15)
        t6 = sum(crash[k] for k in C.B6)
        e15 = C.pooled(err, C.B15)
        e6 = C.pooled(err, C.B6)
        eall = C.pooled(err, C.ITEMS)
        g6 = float(np.mean([mg[k] for k in C.B6]))
        ncc, ne = C.evaluate_nominal(f, n_ep=20, d=d)
        store[(d, nm)] = dict(tot=tot, t15=t15, t6=t6, e15=e15, e6=e6, eall=eall,
                              g6=g6, nom=ne, tr=tr)
        print(f"  {nm:>17s} {tot:>5d}/160 {t15:>5d}/100 {t6:>4d}/60 {e15:>8.2f} "
              f"{e6:>8.2f} {eall:>8.2f} {g6:>7.3f} {ne:>6.2f}")

print("\n" + "=" * 118)
print("E2  SIX-CLAUSE VERDICT PER SPACING  (gates recomputed at each spacing from the")
print("    frozen v1 baseline and from the pole's own pooled value at that spacing)")
print("=" * 118)
for d in (0.20, 0.50, 1.00):
    v1 = store[(d, "v1 RL (baseline)")]
    pol = store[(d, "pole (anchor)")]
    gb = 0.10 * 160
    gc = 0.5 * v1["tot"]
    ge = 1.5 * pol["eall"]
    print(f"\n  d={d:.2f}:  v1 crash {v1['tot']}/160 -> (b)<={gb:.0f}, (c)<={gc:.0f};  "
          f"pole pooled {pol['eall']:.2f} -> (e')<={ge:.2f}")
    print(f"    {'policy':>17s} {'crash':>9s} {'errB*':>8s}   (b) (c) (e')  VERDICT")
    for nm, f, lg in POLS:
        s = store[(d, nm)]
        b = s["tot"] <= gb
        c = s["tot"] <= gc
        e = s["eall"] <= ge
        print(f"    {nm:>17s} {s['tot']:>5d}/160 {s['eall']:>8.2f}   "
              f"{'Y' if b else 'n'} {'Y' if c else 'n'} {'Y' if e else 'n':>3s}   "
              f"{'PASS' if (b and c and e) else 'fail'}   [{lg}]")

print("\n" + "=" * 118)
print("E3  DELIVERED vs COMMANDED leader speed (ACTUAL = leader's own Car.v).")
print("    The frozen env logs v_l = the COMMAND; the leader's delivered speed is not")
print("    logged, so it is captured with an instrumented subclass here.")
print("=" * 118)


class Instrumented(C.ScriptedLeader):
    def step(self, action):
        out = super().step(action)
        self.act_v.append(self.leader.v)
        return out


def delivered(d, kind):
    fe.D_DES = d
    env = Instrumented(domain_randomize=False, gap_max=1.2)
    env.act_v = []
    cmd, act, clip0, n = [], [], 0, 0
    try:
        for k in range(10):
            pol = C.Pole()
            pol.reset()
            obs, _ = env.reset(seed=60000 + k)
            fn = C.attack_fn(kind, 50000 + k)
            env.set_fn(fn)
            env.act_v = []
            while True:
                obs, r, term, trunc, _ = env.step(pol(obs))
                if term or trunc:
                    break
            cmd += [fn(t) for t in env.log["t"]]
            act += list(env.act_v)
            n += len(env.log["t"])
    finally:
        fe.D_DES = 0.20
    return np.array(cmd), np.array(act), n


for d in (0.20, 1.00):
    print(f"\n  d={d:.2f}")
    for kind in C.B6:
        cmd, act, n = delivered(d, kind)
        r = (act.max() - act.min()) / (cmd.max() - cmd.min())
        print(f"    {kind:>8s}  cmd mean {cmd.mean():.3f} pp {cmd.max()-cmd.min():.3f} "
              f"-> ACTUAL mean {act.mean():.3f} pp {act.max()-act.min():.3f}   "
              f"delivered/commanded pp ratio {r:.2f}")

print("\nDONE")
