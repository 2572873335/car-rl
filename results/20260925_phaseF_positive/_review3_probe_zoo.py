"""Round 13 driver: run the degenerate zoo against the six-clause conjunction."""
import sys, time, json
sys.path.insert(0, "/home/zy/car_rl/code0919")
sys.path.insert(0, "/home/zy/car_rl/code0919/results/20260925_phaseF_positive")
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from stable_baselines3 import PPO
import _review3_core as C

REPO = "/home/zy/car_rl/code0919"
NEP = 20

print("=" * 118)
print("R13  CRITERION ATTACK -- degenerate strategy zoo vs H-A3'(a)-(f)")
print(f"     judgement domain B* = B1..B5 (5 items) + B6 hard step-down 0.1/0.2/0.4 s")
print(f"     {NEP} episodes/item, env seeds 60000+k, attack seeds 50000+k, d=0.20")
print("     harness: new policy instance + reset() before EVERY episode (all policies)")
print("=" * 118)

t_start = time.time()

# ---- step 0: the anchor and the named baseline on the same instances ----
pole_c, pole_e, pole_m, pole_t = C.evaluate(lambda: C.Pole(), n_ep=NEP)
pole_pool_all = C.pooled(pole_e, C.ITEMS)
pole_pool_15 = C.pooled(pole_e, C.B15)
pole_pool_6 = C.pooled(pole_e, C.B6)
pole_crash = sum(pole_c.values())
print(f"\nPOLE ANCHOR on B*: crash {pole_crash}/{NEP*len(C.ITEMS)}  "
      f"pooled(all) {pole_pool_all:.2f}  pooled(B1-5) {pole_pool_15:.2f}  "
      f"pooled(B6) {pole_pool_6:.2f}")
for k in C.ITEMS:
    print(f"    {k:>8s}  crash {pole_c[k]:>2d}/{NEP}  err {pole_e[k]:>7.2f}  "
          f"meangap {pole_m[k]:.3f}  terms {pole_t[k]}")
np.save("/tmp/r13_pole_e.npy", np.array([pole_e[k] for k in C.ITEMS]))

V1 = REPO + "/ckpt/follow_stage2_final_v1.zip"
model = PPO.load(V1, device="cpu")


def v1_policy(obs):
    a, _ = model.predict(obs, deterministic=True)
    return np.asarray(a, dtype=np.float32)


v1_c, v1_e, v1_m, v1_t = C.evaluate(lambda: v1_policy, n_ep=NEP)
v1_tot = sum(v1_c.values())
v1_pool_15 = C.pooled(v1_e, C.B15)
v1_pool_all = C.pooled(v1_e, C.ITEMS)
print(f"\nNAMED BASELINE (frozen v1 RL) on B*: crash {v1_tot}/{NEP*len(C.ITEMS)}")
for k in C.ITEMS:
    print(f"    {k:>8s}  crash {v1_c[k]:>2d}/{NEP}  err {v1_e[k]:>7.2f}  terms {v1_t[k]}")
print(f"    pooled(B1-5) {v1_pool_15:.2f}   pooled(all) {v1_pool_all:.2f}")

# ---- gate values ----
N_ALL = NEP * len(C.ITEMS)
GATE_B = 0.10 * N_ALL
GATE_C = 0.5 * v1_tot
GATE_E = 1.5 * pole_pool_all
print(f"\nGATES: (b) crash <= {GATE_B:.1f}/{N_ALL}   (c) crash <= "
      f"{GATE_C:.1f}/{N_ALL} (50% of v1={v1_tot})   (e') pooled err <= "
      f"{GATE_E:.2f} (1.5 x pole pooled {pole_pool_all:.2f})")
print(f"      (b15) crash <= {0.10*NEP*5:.1f}/{NEP*5} if domain NOT extended by B6")

# ---- the zoo ----
rows = []
print("\n" + "=" * 118)
print("ZOO  (crash counts are total over B*, then split B1-5 | B6)")
print("=" * 118)
hdr = (f"  {'policy':>28s} {'logic':>11s} {'crashB*':>8s} {'crashB15':>9s} {'crashB6':>8s} "
       f"{'errB15':>7s} {'errB6':>7s} {'errB*':>7s} {'nom':>6s} {'frozen%':>8s} "
       f"(b) (c) (e') (f)  VERDICT")
print(hdr)
for spec in C.ZOO:
    t0 = time.time()
    nm = spec["name"]
    if nm == "pole (anchor)":
        crash, err, mgap, term = pole_c, pole_e, pole_m, pole_t
    elif nm == "v1 RL (baseline)":
        crash, err, mgap, term = v1_c, v1_e, v1_m, v1_t
    else:
        crash, err, mgap, term = C.evaluate(spec["factory"], n_ep=NEP)
    tot = sum(crash.values())
    t15 = sum(crash[k] for k in C.B15)
    t6 = sum(crash[k] for k in C.B6)
    e15 = C.pooled(err, C.B15)
    e6 = C.pooled(err, C.B6)
    eall = C.pooled(err, C.ITEMS)
    nc, ne = C.evaluate_nominal(spec["factory"], n_ep=NEP)
    b = tot <= GATE_B
    c = tot <= GATE_C
    e = eall <= GATE_E
    f = True   # domain includes B6 by construction
    ok = b and c and e
    rows.append(dict(name=nm, logic=spec["logic"], crash=tot, c15=t15, c6=t6,
                     e15=e15, e6=e6, eall=eall, nom=ne, ok=ok, b=b, c=c, e=e,
                     note=spec["note"], secs=time.time() - t0))
    print(f"  {nm:>28s} {spec['logic']:>11s} {tot:>5d}/{N_ALL} {t15:>6d}/{NEP*5} "
          f"{t6:>5d}/{NEP*3} {e15:>7.2f} {e6:>7.2f} {eall:>7.2f} {ne:>6.2f} "
          f"{'':>8s} {'Y' if b else 'n':>3s} {'Y' if c else 'n':>3s} "
          f"{'Y' if e else 'n':>4s} {'Y':>3s}  {'PASS' if ok else 'fail'}")
    sys.stdout.flush()

print(f"\nelapsed {time.time()-t_start:.1f}s")

# ---- summary: who passes, and what logic do they contain ----
print("\n" + "=" * 118)
print("THE ANSWER: passers of the full conjunction, by logic class")
print("=" * 118)
by_logic = {}
for r in rows:
    by_logic.setdefault(r["logic"], []).append(r)
for lg in sorted(by_logic):
    ps = [r["name"] for r in by_logic[lg] if r["ok"]]
    ns = [r["name"] for r in by_logic[lg] if not r["ok"]]
    print(f"\n  logic={lg}:  PASS {len(ps)}/{len(ps)+len(ns)}")
    for n in ps:
        print(f"      PASS  {n}")
    print(f"      (fail: {len(ns)})")

print("\n  ALL PASSERS:")
for r in rows:
    if r["ok"]:
        print(f"    {r['name']:>28s}  logic={r['logic']:>11s}  crash {r['crash']}/{N_ALL}  "
              f"errB* {r['eall']:.2f}  nom {r['nom']:.2f}   [{r['note']}]")
print("\n  NEAR-MISSES (fail exactly one clause):")
for r in rows:
    fails = [cl for cl, v in (("b", r["b"]), ("c", r["c"]), ("e", r["e"])) if not v]
    if len(fails) == 1:
        print(f"    {r['name']:>28s}  fails {fails[0]}   crash {r['crash']}/{N_ALL}  "
              f"errB* {r['eall']:.2f}")

json.dump(rows, open("/tmp/r13_rows.json", "w"), indent=1)
print("\nDONE")
