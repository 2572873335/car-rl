"""D0.5-D0.7 evaluation: does the regime-extended retrain recover the advantage?

Compares the v2 regime-extended checkpoint against the v1 audit checkpoint and
against P+FF, on the same grid and conventions as the D0 sweep
(results/20260922_D0_scenario_audit/d0_sweep.py), so the tables are directly
comparable.

Pre-registered thresholds (research/ASSUMPTIONS.md, owner ruling):
  D0.5  v=1.0 cells: RL <= P+FF (obs-cm)
  D0.6  v<=0.55 cells: RL not worse than the ORIGINAL checkpoint (<=10% slack)
  D0.7  9-cell (d,v) grid: >=7 cells RL not worse than P+FF and zero collision

Convention (unchanged): settled = mean|e| over e[20%:]; obs-cm = 200*|e[m]|;
aggregation = mean of per-seed settled means; frozen file untouched (wrapper).

Run:  uv run python results/20260922_D0_regime_ext/eval_d05_d07.py
"""
import hashlib
import io
import json
import os
import sys
import time

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

OUT = f"{REPO}/results/20260922_D0_regime_ext"
os.makedirs(OUT, exist_ok=True)

CKPT_V1 = f"{REPO}/ckpt/follow_stage2_final_v1.zip"
CKPT_V2 = f"{REPO}/ckpt/follow_stage2_v2_speedext.zip"
N_SEED = 100
SEED0 = 2000

# (d, v, behavior, is_core) -- identical grid to the D0 sweep
CELLS = [
    (0.20, 0.30, "constant", True),
    (0.20, 0.55, "constant", True),
    (0.20, 1.00, "constant", True),
    (0.50, 0.30, "constant", True),
    (0.50, 0.55, "constant", True),
    (0.50, 1.00, "constant", True),
    (0.50, 1.00, "sinusoid", False),
    (0.50, 1.00, "brake", False),
    (1.00, 0.30, "constant", True),
    (1.00, 0.55, "constant", True),
    (1.00, 1.00, "constant", True),
]


class DDESWrapper:
    """Runtime D_DES override; frozen file untouched (review2 B3: try/finally)."""

    def __init__(self, d_des, v_set, behavior):
        self.d_des, self.v_set, self.behavior = d_des, v_set, behavior

    def make(self):
        self.env = fe.FollowEnv(domain_randomize=False)
        return self.env

    def reset(self, seed):
        fe.D_DES = self.d_des
        obs, info = self.env.reset(seed=seed)
        self.env.v_set = self.v_set
        self.env.behavior = self.behavior
        return obs, info

    def step(self, a):
        return self.env.step(a)

    def close(self):
        fe.D_DES = 0.20


def rollout(policy, d_des, v_set, behavior, n_ep, seed0):
    w = DDESWrapper(d_des, v_set, behavior)
    w.make()
    settled, reasons = [], {}
    try:
        for k in range(n_ep):
            obs, _ = w.reset(seed0 + k)
            while True:
                obs, r, term, trunc, _ = w.step(policy(w.env, obs))
                if term or trunc:
                    break
            e = np.abs(np.array(w.env.log["gap"]) - d_des)
            settled.append(float(e[int(len(e) * 0.2):].mean() * 200.0))
            rr = w.env.term_reason
            reasons[rr] = reasons.get(rr, 0) + 1
    finally:
        w.close()
        assert fe.D_DES == 0.20, f"D_DES leaked: {fe.D_DES}"
    return settled, reasons


def main():
    from stable_baselines3 import PPO
    m1 = PPO.load(CKPT_V1)
    m2 = PPO.load(CKPT_V2)

    policies = {
        "v1":   lambda env, obs: m1.predict(obs, deterministic=True)[0],
        "v2":   lambda env, obs: m2.predict(obs, deterministic=True)[0],
        "P+FF": lambda env, obs: fe.baseline_action(obs, use_ff=True),
    }

    h0 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()
    with io.open(f"{OUT}/eval_config.txt", "w", encoding="utf-8") as f:
        f.write(f"v1: {CKPT_V1}\nv2: {CKPT_V2}\n")
        f.write(f"n_seed: {N_SEED} seed0: {SEED0}\n")
        f.write("convention: settled, obs-cm = 200*|e|, agg = mean of per-seed means\n")
        f.write(f"frozen follow_env.py sha256: {h0}\n")

    raw, summary = [], []
    t0 = time.time()
    print("=" * 104, flush=True)
    print(f"D0.5-D0.7 EVAL  (v1 vs v2 vs P+FF, {len(CELLS)} cells x {N_SEED} seeds)", flush=True)
    print("=" * 104, flush=True)
    print(f"  {'cell':>24s} {'v1':>8s} {'v2':>8s} {'P+FF':>8s} "
          f"{'v2 v1?':>9s} {'v2 PFF?':>9s} {'coll(v2)':>9s}", flush=True)

    for d, v, b, core in CELLS:
        cell = f"d={d:.2f} v={v:.2f} {b[:7]}"
        row = {}
        for name, pol in policies.items():
            settled, reasons = rollout(pol, d, v, b, N_SEED, SEED0)
            arr = np.array(settled)
            row[name] = dict(mean=float(arr.mean()), std=float(arr.std(ddof=1)),
                             se=float(arr.std(ddof=1) / np.sqrt(len(arr))),
                             reasons=reasons, settled=settled)
            raw.append(dict(d=d, v=v, behavior=b, core=core, policy=name,
                            settled=settled, reasons=reasons))
        v2_better_v1 = row["v2"]["mean"] <= row["v1"]["mean"] * 1.10   # D0.6 slack
        v2_beats_pff = row["v2"]["mean"] < row["P+FF"]["mean"]          # D0.5/D0.7
        coll = row["v2"]["reasons"].get("collision", 0)
        print(f"  {cell:>24s} {row['v1']['mean']:>8.3f} {row['v2']['mean']:>8.3f} "
              f"{row['P+FF']['mean']:>8.3f} {str(v2_better_v1):>9s} "
              f"{str(v2_beats_pff):>9s} {coll:>9d}", flush=True)
        summary.append(dict(d=d, v=v, behavior=b, core=core,
                            v1=row["v1"]["mean"], v2=row["v2"]["mean"],
                            pff=row["P+FF"]["mean"],
                            v2_not_worse_v1=v2_better_v1,
                            v2_beats_pff=v2_beats_pff, coll=coll))
        print("-" * 104, flush=True)

    with io.open(f"{OUT}/eval_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=1)

    # ---------------- verdicts ----------------
    print(flush=True)
    print("=" * 104, flush=True)
    print("  VERDICTS vs pre-registered thresholds", flush=True)
    print("=" * 104, flush=True)

    # D0.5: v=1.0 cells, v2 <= P+FF
    v10 = [s for s in summary if s["v"] == 1.00]
    d05 = all(s["v2_beats_pff"] for s in v10)
    print(f"  D0.5 (v=1.0 cells, v2 <= P+FF): {'PASS' if d05 else 'FAIL'}", flush=True)
    for s in v10:
        print(f"        d={s['d']:.2f} {s['behavior']:>8s}  v2={s['v2']:.3f}  "
              f"P+FF={s['pff']:.3f}  -> {'ok' if s['v2_beats_pff'] else 'LOSS'}", flush=True)

    # D0.6: v<=0.55 cells, v2 not worse than v1 (<=10%)
    low = [s for s in summary if s["v"] <= 0.55]
    d06 = all(s["v2_not_worse_v1"] for s in low)
    print(f"\n  D0.6 (v<=0.55 cells, v2 within 10% of v1): {'PASS' if d06 else 'FAIL'}", flush=True)
    worst = max(low, key=lambda s: s["v2"] / max(s["v1"], 1e-9))
    print(f"        worst ratio: d={worst['d']:.2f} v={worst['v']:.2f} "
          f"{worst['behavior']}  v1={worst['v1']:.3f} v2={worst['v2']:.3f} "
          f"({worst['v2']/max(worst['v1'],1e-9):.2f}x)", flush=True)

    # D0.7: core 9-cell grid, >=7 cells v2 beats P+FF and zero collision
    core = [s for s in summary if s["core"]]
    n_ok = sum(1 for s in core if s["v2_beats_pff"] and s["coll"] == 0)
    d07 = n_ok >= 7
    print(f"\n  D0.7 (core {len(core)} cells, >=7 with v2<=P+FF and zero collision): "
          f"{'PASS' if d07 else 'FAIL'}  ({n_ok}/{len(core)})", flush=True)

    with io.open(f"{OUT}/eval_summary.csv", "w", encoding="utf-8") as f:
        f.write("d,v,behavior,core,v1,v2,pff,v2_not_worse_v1,v2_beats_pff,coll\n")
        for s in summary:
            f.write(f"{s['d']},{s['v']},{s['behavior']},{int(s['core'])},"
                    f"{s['v1']:.4f},{s['v2']:.4f},{s['pff']:.4f},"
                    f"{int(s['v2_not_worse_v1'])},{int(s['v2_beats_pff'])},{s['coll']}\n")

    h1 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()
    print(f"\n  frozen follow_env.py intact: {h0 == h1}", flush=True)
    print(f"  elapsed: {time.time() - t0:.0f}s", flush=True)
    print(f"  D0.5={d05}  D0.6={d06}  D0.7={d07}", flush=True)


if __name__ == "__main__":
    main()
