"""Review2 probe (F / 10.4): do degenerate policies pass D0's ACCURACY criteria?

D0 criteria are precision-threshold gates (D0.1 thresh=5, D0.2 thresh=8,
obs-cm + settled). DATA_MANAGEMENT 10.4 requires a degenerate-policy counter
table. This probe supplies it for the following task: zero-action, random,
pure-P, blind a=+1, blind a=-1.

Result: all degenerate policies score 29-176 obs-cm vs thresholds 5/8 (5-20x),
so the accuracy gate excludes them cleanly. (Contrast Phase C: those were
hit-rate/win-type gates that the geometric family could "buy".)

Run:  uv run python results/20260922_D0_scenario_audit_scripts/_d0_review2_degenerate_probe.py
"""
import hashlib
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")
import numpy as np
import follow_env as fe


class DDESWrapper:
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


h0 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]


def run(policy, d, v, b, n_ep, seed0=2000):
    w = DDESWrapper(d, v, b)
    w.make()
    per, coll = [], 0
    for k in range(n_ep):
        obs, _ = w.reset(seed0 + k)
        while True:
            obs, r, t, tr, _ = w.step(policy(w.env, obs))
            if t or tr:
                break
        e = np.abs(np.array(w.env.log["gap"]) - d)
        per.append(e[int(len(e) * 0.2):].mean() * 200.0)
        if w.env.term_reason == "collision":
            coll += 1
    w.close()
    return float(np.mean(per)), coll


zero = lambda env, obs: np.zeros(1, dtype=np.float32)
rng = np.random.default_rng(0)
rand = lambda env, obs: rng.uniform(-1, 1, 1).astype(np.float32)
pureP = lambda env, obs: fe.baseline_action(obs, use_ff=False)
blindp = lambda env, obs: np.ones(1, dtype=np.float32)
blindm = lambda env, obs: -np.ones(1, dtype=np.float32)

print("DEGENERATE-POLICY PROBE (settled obs-cm; D0.1 thresh=5, D0.2 thresh=8)")
for nm, pol in [("zero", zero), ("random", rand), ("pure-P", pureP),
                ("blind +1", blindp), ("blind -1", blindm)]:
    e, c = run(pol, 0.50, 1.00, "constant", 20)
    print(f"  {nm:9s} d=0.50 v=1.00: {e:8.2f}  coll {c}/20")

print("  (reference cells)")
for d, v, b in [(0.50, 0.55, "constant"), (0.50, 1.00, "constant"),
                (1.00, 1.00, "constant")]:
    ez, cz = run(zero, d, v, b, 20)
    er, cr = run(rand, d, v, b, 20)
    print(f"  {'zero':9s} d={d} v={v} {b[:6]}: {ez:8.2f} coll {cz}/20   "
          f"random: {er:8.2f} coll {cr}/20")

h1 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]
print(f"\nfrozen follow_env.py untouched: {h0 == h1} ({h0})")
