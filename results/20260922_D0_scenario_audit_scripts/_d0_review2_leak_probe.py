"""Review2 probe (B3): does DDESWrapper leak D_DES across runs in one process?

Evidence for plan_D0_scenario_audit_v3.md review2 blocking item B3:
  (a) normal interleaving (RL wrapper then P+FF wrapper, each close()s) is clean;
  (b) an exception between reset() and close() leaves fe.D_DES stale, and a
      later RAW env chases the wrong setpoint;
  (c) baseline_action() does NOT read D_DES (theorized risk -- falsified).

Run:  uv run python results/20260922_D0_scenario_audit_scripts/_d0_review2_leak_probe.py
"""
import hashlib
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")
import numpy as np
import follow_env as fe


class DDESWrapper:  # replica of the plan's wrapper (v3 section 1.1)
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

import inspect
src = inspect.getsource(fe.baseline_action)
print("(c) 'D_DES' in baseline_action body:", "D_DES" in src, flush=True)

w = DDESWrapper(1.00, 1.0, "constant")
w.make()
obs, _ = w.reset(2000)
print("\n(a) after RL wrapper reset d=1.00:  fe.D_DES =", fe.D_DES, flush=True)
for _ in range(5):
    obs, *_ = w.step(np.zeros(1, dtype=np.float32))
w.close()
print("    after RL close():               fe.D_DES =", fe.D_DES, flush=True)
w2 = DDESWrapper(0.30, 0.3, "constant")
w2.make()
obs, _ = w2.reset(2001)
print("    after P+FF wrapper reset d=0.30: fe.D_DES =", fe.D_DES, flush=True)
w2.close()
print("    after P+FF close():              fe.D_DES =", fe.D_DES, flush=True)

try:
    w3 = DDESWrapper(0.50, 1.0, "constant")
    w3.make()
    obs, _ = w3.reset(2002)
    print("\n(b) mid-episode fe.D_DES =", fe.D_DES, flush=True)
    raise RuntimeError("simulated crash mid-episode")
except RuntimeError:
    print("    exception escaped WITHOUT calling close()", flush=True)
print("    fe.D_DES AFTER leak (still 0.50):", fe.D_DES, flush=True)

fe.D_DES = 0.50
raw = fe.FollowEnv(domain_randomize=False)
raw.reset(2003)
for _ in range(4000):
    o, r, t, tr, _ = raw.step(np.zeros(1, dtype=np.float32))
    if t or tr:
        break
print(f"    raw env (no wrapper) under leaked D_DES=0.50: terminal gap="
      f"{raw.log['gap'][-1]:.3f}  (chased 0.50, not 0.20)", flush=True)

fe.D_DES = 0.20
h1 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]
print("\nfrozen follow_env.py untouched:", h0 == h1, f"({h0})", flush=True)
