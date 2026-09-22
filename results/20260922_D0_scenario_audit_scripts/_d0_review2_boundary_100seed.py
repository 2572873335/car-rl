"""Review2 probe (B1): is D0.2's 7.98-vs-8.00 a sampling artifact or structural?

D0.2 = "RL settled mean|e| <= 8 (obs-cm) and zero collision at d>=0.50".
The plan never defines the AGGREGATION UNIT (per-episode? per-seed? mean over
100 seeds?). This probe shows the answer changes pass/fail:

  * aggregate mean over 100 seeds: 7.95-7.98 (PASS, stable across 4 blocks)
  * fraction of individual episodes > 8.00: 36-44%  (would FAIL per-episode)

i.e. widening 30->100 seeds does NOT adjudicate the boundary; the value sits
structurally ~0.03 cm below 8.00.

Run:  uv run python results/20260922_D0_scenario_audit_scripts/_d0_review2_boundary_100seed.py
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


def run(policy, d_des, v_set, behavior, n_ep, seed0):
    w = DDESWrapper(d_des, v_set, behavior)
    w.make()
    per_ep, coll = [], 0
    for k in range(n_ep):
        obs, _ = w.reset(seed0 + k)
        while True:
            obs, r, term, trunc, _ = w.step(policy(w.env, obs))
            if term or trunc:
                break
        e = np.abs(np.array(w.env.log["gap"]) - d_des)
        per_ep.append(e[int(len(e) * 0.2):].mean() * 200.0)   # settled obs-cm
        if w.env.term_reason == "collision":
            coll += 1
    w.close()
    return np.array(per_ep), coll


from stable_baselines3 import PPO
model = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
rl = lambda env, obs: model.predict(obs, deterministic=True)[0]
def pff(env, obs):
    return fe.baseline_action(obs, use_ff=True)

print("D0.2 boundary stability, d=0.50 v=1.00 constant, settled obs-cm")
print("RL, 100 seeds x 4 blocks:")
for seed0 in [2000, 3000, 4000, 5000]:
    per, coll = run(rl, 0.50, 1.0, "constant", 100, seed0)
    print(f"  seeds {seed0}-{seed0+99}: agg_mean={per.mean():.3f} std={per.std():.3f} "
          f"min={per.min():.3f} max={per.max():.3f} ep>8.00: {(per > 8.0).sum()}/100 "
          f"coll={coll}/100")

per, coll = run(pff, 0.50, 1.0, "constant", 100, 2000)
print(f"P+FF, 100 seeds: agg_mean={per.mean():.3f} std={per.std():.3f} coll={coll}/100")

h1 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()[:16]
print(f"\nfrozen follow_env.py untouched: {h0 == h1} ({h0})")
