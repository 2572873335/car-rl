"""Final check: does the plain path really see obs[2]=0.92308 at step 40?

The model returns 1.0 for that obs in every call style, matching the WRAPPED path.
So either the plain path never saw that obs, or its model call used something else.

Print the plain path's ACTUAL acting observation at steps 38-42 and the action it
produced, then feed that exact vector to the model and see if it reproduces.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe
from stable_baselines3 import PPO

m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
fe.D_DES = 0.20
env = fe.FollowEnv(domain_randomize=False)
obs, _ = env.reset(seed=2000)
env.v_set, env.behavior = 0.50, "constant"

print("=" * 78)
print("PLAIN PATH: acting obs and action, steps 36-44")
print("=" * 78)
print(f"  {'k':>4s} {'acting_obs':>34s} {'action':>8s} {'reproduced':>11s}")
for k in range(46):
    acting = obs.copy()
    a, _ = m.predict(acting, deterministic=True)
    repro = m.predict(acting, deterministic=True)[0]
    if 36 <= k <= 44:
        print(f"  {k:>4d} {str(np.round(acting,5)):>34s} "
              f"{float(a[0]):>8.4f} {float(repro[0]):>11.4f}")
    obs, r, term, trunc, _ = env.step(a)
    if term or trunc:
        print(f"  (terminated at {k}: {env.term_reason})")
        break

print()
print("  If 'action' != 'reproduced' for the SAME obs, the model is nondeterministic.")
print("  If they match, the plain path genuinely saw those obs values.")
