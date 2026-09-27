"""Same observation, same model, different action -- why?

Decisive trace showed: at step 40 both paths see obs[2]=0.92308, but plain emits
action 0.3745 while wrapped emits 1.0000. That is not a physics difference, it is
a model-call difference. Enumerate the candidate causes.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
from stable_baselines3 import PPO

m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")

# reconstruct the step-40 observation as closely as possible
obs = np.array([0.82023, 0.0, 0.92308, 0.0], dtype=np.float32)

print("=" * 72)
print("SAME OBS, DIFFERENT CALL STYLES")
print("=" * 72)
print(f"  obs                  : {obs}")
print(f"  flat  predict        : {m.predict(obs, deterministic=True)[0]}")
print(f"  batched [None,:]     : {m.predict(obs[None, :], deterministic=True)[0]}")
print(f"  float64              : {m.predict(obs.astype(np.float64), deterministic=True)[0]}")
print(f"  non-contiguous       : {m.predict(obs[::1], deterministic=True)[0]}")

# the actual suspect: is the WRAPPED path's model the same object/state?
print()
print("=" * 72)
print("IS IT THE SAME MODEL? compare two independently loaded copies")
print("=" * 72)
m2 = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
print(f"  copy1: {m.predict(obs, deterministic=True)[0]}")
print(f"  copy2: {m2.predict(obs, deterministic=True)[0]}")

# and what does the policy actually see internally?
print()
print("=" * 72)
print("POLICY LOGITS for the two styles")
print("=" * 72)
import torch
with torch.no_grad():
    t = torch.as_tensor(obs[None, :], dtype=torch.float32)
    d = m.policy.get_distribution(t)
    print(f"  obs as given, [None,:]: mean {d.distribution.mean.item():.4f}")
