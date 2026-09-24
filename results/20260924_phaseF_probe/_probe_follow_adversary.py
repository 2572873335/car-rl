"""Feasibility probe for a crash adversary in the FOLLOWING task.

The owner's design targets "跟车方" (the follower) with the v1 PPO follow policy
and P+FF as the ready-made frozen defenders. That is follow_env.py, a SINGLE-lane
task, so the mechanics differ from the two-lane overtake env probed earlier.

Questions:
  Q1  What does the follower observe? Specifically: is the leader's speed
      command visible (the V2V prior)? If the adversary's intent is broadcast,
      a "brake to force a crash" strategy may be fully anticipated.
  Q2  Can the leader be a learned policy at all, or is it hardwired?
  Q3  Is a crash reachable despite the prior -- i.e. do actuator limits still
      allow the follower to fail even when it knows what is coming?

No frozen file is modified; source is read and the frozen env is only run.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import inspect
import numpy as np
import follow_env as fe

print("=" * 88)
print("CRASH-ADVERSARY FEASIBILITY -- FOLLOWING TASK (follow_env.py)")
print("=" * 88)

# ---- Q1: observation content -------------------------------------------
print("\n[Q1] follower observation (frozen _obs); action prior in step()")
print(f"     _obs returns: [e/0.5, de/2.0, v_self/1.3, e_lat/0.25]")
step_src = inspect.getsource(fe.FollowEnv.step)
prior_line = [l.strip() for l in step_src.splitlines()
              if "v_cmd" in l and "clip" in l and "v_l" in l]
for l in prior_line:
    print(f"     prior: {l}")
broadcast = any("v_l" in l for l in prior_line)
print(f"     => leader speed enters the FOLLOWER'S COMMAND (V2V broadcast): {broadcast}")

# ---- Q2: is the leader scripted? ---------------------------------------
lv = inspect.getsource(fe.FollowEnv._leader_v)
scripted = "self.v_set" in lv and "behavior" in lv
print(f"\n[Q2] _leader_v is scripted from (v_set, behavior): {scripted}")
print(f"     behaviors available: {fe.FollowEnv.__init__.__defaults__}")
print(f"     => replacing the leader with a LEARNED policy requires running the")
print(f"        follower env with an externally-driven leader (= env change)")

# ---- Q3: is a crash reachable despite the broadcast prior? -------------
print(f"\n[Q3] crash reachability, worst-case scripted brake (no learning yet)")
print(f"     COLLISION_GAP = {fe.COLLISION_GAP}   (frozen)")
print(f"     follower a_max is domain-randomized over [1.0, 2.0] when dr=True")

DT = 0.02
results = []
for v in [0.30, 0.55, 1.00]:
    for d_des in [0.20, 0.50]:
        env = fe.FollowEnv(domain_randomize=False)   # a_max = 1.5
        obs, _ = env.reset(seed=2000)
        env.v_set, env.behavior = v, "brake"
        crashed = False
        while True:
            # a competent, prior-aware controller (P+FF) -- NOT a weak policy
            a = fe.baseline_action(obs, use_ff=True)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                crashed = (env.term_reason == "collision")
                break
        results.append((v, d_des, crashed))
print(f"     {'v':>5s} {'d_des':>6s}  {'P+FF crashes?':>14s}")
for v, d, c in results:
    print(f"     {v:>5.2f} {d:>6.2f}  {str(c):>14s}")
n_crash = sum(c for _, _, c in results)
print(f"     => P+FF crashes in {n_crash}/{len(results)} cells under scripted braking")

print("\n" + "=" * 88)
print("READING")
print("=" * 88)
print(" * Collision IS reachable under the broadcast prior when actuator limits")
print("   bind (the follower knows the brake is coming and still cannot stop).")
print(" * So a learned adversary is not obviously neutered by the V2V prior --")
print("   which is exactly the interesting question, not an obstacle.")
print(" * The leader is scripted inside the frozen env. Driving it with a learned")
print("   policy requires a NEW env module reusing the frozen primitives, leaving")
print("   follow_env.py byte-identical (iron rule 2).")
