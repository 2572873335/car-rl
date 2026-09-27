"""Which defender does the AttackerEnv obs[1]==0 defect actually damage?

CLAIM UNDER TEST (handoff section 3.3): the defect weakens BOTH defenders --
v1 RL loses its obs slot 2, and P+FF's feedforward term "silently vanishes",
degrading it to pure P.

WHY THAT SECOND HALF IS SUSPECT: the frozen source (follow_env.py:60-71) is

    a = (kp * e - (0.0 if use_ff else de)) / ACT_GAIN

The `de` read happens ONLY in the pure-P branch (use_ff=False). With use_ff=True
the expression is (kp*e - 0.0)/ACT -- `de` is not consulted, so a zeroed de
cannot nullify anything. If this is right, P+FF is untouched and the defender
that degrades is pure P, in the opposite direction.

METHOD (isolating slot 1 alone):
The side probe env._obs() and the obs returned by the PREVIOUS step() describe
the same physical state -- nothing moved in between. So they should agree in
slots 0, 2, 3 and differ ONLY in slot 1 if de is genuinely the corrupted term.
This script first asserts that, then measures each defender's action under the
two obs values.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

fe.D_DES = 0.20


def scripted_leader(t):
    return 0.95 if int(t / 0.2) % 2 == 0 else 0.05


def collect(n_steps=200):
    """Walk one episode; record (true_obs, side_probe_obs) pairs."""
    env = fe.FollowEnv(domain_randomize=False, t_max=20.0)
    obs, _ = env.reset(seed=7000)
    env._leader_v = lambda: scripted_leader(env.t)

    pairs = []
    for _ in range(n_steps):
        # side probe BEFORE the step: what AttackerEnv.step() feeds the defender
        side = env._obs().copy()
        # `obs` is the post-step obs from the previous iteration: same state
        pairs.append((obs.copy(), side))
        obs, _, term, trunc, _ = env.step(np.array([0.3], dtype=np.float32))
        if term or trunc:
            break
    return pairs


pairs = collect()
print("=" * 78)
print("STEP 1 -- is slot 1 the ONLY corrupted slot?")
print("=" * 78)
true_arr = np.array([p[0] for p in pairs])
side_arr = np.array([p[1] for p in pairs])
for slot, label in enumerate(["e (gap err)", "de (gap rate)", "v_f (own spd)", "e_lat"]):
    delta = np.abs(true_arr[:, slot] - side_arr[:, slot])
    print(f"  slot {slot} {label:<14s}: max |true - side| = {delta.max():.6f}"
          f"   {'<-- CORRUPTED' if delta.max() > 1e-9 else '(identical)'}")

print()
print("  side-probe slot 1 is identically zero everywhere:",
      bool(np.allclose(side_arr[:, 1], 0.0)),
      f"(max |de| on the true path = {np.abs(true_arr[:, 1]).max():.4f})")

print()
print("=" * 78)
print("STEP 2 -- does the corruption change each defender's ACTION?")
print("=" * 78)


def make_defenders():
    d = {}
    d["P+FF  (use_ff=True)"] = lambda o: fe.baseline_action(o, use_ff=True)
    d["pure P (use_ff=False)"] = lambda o: fe.baseline_action(o, use_ff=False)
    try:
        from stable_baselines3 import PPO
        m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip")
        d["v1 RL (frozen)"] = lambda o: m.predict(o, deterministic=True)[0]
    except Exception as e:                                   # noqa: BLE001
        print(f"  (v1 RL unavailable: {e})")
    return d


for name, pol in make_defenders().items():
    diffs = []
    for true_obs, side_obs in pairs:
        a_true = float(np.asarray(pol(true_obs)).ravel()[0])
        a_side = float(np.asarray(pol(side_obs)).ravel()[0])
        diffs.append(abs(a_true - a_side))
    diffs = np.array(diffs)
    n_diff = int((diffs > 1e-9).sum())
    verdict = "DAMAGED" if n_diff > 0 else "UNTOUCHED"
    print(f"\n  {name}")
    print(f"    steps where the bug changes the action: {n_diff}/{len(diffs)}")
    print(f"    mean |delta a| = {diffs.mean():.6f}   max = {diffs.max():.6f}")
    print(f"    => {verdict}")

print()
print("=" * 78)
print("An UNTOUCHED defender means the defect cannot explain its measured result.")
print("=" * 78)
