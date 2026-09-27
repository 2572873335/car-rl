"""Does the AttackerEnv obs[1]==0 defect actually explain the Step 1 failure?

The retraction (commit a9007c7, AGENT_HANDOFF.md section 3.3) reasons:

    the scripted attack that "proved the surface was intact" ran through the same
    AttackerEnv, so it went down the same polluted path -- therefore the control
    is void too, and the whole Step 1 conclusion is void.

That reasoning assumes the corruption damages BOTH defenders. A first check
(_check_defender_damage.py) found it does not: the frozen source is

    a = (kp * e - (0.0 if use_ff else de)) / ACT_GAIN

`de` is read ONLY in the pure-P branch. P+FF (use_ff=True) never consults obs[1],
so a zeroed obs[1] cannot change its action by a single ulp. Measured: 0/106
steps differ.

This script settles the consequence two ways, both without retraining:

  A. Re-evaluate the EXISTING checkpoint under a corrected harness.
     Minimal faithful fix: feed the defender the obs returned by the PREVIOUS
     step() -- which describes the same physical state as the side probe, but
     with a real de. If the attacker still scores ~0/20 against v1, the defect
     does not explain the failure even for the defender it does damage.

  B. Measure the defect's effect on each defender directly: run the KNOWN-WINNING
     scripted step-downs under buggy vs corrected obs, and compare crash rates.
     This says whether "the scripted attack crashed through the broken path" was
     a valid control.

Neither run trains anything; both reuse frozen assets.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
from adversary_env import AttackerEnv, V_BASE, V_SPAN
import follow_env as fe

CKPT = f"{REPO}/results/20260925_phaseF_attacker/attacker_vs_both.zip"


class CorrectedAttackerEnv(AttackerEnv):
    """Same env, but the defender sees the true post-step obs.

    AttackerEnv.step() calls self.env._obs() BEFORE self.env.step(), so de is
    always zero. Here the defender acts on the obs that the PREVIOUS step()
    returned -- the same state, correctly measured.
    """

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._def_obs = self.env._obs().copy()   # valid: reset() just set prev_e
        return obs, info

    def step(self, action):
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        v_cmd = float(np.clip(V_BASE + V_SPAN * a, 0.05, 1.0))
        self.env._attacker_v = v_cmd
        self._attacker_v = v_cmd

        d_act = self.defender(self._def_obs)      # <- the corrected obs
        obs, _r, term, trunc, info = self.env.step(d_act)
        self._def_obs = obs.copy()                # true post-step obs for next time

        e = float(obs[0]) * 0.5
        reward = W_ERR_SCALE * abs(e) / 0.12 - W_IDLE_SCALE
        if term and self.env.term_reason == "collision":
            reward += W_CRASH_SCALE
        elif term:
            reward += 0.1 * W_CRASH_SCALE
        return self._att_obs(obs, self._attacker_v), reward, term, trunc, info


# mirror the module constants so the reward matches (not used for eval, but keep
# the env honest if it is reused)
import adversary_env as _ae
W_ERR_SCALE, W_CRASH_SCALE, W_IDLE_SCALE = _ae.W_ERR, _ae.W_CRASH, _ae.W_IDLE


def make_defender(kind):
    if kind == "pff":
        return lambda o: fe.baseline_action(o, use_ff=True)
    from stable_baselines3 import PPO
    m = PPO.load(f"{REPO}/ckpt/follow_stage2_final_v1.zip", device="cpu")
    return lambda o: m.predict(o, deterministic=True)[0]


def eval_learned(env_cls, kind, n_ep=20, seed0=7000, d_des=0.20):
    """Crash rate of the trained attacker, under the chosen harness."""
    from stable_baselines3 import PPO
    model = PPO.load(CKPT, device="cpu")
    pol = make_defender(kind)
    env = env_cls(pol, d_des=d_des)
    counts = {}
    for k in range(n_ep):
        obs, _ = env.reset(seed=seed0 + k)
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        counts[env.env.term_reason] = counts.get(env.env.term_reason, 0) + 1
    return counts.get("collision", 0), n_ep, counts


def eval_scripted(env_cls, kind, period, n_ep=20, seed0=60000, d_des=0.20):
    """Crash rate of a full-range scripted step-down, under the chosen harness."""
    pol = make_defender(kind)
    env = env_cls(pol, d_des=d_des)
    crash = 0
    for k in range(n_ep):
        env.reset(seed=seed0 + k)
        while True:
            want = 0.95 if int(env.env.t / period) % 2 == 0 else 0.05
            a = np.clip(np.array([(want - V_BASE) / V_SPAN], dtype=np.float32), -1, 1)
            obs, _r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if env.env.term_reason == "collision":
            crash += 1
    return crash, n_ep


print("=" * 84)
print("A. LEARNED ATTACKER, EXISTING CHECKPOINT, BUGGY vs CORRECTED HARNESS")
print("   (no retraining -- same weights, only the obs path is fixed)")
print("=" * 84)
for kind in ["pff", "v1"]:
    cb, nb, cntb = eval_learned(AttackerEnv, kind)
    cc, nc, cntc = eval_learned(CorrectedAttackerEnv, kind)
    print(f"\n  vs {kind:>3s}")
    print(f"    buggy     harness: {cb:>2d}/{nb} crashes   term={cntb}")
    print(f"    corrected harness: {cc:>2d}/{nc} crashes   term={cntc}")

print()
print("=" * 84)
print("B. SCRIPTED STEP-DOWN, BUGGY vs CORRECTED HARNESS")
print("   (was the 'surface is intact' control valid?)")
print("=" * 84)
for kind in ["pff", "v1"]:
    for period in [0.1, 0.2, 0.4]:
        cb, nb = eval_scripted(AttackerEnv, kind, period)
        cc, nc = eval_scripted(CorrectedAttackerEnv, kind, period)
        print(f"  vs {kind:>3s}  step-down {period}s:  "
              f"buggy {cb:>2d}/{nb}   corrected {cc:>2d}/{nc}")

print()
print("=" * 84)
print("Reading:")
print("  * P+FF never reads obs[1], so its two columns must be IDENTICAL -- any")
print("    difference there would itself be a bug.")
print("  * For v1, a gap between the columns measures what the defect was worth.")
print("  * If the learned attacker stays near 0/20 against v1 even when corrected,")
print("    the defect does not explain the Step 1 failure for that defender.")
print("=" * 84)
