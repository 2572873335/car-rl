"""Positive pole for Phase F: a hand-coded "anticipate and yield" follower.

Required by the plan (Q6): this is the threshold anchor for H-A3, so it must
exist and be reviewed BEFORE attacker training starts.

Acceptance (owner):
  A. zero collisions against BOTH probe attacks (step-down 0.1s, osc 0.5Hz)
  B. survives against the whole probe family (not just those two)
  C. NOT fakeable by "crawl far away" -- nominal tracking must stay within
     1.5x the P+FF baseline (H-A4 applies to the positive pole too)

This script searches a FAMILY of hand-coded controllers rather than asserting one
design, because the physical question (can any causal reaction survive a step the
actuator cannot track?) is better answered by measurement than by reasoning.

Interface: obs = [e/0.5, de/2.0, v_self/1.3, e_lat/0.25], action in [-1, 1].
The frozen step computes v_cmd = clip(v_l + 0.8*a, 0, 1.3), so a=0 -> match
leader speed; a<0 -> open the gap.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

ACT = fe.ACT_GAIN


class ScriptedLeader(fe.FollowEnv):
    def set_fn(self, fn):
        self._fn = fn

    def _leader_v(self):
        return float(np.clip(self._fn(self.t), 0.05, 1.0))


# ---------------- candidate controllers ----------------

def p_ff(obs, kp=0.8):
    """Reference: the P+FF rule (known to crash 20/20 on step-down 0.1s)."""
    e, de = obs[0] * 0.5, obs[1] * 2.0
    return np.array([np.clip(kp * e / ACT, -1, 1)], dtype=np.float32)


def yield_inst(obs, kp=0.8, gain=0.6, cap=0.6):
    """Yield in proportion to instantaneous closing speed."""
    e, de = obs[0] * 0.5, obs[1] * 2.0
    margin = min(max(-de, 0.0) * gain, cap)
    e_eff = e - margin
    return np.array([np.clip(kp * e_eff / ACT, -1, 1)], dtype=np.float32)


class YieldStateful:
    """Yield with memory: once a hostile pattern is seen, hold a large gap for a
    while, then recover slowly. Rationale: a purely reactive controller cannot
    outrun an untrackable step, so it must have ALREADY opened the gap."""

    def __init__(self, kp=0.8, margin_max=0.55, attack_thresh=0.35,
                 decay=0.995, rise=1.0):
        self.kp, self.margin_max = kp, margin_max
        self.attack_thresh = attack_thresh
        self.decay, self.rise = decay, rise
        self.margin = 0.0

    def __call__(self, obs):
        e, de = obs[0] * 0.5, obs[1] * 2.0
        # hostile pattern detector: abrupt gap-rate change
        if abs(de) > self.attack_thresh:
            # open the gap hard, fast
            self.margin = min(self.margin + self.rise * abs(de), self.margin_max)
        else:
            self.margin *= self.decay            # recover slowly
        e_eff = e - self.margin
        return np.array([np.clip(self.kp * e_eff / ACT, -1, 1)], dtype=np.float32)

    def reset(self):
        self.margin = 0.0


def sched_gap(obs, kp=0.8, base=0.20, span=0.5, thresh=0.3):
    """Reactive: target a larger gap when |de| is large (no memory)."""
    e, de = obs[0] * 0.5, obs[1] * 2.0
    margin = min(abs(de) / max(thresh, 1e-6), 1.0) * span
    return np.array([np.clip(kp * (e - margin) / ACT, -1, 1)], dtype=np.float32)


# ---------------- evaluation ----------------

def run(policy, fn, d_des, n_ep=20, seed0=2000):
    fe.D_DES = d_des
    env = ScriptedLeader(domain_randomize=False)
    try:
        coll, errs = 0, []
        for k in range(n_ep):
            if hasattr(policy, "reset"):
                policy.reset()
            obs, _ = env.reset(seed=seed0 + k)
            env.set_fn(fn)
            while True:
                obs, r, term, trunc, _ = env.step(policy(obs))
                if term or trunc:
                    break
            g = np.array(env.log["gap"])
            s = g[int(len(g) * 0.2):]
            errs.append(np.abs(s - d_des).mean() * 200)
            if env.term_reason == "collision":
                coll += 1
        return coll, float(np.mean(errs))
    finally:
        fe.D_DES = 0.20


if __name__ == "__main__":
    # probe attacks from the plan
    probes = {
        "step-down 0.1s": lambda t: 0.95 if int(t / 0.1) % 2 == 0 else 0.05,
        "step-down 0.2s": lambda t: 0.95 if int(t / 0.2) % 2 == 0 else 0.05,
        "step-down 0.4s": lambda t: 0.95 if int(t / 0.4) % 2 == 0 else 0.05,
        "osc 0.5Hz": lambda t: 0.5 + 0.45 * np.sin(np.pi * t),
        "osc 0.3Hz": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * 0.3 * t),
        "chirp": lambda t: 0.5 + 0.45 * np.sin(2 * np.pi * (0.1 + 0.25 * min(t, 4.0)) * t),
        "nominal": lambda t: 0.5,
    }
    cands = {
        "P+FF (ref)": p_ff,
        "yield_inst": yield_inst,
        "yield_st_m0.55": YieldStateful(),
        "sched_gap": sched_gap,
    }

    print("=" * 96)
    print("POSITIVE-POLE CANDIDATE SEARCH  (coll/20 and settled err obs-cm, d=0.20)")
    print("=" * 96)
    for name, pol in cands.items():
        print(f"\n  --- {name} ---")
        for aname, fn in probes.items():
            c, e = run(pol, fn, 0.20, n_ep=20)
            print(f"    {aname:>16s}  coll {c:>2d}/20   settled err {e:>6.2f}")
