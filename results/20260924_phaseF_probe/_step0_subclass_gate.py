"""Can a learned adversary drive the leader WITHOUT editing follow_env.py?

The plan needs the leader (the "attacker") to be a learned policy rather than the
frozen scripted function. follow_env.step() calls self._leader_v() internally and
moves the leader with it.

If _leader_v() is an ordinary method, a subclass can override it to return the
attacker's commanded speed -- leaving follow_env.py byte-identical (iron rule 2).
This probe verifies that mechanism end-to-end, including that the frozen file's
hash is unchanged.

Run: uv run python results/20260924_phaseF_probe/_step0_subclass_gate.py
"""
import hashlib
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe


class AdversaryLeaderEnv(fe.FollowEnv):
    """Follower task with an externally-commanded leader (the attacker).

    The attacker's action is stashed on the instance; _leader_v() returns it.
    Everything else -- observation, reward, termination, the V2V prior -- is the
    frozen implementation, unchanged.
    """

    def set_attacker_action(self, a):
        # attacker action in [-1, 1] -> leader speed command
        self._att_action = float(np.clip(a, -1.0, 1.0))

    def _leader_v(self):
        # same range the frozen scripted leader can emit (0.05 .. 1.0),
        # so the follower's V2V prior stays in-distribution in form
        base = getattr(self, "_att_base", 0.3)
        a = getattr(self, "_att_action", 0.0)
        return float(np.clip(base + 0.8 * a, 0.05, 1.0))


def main():
    h0 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()

    env = AdversaryLeaderEnv(domain_randomize=False)
    obs, _ = env.reset(seed=2000)
    env._att_base = 0.30

    speeds = []
    for k in range(400):
        # drive the attacker to a fixed sinusoidal command
        a = 0.6 * np.sin(2 * np.pi * 0.1 * k * fe.DT)
        env.set_attacker_action(a)
        obs, r, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
        speeds.append(env._leader_v())
        if term or trunc:
            break

    speeds = np.array(speeds)
    print("=" * 84)
    print("STEP 0 GATE: adversarial leader via subclass (frozen file untouched)")
    print("=" * 84)
    print(f"  leader command range emitted: {speeds.min():.3f} .. {speeds.max():.3f}")
    print(f"  distinct values (>1 means it really varies): {len(np.unique(np.round(speeds,4)))}")
    print(f"  episodes ran, steps: {k+1}")

    h1 = hashlib.sha256(open(f"{REPO}/follow_env.py", "rb").read()).hexdigest()
    print(f"\n  follow_env.py sha256[:16] before: {h0[:16]}")
    print(f"  follow_env.py sha256[:16] after : {h1[:16]}")
    print(f"  frozen file untouched: {h0 == h1}")
    print()
    ok = (h0 == h1) and (speeds.max() - speeds.min() > 0.2)
    print(f"  GATE: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
