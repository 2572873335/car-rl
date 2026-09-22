"""Step 0.5 feasibility gate for the regime-extension retrain (plan section 8).

The owner ruling asks to widen v_set to U(0.25, 1.50) and warns that a leader
peak above the follower's V_MAX=1.3 would make the task structurally unsolvable.
That premise needs checking against the actual code, because _leader_v() ends in
np.clip(v, 0.05, 1.0) -- the LEADER's speed is capped at 1.0 m/s regardless of
v_set; V_MAX=1.3 is the FOLLOWER's command cap (follow_env.py:163).

This probe measures what v_set ~ U(0.25,1.50) actually produces as leader speed,
so the retrain is not launched on a mis-stated premise.
"""
import sys

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
while "/tmp" in sys.path:
    sys.path.remove("/tmp")

import numpy as np
import follow_env as fe

print("=" * 84)
print("STEP 0.5: what leader speed does v_set ~ U(0.25, 1.50) actually produce?")
print("=" * 84)
print(f"  leader clip in _leader_v(): np.clip(v, 0.05, 1.0)   [read from source]")
print(f"  follower command cap V_MAX = {fe.V_MAX}")
print()

# sample v_set the way reset() does, but over the PROPOSED extended range
rng = np.random.default_rng(0)
N = 200_000
v_set = rng.uniform(0.25, 1.50, N)


def leader_peak(v, amp=0.4):
    """Peak leader speed the env can emit for a given v_set (sinusoid = worst case)."""
    return min(v * (1.0 + amp), 1.0)          # _leader_v clips at 1.0


peaks = np.array([leader_peak(v) for v in v_set])
print(f"  proposed v_set ~ U(0.25, 1.50), n={N}")
print(f"    fraction of samples with v_set > 1.0 (clipped away): "
      f"{(v_set > 1.0).mean():.1%}")
print(f"    leader peak speed:  min={peaks.min():.2f}  max={peaks.max():.2f}  "
      f"mean={peaks.mean():.2f}")
print(f"    fraction of samples pinned AT the 1.00 ceiling: {(peaks >= 1.0 - 1e-9).mean():.1%}")
print(f"    any leader peak > V_MAX ({fe.V_MAX})? {(peaks > fe.V_MAX).any()}")
print()
print(f"  => the owner's premise (leader peak > 1.3 -> unsolvable) cannot occur:")
print(f"     the leader is capped at 1.0, strictly below V_MAX={fe.V_MAX}.")
print()
print(f"  BUT a real design flaw appears: {(v_set > 1.0).mean():.0%} of the proposed")
print(f"     distribution collapses onto EXACTLY 1.0 m/s (a point mass), because")
print(f"     v_set above 1.0 is indistinguishable after the clip. That is not a")
print(f"     smooth extension to 1.5 -- it is U(0.25,1.0) plus a 40% spike at 1.0.")
print()
print("  effective distribution actually trained (constant behavior):")
hist, edges = np.histogram(peaks, bins=10, range=(0.2, 1.05))
for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
    bar = "#" * int(60 * h / hist.max())
    print(f"    [{e0:.2f},{e1:.2f}) {h/N:6.2%} {bar}")
