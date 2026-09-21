"""Pin down the rule-machine mismatch precisely.

Observation from the instrumented episode: the rule machine as follower
accelerates to v=1.30 while the leader holds 0.58, ramming it at t=0.98.

The rule machine was authored against the FROZEN env, where:
    v_cmd = v_l + 0.8*a      (v_l = leader speed)
and its states emit a_speed in {-0.6, 0.4, 0.7, P-approach}.

My world's V2V prior is v_cmd = v_OTHER + 0.8*a -- the same form, so that is
NOT the mismatch. The mismatch must be the OBSERVATION the rule machine reads:
baseline_action_ot decodes gap/delta/lane from specific slots. Through
`frozen_layout` I feed gap = (-delta)%L which is the distance to the other car,
so as the follower closes on the leader, gap SHRINKS and the rule should brake
(gap < 0.15 -> a_speed = -0.6).

But observed v0 climbs to 1.30, meaning it took the P-approach branch
(a_speed = clip(gap - 0.2, -1, 1)). At gap=0.382 that gives 0.18 -> but with
V2V prior v = 0.58 + 0.8*0.18 = 0.72, not 1.30.

So the ACTUAL reference in my world is not the leader. Check: is the follower
being given ref = cars[j].v where j = 1 - i? For i=0, j=1 -> the LEADER. That
is correct...

Unless -- the collision is with the leader while the follower sits on the
OUTER lane and the leader too, and the closure comes from the follower's
initial speed being LOW (0.03) then catching up. v=1.30 means a_speed ~ 1.0.
Where does a_speed reach 1.0? In the dive branch: `if lane == 0 and gap <
0.45: a_speed, a_lane = 0.4, 1.0`. That sets a_LANE=1.0, a_speed=0.4 ->
v = v_l + 0.8*0.4 = 0.90. Still not 1.30.

Let me just measure what the rule machine outputs step by step in my world.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _criterion_v3 import WorldV3
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot, ACT_GAIN, V_MAX

ZERO2 = np.zeros(2)


def main():
    outer, inner = build_paths()
    w = WorldV3(outer, inner, prior="v2v")
    obs = w.reset(np.random.default_rng(123000))

    print("frozen_layout output vs what the rule machine decodes")
    print(f"  {'t':>6s} {'gap_slot':>9s} {'decoded gap':>12s} {'decoded d':>10s} "
          f"{'lane':>5s} {'a_speed':>8s} {'a_lane':>7s} {'ref v':>7s} {'v_cmd':>7s}")
    for k in range(60):
        o1 = obs[0]
        fl = frozen_layout(o1, w.L)
        gap_dec = fl[0] * 0.5 + 0.2
        d_dec = fl[1] * 2.5
        lane_dec = int(round(fl[4]))
        a = baseline_action_ot(fl)
        ref = w.cars[1].v
        v_cmd = float(np.clip(ref + ACT_GAIN * a[0], 0.0, V_MAX))
        if k % 5 == 0:
            print(f"  {w.t:>6.2f} {fl[0]:>9.2f} {gap_dec:>12.3f} {d_dec:>10.3f} "
                  f"{lane_dec:>5d} {a[0]:>8.2f} {a[1]:>7.2f} {ref:>7.2f} "
                  f"{v_cmd:>7.2f}")
        obs, done = w.step(np.stack([a, ZERO2]), leader_action=ZERO2)
        if done:
            print(f"  -> {w.reason} at t={w.t:.2f}; car0 v={w.cars[0].v:.2f}, "
                  f"car1 v={w.cars[1].v:.2f}")
            break

    print()
    print("DIAGNOSIS")
    print("  The rule machine reads 'gap' = distance to the OTHER car. As the")
    print("  follower approaches, gap SHRINKS -> it should brake (gap<0.15) or")
    print("  dive (gap<0.45). If instead it accelerates, the decoded gap is")
    print("  NOT the true closing distance -- i.e. frozen_layout is still wrong")
    print("  for this world's obs convention (self-view vs follower-view).")


if __name__ == "__main__":
    main()
