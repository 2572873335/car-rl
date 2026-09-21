"""Criterion v4: wrap-safe manoeuvre completion + collision gate + t_overtake.

Fixes from review2:
  R1 -- FULL was defeated by a two-line geometry policy (both scored 1.00),
        because "re-entered the outer lane" is free for anyone already ahead.
        The discriminating quantity in the data is t_overtake:
            positive ckpt 1.6 s | rule machine 2.6 s | geom 19.4 s
  R2 -- FULL was decoupled from collisions (FULL=1.00 with 97% collisions).
        Add an explicit collision gate.
  R4 -- held_after_return treated a LAP WRAP as losing the lead. The frozen env
        guards this (overtake_env.py:229-231: ahead_flag never resets). Fix by
        measuring UNWRAPPED cumulative progress instead of wrapped delta.

Design:
  progress = cumulative arc length travelled (unwrapped, so lapping is fine)
  delta_true = progress_follower - progress_leader
  COMPLETE  = delta_true reached > 0.3 at some point AND stayed > -0.1 after
  metric    = (completion rate, collision rate, median t_overtake)
"""
import os
import sys

REPO = "/home/zy/car_rl/code0919"
SCRIPTS = f"{REPO}/results/20260920_phaseC_probe/scripts"
while "/tmp" in sys.path:
    sys.path.remove("/tmp")
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)

import numpy as np
from car_following_sim import Car, PurePursuit, A_START, DT
from _selfplay_design_probe import build_paths, wrapL

V_MAX, ACT_GAIN = 1.3, 0.8
COLLISION_2D, OFFTRACK, SWITCH_COOLDOWN, T_MAX = 0.12, 0.4, 1.5, 30.0
ZERO2 = np.zeros(2)


class WorldV4:
    """Role-fixed follower/leader with UNWRAPPED progress tracking."""

    def __init__(self, outer, inner, lead_lo=0.15, lead_hi=0.30):
        self.outer, self.inner = outer, inner
        self.L = outer.length
        self.lead_lo, self.lead_hi = lead_lo, lead_hi

    def reset(self, rng):
        self.rng = rng
        self.gap0 = float(rng.uniform(0.30, 0.90))
        self.lead_v = float(rng.uniform(self.lead_lo, self.lead_hi))
        s_A = self.outer.nearest(A_START)[0]
        self.cars, self.steer, self.lane = [], [], []
        self.lane_cd = [0.0, 0.0]
        self.switch = [0.0, 0.0]
        for i, off in [(0, -self.gap0), (1, 0.0)]:
            s = (s_A + off) % self.L
            p, t = self.outer.point_at(s), self.outer.tan[0]
            self.cars.append(Car(*p, float(np.arctan2(t[1], t[0]))))
            self.steer.append(PurePursuit())
            self.lane.append(0)
        # unwrapped cumulative arc length per car.
        # NOTE: raw wrapped arclengths cannot be used directly -- the follower
        # starts BEHIND at (s_A - gap), which wraps to a numerically LARGER s
        # than the leader, making delta_true ~ +L at t=0 and marking every
        # policy as "ahead" instantly (a real bug caught by the zero-action
        # control). Anchor the leader at its s and place the follower the true
        # wrapped distance behind it.
        s_f = self.outer.nearest(self.cars[0].pos)[0]
        s_l = self.outer.nearest(self.cars[1].pos)[0]
        forward_gap = (s_l - s_f) % self.L      # positive: follower is behind
        self.prog = [s_l - forward_gap, s_l]     # delta_true = -forward_gap
        self.prev_s = [s_f, s_l]
        self.t, self.done, self.reason = 0.0, False, "timeout"
        self.t_first_ahead = np.nan
        self.min_after_ahead = np.inf
        self.ever_ahead = False
        return self.obs()

    def _advance_progress(self):
        for i in range(2):
            s = self.outer.nearest(self.cars[i].pos)[0]
            ds = wrapL(s - self.prev_s[i], self.L)   # signed, unwrapped step
            self.prog[i] += ds
            self.prev_s[i] = s

    def delta_true(self):
        return self.prog[0] - self.prog[1]

    def obs(self):
        out = []
        for i in range(2):
            j = 1 - i
            d = wrapL(self.outer.nearest(self.cars[j].pos)[0]
                      - self.outer.nearest(self.cars[i].pos)[0], self.L)
            lp = self.inner if self.lane[i] == 1 else self.outer
            e_lat = lp.nearest(self.cars[i].pos)[1]
            out.append(np.array([-d / 2.5, self.cars[i].v / V_MAX,
                                 self.cars[j].v / V_MAX, e_lat / 0.25,
                                 float(self.lane[i]),
                                 (self.L / 2 - abs(d)) / 2.5], np.float32))
        return np.stack(out)

    def step(self, acts, leader_action=None):
        acts = np.asarray(acts, np.float64).reshape(2, 2)
        for i in range(2):
            _, a_ln = np.clip(acts[i], -1, 1)
            self.lane_cd[i] = max(0.0, self.lane_cd[i] - DT)
            tgt = 1 if a_ln > 0.3 else (0 if a_ln < -0.3 else self.lane[i])
            if tgt != self.lane[i] and self.lane_cd[i] <= 0:
                self.lane[i] = tgt
                self.lane_cd[i] = SWITCH_COOLDOWN
                self.switch[i] = 0.8
        for i in range(2):
            j = 1 - i
            a_sp = float(np.clip(acts[i][0], -1, 1))
            if i == 1 and leader_action is not None:
                v_cmd = float(np.clip(self.lead_v, 0.05, V_MAX))
            else:
                v_cmd = float(np.clip(self.cars[j].v + ACT_GAIN * a_sp,
                                      0.0, V_MAX))
            lp = self.inner if self.lane[i] == 1 else self.outer
            if self.switch[i] > 0:
                self.switch[i] = max(0.0, self.switch[i] - DT)
                v_cmd = min(v_cmd, 0.6)
            self.cars[i].step(v_cmd, self.steer[i].omega_cmd(self.cars[i], lp),
                              DT)
        self.t += DT
        self._advance_progress()

        d = self.delta_true()
        if d > 0.30 and np.isnan(self.t_first_ahead):
            self.t_first_ahead = self.t
        if not np.isnan(self.t_first_ahead):
            self.ever_ahead = True
            self.min_after_ahead = min(self.min_after_ahead, d)

        coll = np.hypot(*(self.cars[0].pos - self.cars[1].pos)) < COLLISION_2D
        off = any(abs((self.inner if self.lane[i] == 1 else self.outer)
                      .nearest(self.cars[i].pos)[1]) > OFFTRACK for i in range(2))
        if coll:
            self.reason, self.done = "collision", True
        elif off:
            self.reason, self.done = "offtrack", True
        elif self.t >= T_MAX:
            self.reason, self.done = "timeout", True
        return self.obs(), self.done

    def complete(self):
        """Wrap-safe: overtook AND never lost the lead afterwards."""
        if np.isnan(self.t_first_ahead):
            return 0.0
        return 1.0 if self.min_after_ahead > -0.10 else 0.0


def run(pol, seed0, n_ep=30, lo=0.15, hi=0.30):
    outer, inner = build_paths()
    comp, coll, ts = [], 0, []
    for k in range(n_ep):
        w = WorldV4(outer, inner, lo, hi)
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            obs, done = w.step(np.stack([pol(obs[0], w.L), ZERO2]),
                               leader_action=ZERO2)
            if done:
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"))


def table(seed0=123000, label="leader band 0.15-0.30"):
    from stable_baselines3 import PPO
    from _ckpt_as_opponent import frozen_layout
    from overtake_env import baseline_action_ot
    ckpt = PPO.load(f"{REPO}/ckpt_ot/overtake_final_v1.zip")
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    def geom(o, L):
        d = o[0] * 2.5
        return np.array([0.0, 1.0 if d < 0.30 else -1.0])

    def geom_slow(o, L):
        d = o[0] * 2.5
        return np.array([-0.3, 1.0 if d < 0.30 else -1.0])

    def geom_fast(o, L):
        d = o[0] * 2.5
        return np.array([1.0, 1.0 if d < 0.30 else -1.0])

    rng = np.random.default_rng(5)
    pols = [
        ("ckpt (+)        ", ck),
        ("rule machine    ", rule),
        ("geom 2-line     ", geom),
        ("geom_slow       ", geom_slow),
        ("geom_fast       ", geom_fast),
        ("loiter          ", loiter),
        ("zero-action     ", lambda o, L: np.zeros(2)),
        ("full-throttle   ", lambda o, L: np.array([1.0, 0.0])),
        ("random          ", lambda o, L: rng.uniform(-1, 1, 2)),
    ]

    print("=" * 76)
    print(f"CRITERION v4 ({label}, seed0={seed0}, 30 eps)")
    print("  completion = wrap-safe overtake + never lost lead; + collision gate")
    print("=" * 76)
    print(f"  {'policy':>17s} {'COMPLETE':>9s} {'coll%':>7s} {'med_t_ot':>9s} "
          f"{'accept?':>8s}")
    print("-" * 76)
    for name, fn in pols:
        c, cl, t = run(fn, seed0)
        # acceptance: complete AND low collision
        accept = "PASS" if (c >= 0.8 and cl <= 0.10) else "fail"
        print(f"  {name:>17s} {c:>9.2f} {cl:>6.0%} {t:>9.2f} {accept:>8s}")


if __name__ == "__main__":
    for s in (123000, 200000, 300000):
        table(s)
        print()
