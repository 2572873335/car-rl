"""W6 criterion v5: SCRIPTED BLOCKER as the opponent (non-circular).

Rationale (user's ruling): M1 is the gate for "can the training pipeline learn
something". If the gate's own criterion needs a TRAINED adversary, the gate
becomes circular -- validating the pipeline with its own product. A scripted
blocker breaks that circularity while preserving the point: blocking makes
geometric lane-taking stop being free, which is exactly RQ3's subject.

World model: reuse the wrap-safe two-car world (WorldV4). Car 0 = follower
(the policy under test). Car 1 = the scripted blocker, cruising at a fixed
speed, with a pure-geometry rule for the lane.

Blocker rule (no speed cleverness, pure geometry):
  - default: outer lane cruise
  - if the follower is behind within D metres AND the follower is NOT already
    ahead, move to the inner lane to deny the entry
  - otherwise hold outer

This file first MEASURES what the geometry family does against it. The counter
script (the positive control) is developed separately, from observed behaviour.
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


class BlockerWorld:
    """Car 0 = follower (policy under test). Car 1 = scripted blocker."""

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
        s_f = self.outer.nearest(self.cars[0].pos)[0]
        s_l = self.outer.nearest(self.cars[1].pos)[0]
        fwd = (s_l - s_f) % self.L
        self.prog = [s_l - fwd, s_l]
        self.prev_s = [s_f, s_l]
        self.t, self.done, self.reason = 0.0, False, "timeout"
        self.t_first_ahead = np.nan
        self.min_after_ahead = np.inf
        self.blocker_switches = 0
        return self.obs()

    def _advance_progress(self):
        for i in range(2):
            s = self.outer.nearest(self.cars[i].pos)[0]
            self.prog[i] += wrapL(s - self.prev_s[i], self.L)
            self.prev_s[i] = s

    def delta_true(self):
        return self.prog[0] - self.prog[1]

    def gap_forward(self):
        """Along-track distance from follower(car0) to blocker(car1), wrapped."""
        s_f = self.outer.nearest(self.cars[0].pos)[0]
        s_l = self.outer.nearest(self.cars[1].pos)[0]
        return (s_l - s_f) % self.L

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

    def blocker_action(self, block_dist=0.60):
        """Scripted blocker: pure geometry, no speed cleverness.
        Deny the inner lane when the follower is close behind.
        Returns (a_speed_unused, a_lane) for car 1.
        """
        g = self.gap_forward()
        behind_close = g < block_dist
        if behind_close:
            return np.array([0.0, 1.0])      # move to inner: deny entry
        return np.array([0.0, -1.0])          # return to outer

    def step(self, a0, blocker_a=None):
        if blocker_a is None:
            blocker_a = self.blocker_action()
        acts = np.stack([np.asarray(a0, np.float64), np.asarray(blocker_a, np.float64)])
        for i in range(2):
            _, a_ln = np.clip(acts[i], -1, 1)
            self.lane_cd[i] = max(0.0, self.lane_cd[i] - DT)
            tgt = 1 if a_ln > 0.3 else (0 if a_ln < -0.3 else self.lane[i])
            if tgt != self.lane[i] and self.lane_cd[i] <= 0:
                self.lane[i] = tgt
                self.lane_cd[i] = SWITCH_COOLDOWN
                self.switch[i] = 0.8
                if i == 1:
                    self.blocker_switches += 1
        for i in range(2):
            j = 1 - i
            a_sp = float(np.clip(acts[i][0], -1, 1))
            if i == 1:
                v_cmd = float(np.clip(self.lead_v, 0.05, V_MAX))   # cruise
            else:
                v_cmd = float(np.clip(self.cars[j].v + ACT_GAIN * a_sp,
                                      0.0, V_MAX))
            lp = self.inner if self.lane[i] == 1 else self.outer
            if self.switch[i] > 0:
                self.switch[i] = max(0.0, self.switch[i] - DT)
                v_cmd = min(v_cmd, 0.6)
            self.cars[i].step(v_cmd, self.steer[i].omega_cmd(self.cars[i], lp), DT)
        self.t += DT
        self._advance_progress()

        d = self.delta_true()
        if d > 0.30 and np.isnan(self.t_first_ahead):
            self.t_first_ahead = self.t
        if not np.isnan(self.t_first_ahead):
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
        return self.done

    def complete(self):
        if np.isnan(self.t_first_ahead):
            return 0.0
        return 1.0 if self.min_after_ahead > -0.10 else 0.0


def run(pol, seed0, n_ep=20, lo=0.15, hi=0.30, block_dist=0.60):
    outer, inner = build_paths()
    comp, coll, ts, sw = [], 0, [], []
    for k in range(n_ep):
        w = BlockerWorld(outer, inner, lo, hi)
        w.reset(np.random.default_rng(seed0 + k))
        while True:
            obs = w.obs()
            done = w.step(pol(obs[0], w.L))
            if done:
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
        sw.append(w.blocker_switches)
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"), float(np.mean(sw)))


def main():
    from stable_baselines3 import PPO
    from _ckpt_as_opponent import frozen_layout
    from overtake_env import baseline_action_ot

    ckpt = PPO.load(f"{REPO}/ckpt_ot/overtake_final_v1.zip")
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))
    rng = np.random.default_rng(11)

    def geom(o, L):
        d = o[0] * 2.5
        return np.array([0.0, 1.0 if d < 0.30 else -1.0])

    def geom_dive_early(o, L):
        d = o[0] * 2.5
        return np.array([1.0, 1.0 if d < 0.60 else -1.0])

    def geom_smart(o, L):
        d = o[0] * 2.5
        return np.array([1.0 if d < 0.45 else -0.5,
                         1.0 if d < 0.45 else -1.0])

    pols = [
        ("zero-action    ", lambda o, L: np.zeros(2)),
        ("geom 2-line    ", geom),
        ("geom_dive_early", geom_dive_early),
        ("geom_smart     ", geom_smart),
        ("rule machine   ", rule),
        ("ckpt (+)       ", ck),
    ]

    print("=" * 78)
    print("SCRIPTED BLOCKER vs the policies (block_dist=0.60, 20 eps)")
    print("=" * 78)
    for bd in (0.45, 0.60, 0.90):
        print(f"\n  block_dist = {bd:.2f} m")
        print(f"    {'policy':>16s} {'complete':>9s} {'coll%':>7s} "
              f"{'med_t':>7s} {'blk_sw':>7s}")
        for name, fn in pols:
            c, cl, t, s = run(fn, 123000, n_ep=20, block_dist=bd)
            print(f"    {name:>16s} {c:>9.2f} {cl:>6.0%} {t:>7.2f} {s:>7.1f}")


if __name__ == "__main__":
    main()
