"""Blocker v3: separate COMMANDED lane from PHYSICAL lane occupancy.

Measured fact (see _measure_transient.py): a lane command already produces a
real ~0.86 s physical transition, because PurePursuit has to steer the body
across. But the model updated the logical `lane` variable INSTANTLY, so:
  - the blocker "occupied" the inner lane the moment it decided to, and
  - denial was decided on the flag rather than on where the body actually was.

Fix: keep lane_target (commanded) and lane (physical). The physical lane
updates only when the body has arrived on the target path. Collision continues
to use true 2D distance, so a car mid-transition is where it physically is.

This does NOT weaken the blocker's rule -- it only makes its occupancy honest.
The regression check (geometry family must still score 0.00) enforces that.
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
ARRIVE_TOL = 0.03          # |e_lat| on target path at which lane is "reached"


class BlockerWorld3:
    """Car 0 = follower (under test). Car 1 = scripted blocker.
    Physical lane occupancy is tracked separately from the lane command.
    """

    def __init__(self, outer, inner, cruise_v=0.55, guard_hi=0.90,
                 guard_lo=0.08):
        self.outer, self.inner = outer, inner
        self.L = outer.length
        self.cruise_v = cruise_v
        self.guard_hi, self.guard_lo = guard_hi, guard_lo

    def reset(self, rng):
        self.rng = rng
        self.gap0 = float(rng.uniform(0.40, 0.90))
        s_A = self.outer.nearest(A_START)[0]
        self.cars, self.steer = [], []
        self.lane = [0, 0]            # physical occupancy
        self.lane_target = [0, 0]     # commanded
        self.lane_cd = [0.0, 0.0]
        self.switch = [0.0, 0.0]
        for i, off in [(0, -self.gap0), (1, 0.0)]:
            s = (s_A + off) % self.L
            p, t = self.outer.point_at(s), self.outer.tan[0]
            self.cars.append(Car(*p, float(np.arctan2(t[1], t[0]))))
            self.steer.append(PurePursuit())
        s_f = self.outer.nearest(self.cars[0].pos)[0]
        s_l = self.outer.nearest(self.cars[1].pos)[0]
        fwd = (s_l - s_f) % self.L
        self.prog = [s_l - fwd, s_l]
        self.prev_s = [s_f, s_l]
        self.t, self.done, self.reason = 0.0, False, "timeout"
        self.t_first_ahead = np.nan
        self.min_after_ahead = np.inf
        self.transition_time = []      # measured physical transition durations
        self._trans_start = [None, None]
        return self.obs()

    # ---------------------------------------------------------------- state --
    def _advance_progress(self):
        for i in range(2):
            s = self.outer.nearest(self.cars[i].pos)[0]
            self.prog[i] += wrapL(s - self.prev_s[i], self.L)
            self.prev_s[i] = s

    def delta_true(self):
        return self.prog[0] - self.prog[1]

    def gap_forward(self):
        s_f = self.outer.nearest(self.cars[0].pos)[0]
        s_l = self.outer.nearest(self.cars[1].pos)[0]
        return (s_l - s_f) % self.L

    def _phys_arrived(self, i):
        """True when car i's body is on its target path."""
        tgt = self.inner if self.lane_target[i] == 1 else self.outer
        return abs(tgt.nearest(self.cars[i].pos)[1]) < ARRIVE_TOL

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

    def blocker_action(self):
        """Command the contested lane. Decision uses PHYSICAL occupancy."""
        g = self.gap_forward()
        d = self.delta_true()
        follower_behind = d < 0.0
        in_band = self.guard_lo < g < self.guard_hi
        if follower_behind and in_band:
            return np.array([0.0, 1.0])
        return np.array([0.0, -1.0])

    # ----------------------------------------------------------------- step --
    def step(self, a0, blocker_a=None):
        if blocker_a is None:
            blocker_a = self.blocker_action()
        acts = np.stack([np.asarray(a0, np.float64),
                         np.asarray(blocker_a, np.float64)])

        # 1. lane COMMANDS (start a transition if the target differs)
        for i in range(2):
            _, a_ln = np.clip(acts[i], -1, 1)
            self.lane_cd[i] = max(0.0, self.lane_cd[i] - DT)
            tgt = 1 if a_ln > 0.3 else (0 if a_ln < -0.3 else self.lane_target[i])
            if tgt != self.lane_target[i] and self.lane_cd[i] <= 0:
                self.lane_target[i] = tgt
                self.lane_cd[i] = SWITCH_COOLDOWN
                self.switch[i] = 0.8
                self._trans_start[i] = self.t

        # 2. physical lane updates only on arrival
        for i in range(2):
            if self.lane[i] != self.lane_target[i] and self._phys_arrived(i):
                self.lane[i] = self.lane_target[i]
                if self._trans_start[i] is not None:
                    self.transition_time.append(self.t - self._trans_start[i])
                    self._trans_start[i] = None

        # 3. motion: steer toward the COMMANDED path (the body chases it)
        for i in range(2):
            j = 1 - i
            a_sp = float(np.clip(acts[i][0], -1, 1))
            if i == 1:
                v_cmd = self.cruise_v
            else:
                v_cmd = float(np.clip(self.cars[j].v + ACT_GAIN * a_sp,
                                      0.0, V_MAX))
            lp = self.inner if self.lane_target[i] == 1 else self.outer
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


def run(pol, seed0, n_ep=20, cruise_v=0.55, guard_hi=0.90, guard_lo=0.08,
        counter=False):
    outer, inner = build_paths()
    comp, coll, ts, trans = [], 0, [], []
    for k in range(n_ep):
        w = BlockerWorld3(outer, inner, cruise_v=cruise_v,
                          guard_hi=guard_hi, guard_lo=guard_lo)
        w.reset(np.random.default_rng(seed0 + k))
        while True:
            if counter:
                from _counter3 import counter_action
                a0 = counter_action(w)
            else:
                a0 = pol(w.obs()[0], w.L)
            if w.step(a0):
                break
        comp.append(w.complete())
        if w.reason == "collision":
            coll += 1
        if not np.isnan(w.t_first_ahead):
            ts.append(w.t_first_ahead)
        trans += w.transition_time
    return (float(np.mean(comp)), coll / n_ep,
            float(np.median(ts)) if ts else float("nan"),
            float(np.mean(trans)) if trans else float("nan"))


def main():
    from stable_baselines3 import PPO
    from _ckpt_as_opponent import frozen_layout
    from overtake_env import baseline_action_ot

    ckpt = PPO.load(f"{REPO}/ckpt_ot/overtake_final_v1.zip")
    ck = lambda o, L: ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
    rule = lambda o, L: baseline_action_ot(frozen_layout(o, L))

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

    print("=" * 80)
    print("REGRESSION (must stay 0.00): geometry family vs blocker v3")
    print("  blocker NOT weakened -- physical lane occupancy is honest, rule same")
    print("=" * 80)
    print(f"  {'policy':>16s} {'complete':>9s} {'coll%':>7s} {'med_t':>7s} "
          f"{'trans_s':>8s}")
    for name, fn in [("geom 2-line", geom), ("geom_dive_early", geom_dive_early),
                     ("geom_smart", geom_smart), ("rule machine", rule),
                     ("ckpt (+)", ck)]:
        c, cl, t, tr = run(fn, 123000, n_ep=20)
        print(f"  {name:>16s} {c:>9.2f} {cl:>6.0%} {t:>7.2f} {tr:>8.2f}")


if __name__ == "__main__":
    main()
