"""Criterion rework (review2 input): world v2 with REAL randomness, and a
comparison of candidate dependent variables including the loiter adversary.

Fixes attempted:
  B2 -> reset() now samples gap0, opponent speed, and opponent phase from rng,
        so deterministic policies produce MANY distinct outcomes.
  B3 -> delta is wrap-saturated (+-L/2) so it cannot be the dependent variable.
        Test three candidates against a degenerate set that INCLUDES loiter:
          M_delta : final delta > thr                (current, known-bad)
          M_time  : time to first reach delta > eps  (task-level; lower better)
          M_task  : completed an overtake AND returned to following
                    (mirrors the contest task, not just "be ahead")
Positives to test: the frozen checkpoint (out of distribution here) and, if
time allows, a freshly trained in-world PPO.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from car_following_sim import Car, PurePursuit, A_START, DT
from _selfplay_design_probe import build_paths, wrapL
from _ckpt_as_opponent import frozen_layout          # now FIXED
from overtake_env import baseline_action_ot

V_MAX, ACT_GAIN = 1.3, 0.8
COLLISION_2D, OFFTRACK, SWITCH_COOLDOWN, T_MAX = 0.12, 0.4, 1.5, 30.0
OVERTAKE_EPS = 0.30      # frozen env's success margin (overtake_env.py:263)


class WorldV2:
    """Role-fixed: car 0 = follower (starts behind), car 1 = leader.
    NOW WITH REAL RANDOMNESS (B2 fix).
    """

    def __init__(self, outer, inner, prior="v2v"):
        self.outer, self.inner, self.prior = outer, inner, prior
        self.L = outer.length

    def reset(self, rng):
        self.rng = rng
        # --- B2 FIX: real randomisation ---
        self.gap0 = float(rng.uniform(0.30, 0.90))      # behind distance
        self.lead_v = float(rng.uniform(0.40, 0.80))    # leader cruise speed
        self.lead_bias = float(rng.uniform(-0.15, 0.15))  # leader speed jitter
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
        self.t, self.done, self.reason = 0.0, False, "timeout"
        self.t_first_ahead = np.nan
        self.min_delta_after_ahead = np.inf
        return self.obs()

    def _s(self, car):
        return self.outer.nearest(car.pos)[0]

    def delta(self, i):
        j = 1 - i
        return -wrapL(self._s(self.cars[j]) - self._s(self.cars[i]), self.L)

    def obs(self):
        out = []
        for i in range(2):
            j = 1 - i
            d = wrapL(self._s(self.cars[j]) - self._s(self.cars[i]), self.L)
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
                # leader is SCRIPTED at a cruising speed (not the rule machine):
                # its command holds lead_v with jitter
                v_cmd = float(np.clip(self.lead_v + self.lead_bias * a_sp,
                                      0.05, V_MAX))
            else:
                ref = self.cars[j].v if self.prior == "v2v" else self.cars[i].v
                v_cmd = float(np.clip(ref + ACT_GAIN * a_sp, 0.0, V_MAX))
            lp = self.inner if self.lane[i] == 1 else self.outer
            if self.switch[i] > 0:
                self.switch[i] = max(0.0, self.switch[i] - DT)
                v_cmd = min(v_cmd, 0.6)
            self.cars[i].step(v_cmd, self.steer[i].omega_cmd(self.cars[i], lp), DT)
        self.t += DT

        d = self.delta(0)
        if np.isnan(self.t_first_ahead) and d > OVERTAKE_EPS:
            self.t_first_ahead = self.t
        if not np.isnan(self.t_first_ahead):
            self.min_delta_after_ahead = min(self.min_delta_after_ahead, d)

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

    # ---- the three candidate metrics ----
    def m_delta(self):
        return self.delta(0)

    def m_time(self):
        """Time to complete the overtake; inf if never. LOWER is better."""
        return self.t_first_ahead

    def m_task(self):
        """Completed an overtake AND stayed ahead afterwards (did not fall
        back behind). Mirrors the contest task: pass, then hold the lead."""
        if np.isnan(self.t_first_ahead):
            return 0.0
        return 1.0 if self.min_delta_after_ahead > 0.0 else 0.0


def run(pol, n_ep=40, seed0=123000):
    outer, inner = build_paths()
    deltas, times, tasks, colls, reasons = [], [], [], 0, {}
    for k in range(n_ep):
        w = WorldV2(outer, inner, prior="v2v")
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0 = pol(obs[0], w.L)
            obs, done = w.step(np.stack([a0, np.zeros(2)]))
            if done:
                break
        deltas.append(w.m_delta())
        times.append(w.m_time())
        tasks.append(w.m_task())
        reasons[w.reason] = reasons.get(w.reason, 0) + 1
        if w.reason == "collision":
            colls += 1
    return (np.array(deltas), np.array(times), np.array(tasks),
            colls / n_ep, reasons)


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    pols = [
        ("zero-action", lambda o, L: np.zeros(2)),
        ("full-throttle", lambda o, L: np.array([1.0, 0.0])),
        ("loiter (skill-less)", loiter),
        ("ckpt (out-of-dist)", lambda o, L: ckpt.predict(
            frozen_layout(o, L), deterministic=True)[0]),
    ]

    print("=" * 84)
    print("WORLD v2 (random gap0 0.3-0.9, leader 0.4-0.8 m/s, jitter)")
    print("  3 candidate metrics vs a degenerate set that INCLUDES loiter")
    print("=" * 84)
    print(f"  {'policy':>20s} {'uniq_delta':>11s} {'mean_delta':>11s} "
          f"{'mean_t_ot':>10s} {'task%':>7s} {'coll%':>7s}")
    print("-" * 84)
    for name, fn in pols:
        d, t, k, c, rs = run(fn)
        u = len(set(np.round(d, 6)))
        mt = np.mean([x for x in t if np.isfinite(x)]) if np.any(np.isfinite(t)) else float("nan")
        print(f"  {name:>20s} {u:>11d} {d.mean():>+11.3f} {mt:>10.2f} "
              f"{100*k.mean():>6.1f}% {100*c:>6.1f}%")

    print()
    print("=" * 84)
    print("READING")
    print("=" * 84)
    print("  uniq_delta > 1 => B2 fixed (world now randomises).")
    print("  M_delta : if loiter still tops the table, it is wrap-saturated (B3).")
    print("  M_time  : a skill-less policy that never overtakes -> inf (fails).")
    print("  M_task  : requires passing AND holding the lead, mirroring the task.")
    print()
    print("  NOTE: the frozen checkpoint is out of distribution here (it was")
    print("  trained vs a SLOW leader, 0.15-0.30 m/s). It is NOT a valid")
    print("  positive control for this world; a positive must be trained")
    print("  in-world before any threshold can be set.")


if __name__ == "__main__":
    main()
