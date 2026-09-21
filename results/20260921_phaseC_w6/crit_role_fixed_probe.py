"""Feasibility probe for the role-fixed criterion (BEFORE writing the plan).

Load-bearing question: in the user's proposed design (role-fixed, task-level
criterion delta > +0.3, metric = min(win as follower, win as leader)), can the
POSITIVE CONTROL actually score? If the frozen checkpoint cannot demonstrate
"good" under this criterion, the criterion has no positive pole and is useless
(all-zero is indistinguishable from all-bad).

Second question: which action prior should M1 use?
  - V2V (frozen env semantics): the frozen checkpoint and the rule machine are
    natively valid -> both anchors usable, and we inherit the frozen env's own
    success definition unchanged.
  - self-relative (W5 semantics): keeps comparability with the W5 probe, but
    the frozen checkpoint is out of distribution.

Test both priors, role-fixed, follower starts a fixed 0.5 m behind.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from car_following_sim import Car, PurePursuit, A_START, DT
from _selfplay_design_probe import build_paths, wrapL
from _ckpt_as_opponent import frozen_layout

V_MAX, ACT_GAIN = 1.3, 0.8
COLLISION_2D, OFFTRACK, SWITCH_COOLDOWN, T_MAX = 0.12, 0.4, 1.5, 30.0
WIN_MARGIN = 0.3          # frozen env's own success margin (overtake_env.py:263)


class RoleWorld:
    """car 0 = FOLLOWER (starts behind), car 1 = LEADER (starts ahead).

    prior: 'v2v'          -> v_cmd = v_other + 0.8*a   (frozen env semantics)
           'self'         -> v_cmd = v_self  + 0.8*a   (W5 probe semantics)
    """

    def __init__(self, outer, inner, prior="v2v"):
        self.outer, self.inner, self.prior = outer, inner, prior
        self.L = outer.length

    def reset(self, rng, gap=0.5):
        self.rng = rng
        self.gap0 = float(gap)
        s_A = self.outer.nearest(A_START)[0]
        self.cars, self.steer, self.lane = [], [], []
        self.lane_cd = [0.0, 0.0]
        self.switch = [0.0, 0.0]
        # car 0 behind at s_A - gap ; car 1 at s_A (ahead)
        for i, off in [(0, -self.gap0), (1, 0.0)]:
            s = (s_A + off) % self.L
            p, t = self.outer.point_at(s), self.outer.tan[0]
            self.cars.append(Car(*p, float(np.arctan2(t[1], t[0]))))
            self.steer.append(PurePursuit())
            self.lane.append(0)
        self.t, self.done, self.reason = 0.0, False, "timeout"
        return self.obs()

    def _s(self, car):
        return self.outer.nearest(car.pos)[0]

    def delta(self, i):
        """Signed lead of car i over the OTHER car (>0 = car i ahead)."""
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

    def step(self, acts):
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
            ref = self.cars[j].v if self.prior == "v2v" else self.cars[i].v
            v_cmd = float(np.clip(ref + ACT_GAIN * a_sp, 0.0, V_MAX))
            lp = self.inner if self.lane[i] == 1 else self.outer
            if self.switch[i] > 0:
                self.switch[i] = max(0.0, self.switch[i] - DT)
                v_cmd = min(v_cmd, 0.6)
            self.cars[i].step(v_cmd, self.steer[i].omega_cmd(self.cars[i], lp), DT)
        self.t += DT
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


def evaluate(policy_fn, opponent_fn, prior, n_ep=40, seed0=90000,
             as_role="follower", gap=0.5):
    """Follower = car 0 (starts `gap` behind). Win = follower ends delta > 0.3.

    as_role='follower': policy_fn drives car 0, opponent_fn drives car 1
    as_role='leader'  : policy_fn drives car 1, opponent_fn drives car 0
    """
    wins = 0
    for k in range(n_ep):
        w = RoleWorld(*build_paths(), prior=prior)
        obs = w.reset(np.random.default_rng(seed0 + k), gap=gap)
        while True:
            if as_role == "follower":
                a0, a1 = policy_fn(obs[0], w.L), opponent_fn(obs[1], w.L)
            else:
                a0, a1 = opponent_fn(obs[0], w.L), policy_fn(obs[1], w.L)
            obs, done = w.step(np.stack([a0, a1]))
            if done:
                break
        follower_ahead = w.delta(0) > WIN_MARGIN      # car 0 is the follower
        if as_role == "follower":
            if follower_ahead:
                wins += 1
        else:
            if not follower_ahead:
                wins += 1
    return wins / n_ep


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")

    rng = np.random.default_rng(17)
    refs = {
        "zero-action":  lambda o, L: np.zeros(2),
        "full-throttle": lambda o, L: np.array([1.0, 0.0]),
        "random":       lambda o, L: rng.uniform(-1, 1, 2),
    }

    for prior in ["v2v", "self"]:
        print("=" * 78)
        print(f"PRIOR = {prior}   (follower starts {0.5} m behind; "
              f"win = delta_follower > {WIN_MARGIN})")
        print("=" * 78)
        print(f"  {'policy':>16s} {'as follower':>12s} {'as leader':>11s} "
              f"{'min':>7s}   vs opponent = rule machine")
        for name, fn in refs.items():
            from overtake_env import baseline_action_ot
            opp = lambda o, L: baseline_action_ot(frozen_layout(o, L))
            wf = evaluate(fn, opp, prior, as_role="follower")
            wl = evaluate(fn, opp, prior, as_role="leader")
            print(f"  {name:>16s} {wf:>12.2f} {wl:>11.2f} {min(wf,wl):>7.2f}")

        # positive control: the frozen checkpoint
        def ckp(o, L):
            return ckpt.predict(frozen_layout(o, L), deterministic=True)[0]
        from overtake_env import baseline_action_ot
        opp = lambda o, L: baseline_action_ot(frozen_layout(o, L))
        wf = evaluate(ckp, opp, prior, as_role="follower")
        wl = evaluate(ckp, opp, prior, as_role="leader")
        print(f"  {'FROZEN CKPT(+)':>16s} {wf:>12.2f} {wl:>11.2f} {min(wf,wl):>7.2f}")
        print()


if __name__ == "__main__":
    main()
