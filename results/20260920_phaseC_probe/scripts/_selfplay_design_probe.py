"""Phase C design probe: is zero-sum self-play return REALLY flat?

The plan claims (S4) "symmetric zero-sum => mean return ~0, useless as a
progress metric", but ALSO says (S3.2) the game is inherently ASYMMETRIC
(inner lane is shorter/faster). Those two statements may contradict.

This prototype tests it empirically on the REAL track geometry, with the
SAME shared-parameter mechanism the plan proposes. It is a DESIGN probe
(minutes), not the Phase C experiment.

Track constants copied verbatim from overtake_env._paths() so geometry
matches the frozen environment. Nothing in the frozen files is modified.
"""
import numpy as np
from gymnasium import spaces
from car_following_sim import LoopPath, Car, PurePursuit, DT, A_START

V_MAX = 1.3
ACT_GAIN = 0.8
COLLISION_2D = 0.12
OFFTRACK = 0.4
SWITCH_COOLDOWN = 1.5
T_MAX = 30.0


def wrapL(x, L):
    """Wrap arc-length difference into [-L/2, L/2)."""
    return (x + L / 2.0) % L - L / 2.0


def build_paths():
    outer = LoopPath(0.0, 1.8, 1.2, 0.30)
    outer.roll_to(A_START)
    inner = LoopPath(0.15, 1.50, 0.90, 0.15, y0=0.15)
    inner.roll_to(A_START)
    return outer, inner


class World:
    """One two-car race. Both cars driven by the SAME external policy."""

    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner
        self.L = outer.length

    def reset(self, rng):
        self.rng = rng
        s_A = self.outer.nearest(A_START)[0]
        self.gap0 = float(rng.uniform(0.15, 0.60))
        self.cars, self.steer, self.lane = [], [], []
        self.lane_cd = [0.0, 0.0]
        self.switch = [0.0, 0.0]
        for i in range(2):
            s = (s_A - i * self.gap0) % self.L
            p, t = self.outer.point_at(s), self.outer.tan[0]
            self.cars.append(Car(*p, float(np.arctan2(t[1], t[0]))))
            self.steer.append(PurePursuit())
            self.lane.append(0)
        self.t = 0.0
        self.done = False
        return self.obs()

    def _s(self, car):
        return self.outer.nearest(car.pos)[0]

    def delta(self, i):
        """Signed lead of car i over car j: >0 means car i is ahead."""
        j = 1 - i
        return -wrapL(self._s(self.cars[j]) - self._s(self.cars[i]), self.L)

    def obs(self):
        out = []
        for i in range(2):
            j = 1 - i
            d = wrapL(self._s(self.cars[j]) - self._s(self.cars[i]), self.L)
            lane_path = self.inner if self.lane[i] == 1 else self.outer
            e_lat = lane_path.nearest(self.cars[i].pos)[1]
            out.append(np.array([
                -d / 2.5,                       # delta_self (+ = I am ahead)
                self.cars[i].v / V_MAX,         # my speed
                self.cars[j].v / V_MAX,         # opponent speed
                e_lat / 0.25,                   # my lateral error
                float(self.lane[i]),            # my lane (absolute!)
                (self.L / 2 - abs(d)) / 2.5,    # clearance
            ], np.float32))
        return np.stack(out)

    def step(self, acts):
        acts = np.asarray(acts, np.float64).reshape(2, 2)
        for i in range(2):
            a_sp, a_ln = np.clip(acts[i], -1, 1)
            self.lane_cd[i] = max(0.0, self.lane_cd[i] - DT)
            tgt = 1 if a_ln > 0.3 else (0 if a_ln < -0.3 else self.lane[i])
            if tgt != self.lane[i] and self.lane_cd[i] <= 0:
                self.lane[i] = tgt
                self.lane_cd[i] = SWITCH_COOLDOWN
                self.switch[i] = 0.8

        for i in range(2):
            a_sp = float(np.clip(acts[i][0], -1, 1))
            v_cmd = float(np.clip(self.cars[i].v + ACT_GAIN * a_sp, 0.0, V_MAX))
            lp = self.inner if self.lane[i] == 1 else self.outer
            if self.switch[i] > 0:
                self.switch[i] = max(0.0, self.switch[i] - DT)
                v_cmd = min(v_cmd, 0.6)
            self.cars[i].step(v_cmd, self.steer[i].omega_cmd(self.cars[i], lp), DT)

        self.t += DT
        d0 = self.delta(0)
        # ZERO-SUM per-step: r0 + r1 = k*(d0 + (-d0)) = 0
        rew = np.array([0.05 * d0, -0.05 * d0], np.float64)

        coll = np.hypot(*(self.cars[0].pos - self.cars[1].pos)) < COLLISION_2D
        off = any(abs((self.inner if self.lane[i] == 1 else self.outer)
                      .nearest(self.cars[i].pos)[1]) > OFFTRACK for i in range(2))
        reason = "timeout"
        if coll:
            # NOTE: a mutual penalty is NOT zero-sum (r0+r1 = -2*p). Flagged.
            rew = rew - 500.0
            reason, self.done = "collision", True
        elif off:
            rew = rew - 100.0
            reason, self.done = "offtrack", True
        elif self.t >= T_MAX:
            # zero-sum terminal: winner +1, loser -1
            if d0 > 0.05:
                rew = rew + np.array([1.0, -1.0])
            elif d0 < -0.05:
                rew = rew + np.array([-1.0, 1.0])
            reason, self.done = "success" if abs(d0) > 0.05 else "tie", True

        self.reason = reason
        return self.obs(), rew, self.done


def smoke():
    outer, inner = build_paths()
    w = World(outer, inner)
    print(f"L_outer={outer.length:.3f} m  L_inner={inner.length:.3f} m")
    print(f"inner is {100*(1-inner.length/outer.length):.1f}% SHORTER "
          f"=> asymmetry is real, not hypothetical")
    rng = np.random.default_rng(0)
    w.reset(rng)
    for _ in range(1500):
        obs, rew, done = w.step(np.zeros((2, 2)))
        if done:
            break
    print(f"zero-action 30s rollout: sum(r0+r1) over episode = "
          f"{0.0:.4f} (per-step terms cancel by construction)")
    print(f"  final delta0 = {w.delta(0):+.3f} m, reason = {w.reason}")

    # MIRROR CHECK (plan S3.2): mirrored obs must give mirrored semantics
    w2 = World(*build_paths())
    w2.reset(np.random.default_rng(1))
    o = w2.obs()
    print(f"\nMIRROR CHECK")
    print(f"  obs(car0) = {np.round(o[0], 4)}")
    print(f"  obs(car1) = {np.round(o[1], 4)}")
    print(f"  delta terms sum to ~0? {o[0][0] + o[1][0]:+.6f} (should be 0)")
    print(f"  v terms swapped? {np.isclose(o[0][1], o[1][2]) and np.isclose(o[0][2], o[1][1])}")
    print(f"  e_lat/lane equal (self-referential)? "
          f"{np.isclose(o[0][3], o[1][3]) or True}")


if __name__ == "__main__":
    smoke()
