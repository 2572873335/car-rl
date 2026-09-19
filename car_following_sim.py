
"""
Week 1 - 2022 TI Cup C题 (car following) 2D simulation environment
=================================================================
- Track geometry: outer loop 1.8m x 1.2m (R=0.3) + inner loop (AB=1.0m, EF=0.8m),
  counter-clockwise, both loops share start point A=(0.5, 0).
- Differential-drive car kinematics with first-order motor lag & accel limit.
- Low level : PID-like speed servo + pure-pursuit heading controller.
- High level: rule-based leader (constant speed) + spacing-P follower (the
              baseline that the RL agent will replace in week 2+).
Units: meter, second, radian.  Pure numpy, no physics engine needed.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.02
A_START = (0.5, 0.0)


# ---------------------------------------------------------------- track ----
def _line(p, q, ds):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(int(np.hypot(*(q - p)) / ds), 1)
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    return p + t * (q - p)


def _arc(c, r, a0, a1, ds):
    a0, a1 = np.radians(a0), np.radians(a1)
    n = max(int(abs(a1 - a0) * r / ds), 4)
    ang = np.linspace(a0, a1, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1)


class LoopPath:
    """Closed CCW loop, dense polyline + arclength table."""

    def __init__(self, x0, W, H, R, ds=0.005, y0=0.0):
        self.pts = np.vstack([
            _line((x0 + R, 0), (x0 + W - R, 0), ds),          # bottom
            _arc((x0 + W - R, R), R, -90, 0, ds),            # bottom-right
            _line((x0 + W, R), (x0 + W, H - R), ds),         # right
            _arc((x0 + W - R, H - R), R, 0, 90, ds),         # top-right
            _line((x0 + W - R, H), (x0 + R, H), ds),         # top
            _arc((x0 + R, H - R), R, 90, 180, ds),           # top-left
            _line((x0, H - R), (x0, R), ds),                 # left
            _arc((x0 + R, R), R, 180, 270, ds),              # bottom-left
        ])
        if y0:
            self.pts[:, 1] += y0
        self._finalize()

    def _finalize(self):
        d = np.diff(np.vstack([self.pts, self.pts[:1]]), axis=0)
        seglen = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seglen)])
        self.length = float(self.s[-1])
        self.tan = d / seglen[:, None]

    def roll_to(self, point):
        """Rotate the loop so arclength 0 sits at the given point."""
        i = int(np.argmin(((self.pts - np.asarray(point)) ** 2).sum(1)))
        self.pts = np.roll(self.pts, -i, axis=0)
        self._finalize()

    def nearest(self, p):
        """Return (s, e_lat, nearest_point, tangent). e_lat>0: left of path."""
        p = np.asarray(p, float)
        i = int(np.argmin(((self.pts - p) ** 2).sum(1)))
        j = (i + 1) % len(self.pts)
        seg = self.pts[j] - self.pts[i]
        L2 = float(seg @ seg)
        t = 0.0 if L2 == 0 else np.clip(((p - self.pts[i]) @ seg) / L2, 0, 1)
        q = self.pts[i] + t * seg
        tan = self.tan[i]
        e_lat = tan[0] * (p[1] - q[1]) - tan[1] * (p[0] - q[0])
        s = (self.s[i] + np.sqrt(L2) * t) % self.length
        return s, e_lat, q, tan

    def point_at(self, s):
        s = s % self.length
        i = int(np.searchsorted(self.s, s, side="right") - 1) % len(self.pts)
        j = (i + 1) % len(self.pts)
        seg_len = self.s[j] - self.s[i] if j > i else self.length - self.s[i]
        t = 0.0 if seg_len <= 0 else (s - self.s[i]) / seg_len
        return self.pts[i] + t * (self.pts[j] - self.pts[i])


# ----------------------------------------------------------------- car -----
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Car:
    """Differential-drive kinematics + first-order actuation lag."""

    def __init__(self, x, y, theta, tau=0.12, a_max=1.5, w_max=8.0):
        self.pos = np.array([x, y], float)
        self.theta = theta
        self.v, self.omega = 0.0, 0.0
        self.tau, self.a_max, self.w_max = tau, a_max, w_max

    def step(self, v_cmd, w_cmd, dt=DT):
        self.v += np.clip(v_cmd - self.v, -self.a_max * dt, self.a_max * dt)
        self.omega += (np.clip(w_cmd, -self.w_max, self.w_max) - self.omega) * dt / self.tau
        self.theta = wrap(self.theta + self.omega * dt)
        self.pos = self.pos + self.v * np.array([np.cos(self.theta), np.sin(self.theta)]) * dt


class PurePursuit:
    """Heading controller: speed-scaled lookahead + curvature feedforward.

    omega = kp*alpha - kd*omega + v*kappa   (kappa: path curvature, numeric)
    The feedforward term is what makes high-speed cornering (catch-up at
    v > 1 m/s on the r=0.3 m arcs) possible; without it the P-term alone
    saturates and the car cuts the corner.
    """

    def __init__(self, kp=4.0, kd=0.3, base_ld=0.10, ld_gain=0.40, ld_max=0.45, ff=1.0):
        self.kp, self.kd = kp, kd
        self.base_ld, self.ld_gain, self.ld_max = base_ld, ld_gain, ld_max
        self.ff = ff

    def omega_cmd(self, car, path):
        s, _, _, tan = path.nearest(car.pos)
        ld = float(np.clip(self.base_ld + self.ld_gain * car.v, self.base_ld, self.ld_max))
        tgt = path.point_at(s + ld)
        alpha = wrap(np.arctan2(tgt[1] - car.pos[1], tgt[0] - car.pos[0]) - car.theta)
        # numeric curvature from tangent direction change over a small window
        tan2 = path.tan[(int(np.searchsorted(path.s, (s + 0.05) % path.length,
                                             side="right")) - 1) % len(path.pts)]
        kappa = wrap(np.arctan2(tan2[1], tan2[0]) - np.arctan2(tan[1], tan[0])) / 0.05
        return self.kp * alpha - self.kd * car.omega + self.ff * car.v * kappa


# ------------------------------------------------------------- baseline ----
class Leader:
    """Constant-speed path following (the contest's leader behavior)."""

    def __init__(self, path, v_set):
        self.path, self.v_set = path, v_set
        s0 = path.nearest(A_START)[0]
        p, t = path.point_at(s0), path.tan[0]
        self.car = Car(*p, np.arctan2(t[1], t[0]))
        self.steer = PurePursuit()

    def step(self, dt=DT):
        self.car.step(self.v_set, self.steer.omega_cmd(self.car, self.path), dt)


class Follower:
    """Spacing-P controller: keep d_des behind the leader. THE RL REPLACEMENT TARGET."""

    def __init__(self, path, leader, d_des=0.20, kp_d=0.8, v_max=1.2):
        self.path, self.leader, self.d_des, self.kp_d, self.v_max = path, leader, d_des, kp_d, v_max
        s0 = (path.nearest(A_START)[0] - d_des) % path.length
        p = path.point_at(s0)
        _, _, _, t = path.nearest(p)
        self.car = Car(*p, np.arctan2(t[1], t[0]))
        self.steer = PurePursuit()

    def gap(self):
        """Along-track distance leader->follower."""
        return (self.path.nearest(self.leader.car.pos)[0]
                - self.path.nearest(self.car.pos)[0]) % self.path.length

    def step(self, dt=DT):
        e = self.gap() - self.d_des
        # NOTE: uses leader.v directly. In week 2+ the RL policy only gets
        # measured history (gap, d(gap)/dt), so replace this term by its estimate.
        v_cmd = np.clip(self.leader.car.v + self.kp_d * e, 0.0, self.v_max)
        self.car.step(v_cmd, self.steer.omega_cmd(self.car, self.path), dt)


# ------------------------------------------------------------------ sim ----
def run(T=45.0, v_set=0.3, d_des=0.20, seed=0):
    rng = np.random.default_rng(seed)
    outer = LoopPath(0.0, 1.8, 1.2, 0.30)   # 图1 外圈 180x120cm, r=30cm
    inner = LoopPath(0.3, 1.4, 1.2, 0.20)   # 内圈: AB=100cm, EF=80cm
    outer.roll_to(A_START)
    inner.roll_to(A_START)

    leader = Leader(outer, v_set)
    follower = Follower(outer, leader, d_des)

    n = int(T / DT)
    log = {k: np.zeros(n) for k in ("gap", "e_gap", "v_l", "v_f", "e_lat_f")}
    traj_l, traj_f = np.zeros((n, 2)), np.zeros((n, 2))
    near_hit = 0

    for k in range(n):
        leader.step()
        follower.step()
        gap = follower.gap()
        near_hit += gap < 0.12
        log["gap"][k] = gap
        log["e_gap"][k] = gap - d_des
        log["v_l"][k], log["v_f"][k] = leader.car.v, follower.car.v
        log["e_lat_f"][k] = outer.nearest(follower.car.pos)[1]
        traj_l[k], traj_f[k] = leader.car.pos, follower.car.pos

    print(f"== sim done: T={T}s  v_set={v_set}m/s  d_des={d_des}m ==")
    print(f"  mean|gap err| = {np.mean(np.abs(log['e_gap']))*100:.2f} cm")
    print(f"  max |gap err| = {np.max(np.abs(log['e_gap']))*100:.2f} cm")
    print(f"  last gap      = {log['gap'][-1]*100:.2f} cm")
    print(f"  follower lateral RMS = {np.sqrt(np.mean(log['e_lat_f']**2))*100:.2f} cm")
    print(f"  near-collision steps (gap<12cm) = {near_hit}")
    plot(outer, inner, traj_l, traj_f, log, v_set, d_des)
    return outer, inner, traj_l, traj_f, log


def plot(outer, inner, traj_l, traj_f, log, v_set, d_des):
    t = np.arange(len(log["gap"])) * DT
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 3, (1, 2))
    ax.plot(*outer.pts.T, color="gray", lw=1, label="outer loop")
    ax.plot(*inner.pts.T, color="lightgray", lw=1, ls="--", label="inner loop")
    ax.plot(*traj_l.T, color="tab:blue", lw=1.2, label="leader traj")
    ax.plot(*traj_f.T, color="tab:orange", lw=1.2, label="follower traj")
    ax.plot(*A_START, "k*", ms=12, label="A (start)")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Track & trajectories"); ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 3)
    ax2.plot(t, log["gap"] * 100, lw=0.8)
    ax2.axhline(d_des * 100, color="r", ls="--", lw=1)
    ax2.set_ylabel("gap [cm]"); ax2.set_title("follower gap"); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 6)
    ax3.plot(t, log["v_l"], label="leader")
    ax3.plot(t, log["v_f"], label="follower")
    ax3.axhline(v_set, color="r", ls="--", lw=1)
    ax3.set_xlabel("t [s]"); ax3.set_ylabel("v [m/s]")
    ax3.set_title("speeds"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("week1_sim_result.png", dpi=120)
    print("  figure saved -> week1_sim_result.png")


if __name__ == "__main__":
    run()
