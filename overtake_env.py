
"""
Week 3 - Overtaking environment (two-lane track)
=================================================
Leader : scripted, slow (v_set ~ U(0.15, 0.30)), outer lane only.
Follower: learns WHEN to change to the inner (shorter, faster) lane to
          overtake, and when to return.  Speed action uses the same V2V
          prior as week 2 (v_cmd = v_l + 0.8*a, a=0 never rear-ends);
          lane action: |a_lane| > 0.3 triggers a lane change (1s cooldown).

Obs    : [(gap_ref-0.2)/0.5, delta/2.5, v_f/1.3, e_lat/0.25, lane, v_l/0.5]
         gap_ref: along-track gap on the OUTER loop (common reference)
         delta  : signed progress difference, + = follower ahead (wrapped)
Reward : 0.3 * v_f/1.3  (progress) + 30 on crossing to "ahead"
         -500 collision (2D dist < 0.15), -100 off-track / lost
"""
import numpy as np
from car_following_sim import LoopPath, Car, PurePursuit, DT, A_START

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    from follow_env import gym          # reuse the minimal fallback
    spaces = gym.spaces

D_DES = 0.20
COLLISION_2D = 0.12
OFFTRACK = 0.4   # soft cliff: lane sep is 0.15, switch transients peak ~0.3
V_MAX = 1.3
ACT_GAIN = 0.8
SWITCH_COOLDOWN = 1.5      # [s] between lane changes

_PATHS = None

def _paths():
    global _PATHS
    if _PATHS is None:
        outer = LoopPath(0.0, 1.8, 1.2, 0.30)
        outer.roll_to(A_START)
        # uniformly inset lane: y0 = +0.15 makes ALL four straights AND the
        # arcs parallel to the outer loop (arc-center distance = R - R' = 0.15,
        # cornering speed limit w_max*R' = 8*0.15 = 1.2 m/s at pass speed)
        inner = LoopPath(0.15, 1.50, 0.90, 0.15, y0=0.15)
        inner.roll_to(A_START)
        _PATHS = (outer, inner)
    return _PATHS


def baseline_action_ot(obs):
    """Hand-crafted overtaking state machine (the ablation opponent).

    Hand-engineered overtaking state machine (the ablation opponent):
    P-approach from far, brake if too close to the bumper, dive inside
    when roughly aligned (gap in [0.15, 0.45]), COMMIT to the pass at
    full push until ahead, then return to the outer line.  Safe and
    effective, but its fixed thresholds are fragile (see report).
    """
    gap = obs[0] * 0.5 + 0.2
    delta = obs[1] * 2.5
    lane = int(round(obs[4]))
    a_lane = 0.0
    if delta < -0.05:                          # behind
        if lane == 0 and gap < 0.15:           # same-lane bumper proximity
            a_speed = -0.6                     # brake, open lateral room
            # NOTE: this rule uses the ALONG-TRACK gap, which is meaningless
            # in the adjacent lane (dist = sqrt(gap^2 + 0.15^2) >= 0.15 >
            # collision radius) - hand rules conflate the two, RL can learn
            # the true 2D geometry.  See report ablation.
        elif lane == 0:
            if gap < 0.45:                     # roughly aligned -> dive
                a_speed, a_lane = 0.4, 1.0
            else:
                a_speed = float(np.clip(gap - 0.2, -1, 1))   # P approach
        else:
            a_speed = 0.7                      # COMMIT: on the inner lane,
                                               # pushing until ahead
    else:                                      # ahead
        a_speed = 0.4
        if lane == 1:
            a_lane = -1.0
    return np.array([a_speed, a_lane], dtype=np.float32)


class OvertakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, domain_randomize=True, t_max=30.0, gap_max=1.0,
                 behaviors=("constant", "sinusoid", "brake"),
                 v_lo=0.15, v_hi=0.30):
        super().__init__()
        self.dr, self.t_max, self.gap_max = domain_randomize, t_max, gap_max
        self.behaviors = tuple(behaviors)   # curriculum: allowed leader modes
        self.v_lo, self.v_hi = v_lo, v_hi   # curriculum: leader speed range
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)

    def _leader_v(self):
        t = self.t
        v = self.v_set
        if self.behavior == "sinusoid":
            v *= 1.0 + self.amp * np.sin(2 * np.pi * self.freq * t + self.phase)
        elif self.behavior == "brake":
            if (t % self.period) < self.brake_dur:
                v *= 0.3
        return float(np.clip(v, 0.05, 1.0))

    def _min_turn_radius(self, path, s, window=0.8):
        """Min turn radius over the `window` meters ahead (vectorized)."""
        N = len(path.pts)
        i0 = int(np.searchsorted(path.s, s % path.length, side="right") - 1) % N
        m = int(window / 0.005)
        idx = (i0 + np.arange(m + 1)) % N
        ang = np.arctan2(path.tan[idx, 1], path.tan[idx, 0])
        dth = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
        k_max = float(dth.max()) / 0.005
        return 1.0 / k_max if k_max > 1e-6 else 1e6

    def _measure(self):
        """Return (gap_ref, delta, e_lat) using the outer loop as reference."""
        s_f_out = self.outer.nearest(self.follower.pos)[0]
        s_l_out = self.outer.nearest(self.leader.pos)[0]
        gap_ref = (s_l_out - s_f_out) % self.outer.length
        delta = (s_f_out - s_l_out + self.outer.length / 2) % self.outer.length \
                - self.outer.length / 2
        lane_path = self.inner if self.lane == 1 else self.outer
        e_lat = lane_path.nearest(self.follower.pos)[1]
        return gap_ref, delta, e_lat

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif not hasattr(self, "rng"):
            self.rng = np.random.default_rng()
        self.outer, self.inner = _paths()
        r = self.rng
        self.v_set = float(r.uniform(self.v_lo, self.v_hi))   # SLOW leader
        self.behavior = str(r.choice(self.behaviors))
        self.amp = float(r.uniform(0.2, 0.4))
        self.freq = float(r.uniform(0.05, 0.15))
        self.phase = float(r.uniform(0, 2 * np.pi))
        self.period = float(r.uniform(5.0, 9.0))
        self.brake_dur = float(r.uniform(1.0, 2.0))
        tau = float(r.uniform(0.08, 0.18)) if self.dr else 0.12
        a_max = float(r.uniform(1.0, 2.0)) if self.dr else 1.5

        s_A = self.outer.nearest(A_START)[0]
        pA, t0 = self.outer.point_at(s_A), self.outer.tan[0]
        th0 = float(np.arctan2(t0[1], t0[0]))
        self.leader = Car(*pA, th0, tau=tau, a_max=a_max)
        self.leader_steer = PurePursuit()

        self.gap0 = float(r.uniform(0.30, self.gap_max))
        pf = self.outer.point_at((s_A - self.gap0) % self.outer.length)
        _, _, _, tf = self.outer.nearest(pf)
        self.follower = Car(*pf, float(np.arctan2(tf[1], tf[0])),
                            tau=tau, a_max=a_max)
        self.follower_steer = PurePursuit()
        self.lane = 0                     # 0 = outer, 1 = inner
        self.lane_cd = 0.0
        self.switch_timer = 0.0           # >0: lateral maneuver in progress
        self.ahead_flag = False
        self.t, self.k = 0.0, 0
        self.prev_delta = -self.gap0
        self.term_reason = "timeout"
        self.log = {"t": [], "gap": [], "delta": [], "v_l": [], "v_f": [],
                    "e_lat": [], "lane": []}
        return self._obs(), {}

    def _obs(self):
        gap, delta, e_lat = self._measure()
        return np.array([np.clip((gap - 0.2) / 0.5, -1.0, 6.0),
                         delta / 2.5,
                         self.follower.v / 1.3,
                         e_lat / 0.25,
                         float(self.lane),
                         self._leader_v() / 0.5], dtype=np.float32)

    def step(self, action):
        a_speed, a_lane = np.clip(np.asarray(action, dtype=np.float64).reshape(2),
                                  -1.0, 1.0)
        # 1. leader moves
        self.leader.step(self._leader_v(),
                         self.leader_steer.omega_cmd(self.leader, self.outer), DT)
        # 2. lane change (with cooldown)
        self.lane_cd = max(0.0, self.lane_cd - DT)
        target = 1 if a_lane > 0.3 else (0 if a_lane < -0.3 else self.lane)
        if target != self.lane and self.lane_cd <= 0.0:
            # lane change executes immediately, but the vehicle slows down
            # for the lateral maneuver (like real lane-change assist):
            # a transient speed cap keeps the curvature-mismatch lateral
            # overshoot well below the off-track threshold
            self.lane = target
            self.lane_cd = SWITCH_COOLDOWN
            self.switch_timer = 0.8
        # 3. follower moves (speed prior: zero action matches leader speed)
        v_l = self._leader_v()
        v_cmd = float(np.clip(v_l + ACT_GAIN * a_speed, 0.0, V_MAX))
        lane_path = self.inner if self.lane == 1 else self.outer
        # vehicle stability control: cap speed by upcoming curvature
        # (v <= 0.8*w_max*R_min over a 0.8 m lookahead window) - part of the
        # car's own dynamics, applies to ANY policy using this vehicle
        R_min = self._min_turn_radius(lane_path,
                                      lane_path.nearest(self.follower.pos)[0])
        v_cmd = min(v_cmd, 0.8 * self.follower.w_max * R_min)
        if self.switch_timer > 0.0:               # lane-change transient cap
            self.switch_timer = max(0.0, self.switch_timer - DT)
            v_cmd = min(v_cmd, 0.6)
        self.follower.step(v_cmd,
                           self.follower_steer.omega_cmd(self.follower, lane_path), DT)
        self.t += DT
        self.k += 1

        # 4. observe & reward
        gap, delta, e_lat = self._measure()
        reward = 0.3 * (self.follower.v / V_MAX)
        reward -= 0.5 * (e_lat / OFFTRACK) ** 2
        # PBRS on delta: immediate credit for closing the leader during the
        # pass (the +60 bonus alone is too sparse to discover).  Disabled
        # near the +-L/2 wrap (|delta| > 1.2) where phi is discontinuous.
        phi = lambda x: 2.0 * float(np.clip(x, -1.0, 1.0))
        if abs(delta) < 1.2 and abs(self.prev_delta) < 1.2:
            reward += 0.99 * phi(delta) - phi(self.prev_delta)
        if not self.ahead_flag and delta > 0.05:
            self.ahead_flag = True
            reward += 60.0                   # overtake completed
        # NOTE: ahead_flag is NEVER reset - delta wraps at +-L/2 once the
        # faster follower laps the leader, which must not count as
        # "losing the lead".

        terminated, truncated = False, False
        dist2d = float(np.hypot(*(self.follower.pos - self.leader.pos)))
        self.term_reason = "timeout"
        if dist2d < COLLISION_2D:
            reward -= 500.0
            terminated, self.term_reason = True, "collision"
        elif abs(e_lat) > OFFTRACK:
            reward -= 100.0
            terminated, self.term_reason = True, "offtrack"
        elif not self.ahead_flag and delta < 0.0 and gap > 2.0:
            # only before the first overtake: afterwards the faster follower
            # is SUPPOSED to lap the leader (delta wraps at +-L/2)
            reward -= 100.0
            terminated, self.term_reason = True, "lost"

        self.prev_delta = delta
        self.log["t"].append(self.t)
        self.log["gap"].append(gap)
        self.log["delta"].append(delta)
        self.log["v_l"].append(v_l)
        self.log["v_f"].append(self.follower.v)
        self.log["e_lat"].append(e_lat)
        self.log["lane"].append(self.lane)
        if self.t >= self.t_max:
            truncated = True
            if not self.ahead_flag:
                # task failure: hovering behind the leader must be worse than
                # finishing the pass (success bonus + early termination)
                reward -= 20.0
                self.term_reason = "failed"
        elif self.ahead_flag and delta > 0.3:
            # task accomplished: completed the overtake with a safe margin
            truncated, self.term_reason = True, "success"
        return self._obs(), float(reward), terminated, truncated, {}
