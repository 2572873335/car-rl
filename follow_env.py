
"""
Week 2 - Gymnasium environment for the TI Cup C-following task
==============================================================
Task : follow a *scripted* leader at d_des, whose speed varies
       (constant / sinusoid / random braking) -> the controller must
       anticipate, not just react.  The RL agent replaces the rule-based
       spacing controller only; steering stays pure-pursuit (layered arch).

Obs  : [e_gap/0.5, de_gap/2.0, v_follower/1.3, e_lat/0.25]  (normalized, float32)
Act  : a in [-1, 1]  ->  v_cmd = clip(v_f + de + 0.8*a, 0, 1.3)
       (action prior: a=0 matches the leader speed; RL learns residuals)
Reward: -(e/0.08)^2 - 0.3*(de/1.0)^2 - 0.5*(e_lat/0.15)^2  per step
        collision (gap<0.12) : -100, terminated
        lost    (gap>2.0, |e_lat|>0.25) : -20, terminated
Domain randomization (toggle): follower motor lag tau & accel limit a_max.
Baseline (rule) controller with the SAME obs/action interface is provided
for the ablation study:  baseline_action(obs, use_ff=...)  P / P+feedforward.
"""
import numpy as np
from car_following_sim import LoopPath, Car, PurePursuit, DT, A_START

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # minimal fallback so the dynamics can be smoke-tested
    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype
    class _Spaces:
        Box = _Box
    class _Gym:
        Env = object
        spaces = _Spaces()
    gym = _Gym()

# ------------------------------------------------------------- constants ---
D_DES = 0.20        # target following gap [m]
COLLISION_GAP = 0.12
MAX_GAP = 2.0
OFFTRACK = 0.25
V_MAX = 1.3
ACT_GAIN = 0.8      # v_cmd = v_f + ACT_GAIN * action

_PATHS = None

def _paths():
    """Build track once per process (safe under SubprocVecEnv fork)."""
    global _PATHS
    if _PATHS is None:
        outer = LoopPath(0.0, 1.8, 1.2, 0.30)
        outer.roll_to(A_START)
        _PATHS = outer
    return _PATHS


def baseline_action(obs, kp=0.8, use_ff=False):
    """Rule-based spacing controller, identical interface to the RL policy.

    v_cmd = v_f + kp*e  (P only)            use_ff=False
    v_cmd = v_l + kp*e  (P + speed feedforward via de/dt)  use_ff=True
    """
    e, de = obs[0] * 0.5, obs[1] * 2.0   # un-normalize (see FollowEnv._obs)
    # env computes v_cmd = v_f + de + ACT_GAIN*a, so:
    #   use_ff=True  -> v_cmd = v_l + kp*e      (classic P + speed feedforward)
    #   use_ff=False -> v_cmd = v_f + kp*e      (pure P; must fight the prior)
    a = (kp * e - (0.0 if use_ff else de)) / ACT_GAIN
    return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)


class FollowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, domain_randomize=True, t_max=20.0,
                 gap_max=1.2, behaviors=("constant", "sinusoid", "brake")):
        super().__init__()
        self.dr = domain_randomize
        self.t_max = t_max
        self.gap_max = gap_max            # curriculum: cap the initial gap
        self.behaviors = tuple(behaviors)  # curriculum: allowed leader modes
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ---------------------------------------------------------- helpers ---
    def _leader_v(self):
        t = self.t
        v = self.v_set
        if self.behavior == "sinusoid":
            v *= 1.0 + self.amp * np.sin(2 * np.pi * self.freq * t + self.phase)
        elif self.behavior == "brake":
            if (t % self.period) < self.brake_dur:
                v *= 0.3
        return float(np.clip(v, 0.05, 1.0))

    def _measure(self):
        s_l = self.path.nearest(self.leader.pos)[0]
        s_f, e_lat, _, _ = self.path.nearest(self.follower.pos)
        return (s_l - s_f) % self.path.length, e_lat

    # ------------------------------------------------------------ gym API ---
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif not hasattr(self, "rng"):
            self.rng = np.random.default_rng()

        self.path = _paths()
        r = self.rng
        # ---- task variation (resampled every episode) ----
        self.v_set = float(r.uniform(0.25, 0.50))
        self.behavior = str(r.choice(self.behaviors))
        self.amp = float(r.uniform(0.2, 0.4))          # sinusoid
        self.freq = float(r.uniform(0.05, 0.15))
        self.phase = float(r.uniform(0, 2 * np.pi))
        self.period = float(r.uniform(5.0, 9.0))       # braking
        self.brake_dur = float(r.uniform(1.0, 2.0))
        # ---- domain randomization (follower actuation) ----
        tau = float(r.uniform(0.08, 0.18)) if self.dr else 0.12
        a_max = float(r.uniform(1.0, 2.0)) if self.dr else 1.5

        s_A = self.path.nearest(A_START)[0]
        pA, t0 = self.path.point_at(s_A), self.path.tan[0]
        th0 = float(np.arctan2(t0[1], t0[0]))
        self.leader = Car(*pA, th0, tau=tau, a_max=a_max)
        self.leader_steer = PurePursuit()

        gap0 = float(r.uniform(0.20, self.gap_max))   # catch-up scenarios
        self.gap0 = gap0
        pf = self.path.point_at((s_A - gap0) % self.path.length)
        _, _, _, tf = self.path.nearest(pf)
        self.follower = Car(*pf, float(np.arctan2(tf[1], tf[0])),
                            tau=tau, a_max=a_max)
        self.follower_steer = PurePursuit()

        self.t, self.k = 0.0, 0
        self.prev_e, _ = self._measure()
        self.prev_e -= D_DES
        self.log = {"t": [], "gap": [], "v_l": [], "v_f": [], "e_lat": []}
        return self._obs(), {}

    def _obs(self):
        gap, e_lat = self._measure()
        e = gap - D_DES
        de = (e - self.prev_e) / DT
        return np.array([e / 0.5, de / 2.0, self.follower.v / 1.3, e_lat / 0.25],
                        dtype=np.float32)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).item()
        # 1. leader moves (scripted)
        self.leader.step(self._leader_v(),
                         self.leader_steer.omega_cmd(self.leader, self.path), DT)
        # 2. follower moves.  Action prior: the scripted leader broadcasts its
        #    speed command (V2V), so zero action -> v_cmd = v_l (speed match,
        #    never rear-ends); RL learns only the residual correction.
        #    The obs `de` (post-move) still has to be learned by the policy.
        v_l = self._leader_v()
        v_cmd = float(np.clip(v_l + ACT_GAIN * a, 0.0, V_MAX))
        self.follower.step(v_cmd,
                           self.follower_steer.omega_cmd(self.follower, self.path), DT)
        self.t += DT
        self.k += 1

        # 3. observe & reward (post-move state)
        gap, e_lat = self._measure()
        e = gap - D_DES
        de = (e - self.prev_e) / DT

        reward = (-(e / 0.12) ** 2
                  - 0.3 * (de / 2.0) ** 2
                  - 0.5 * (e_lat / 0.15) ** 2)
        # potential-based reward shaping: instant credit for closing the gap,
        # theory-safe (does not change the optimal policy)
        GAMMA_ENV = 0.99
        phi = lambda x: -0.5 * min(abs(x), 1.0)
        reward += GAMMA_ENV * phi(e) - phi(self.prev_e)
        self.prev_e = e

        terminated, truncated, self.term_reason = False, False, "timeout"
        if gap < COLLISION_GAP:
            reward -= 500.0
            terminated, self.term_reason = True, "collision"
        elif gap > MAX_GAP:
            reward -= 100.0
            terminated, self.term_reason = True, "lost"
        elif abs(e_lat) > OFFTRACK:
            reward -= 100.0
            terminated, self.term_reason = True, "offtrack"
        if self.t >= self.t_max:
            truncated = True

        obs = np.array([e / 0.5, de / 2.0, self.follower.v / 1.3, e_lat / 0.25],
                       dtype=np.float32)

        self.log["t"].append(self.t)
        self.log["gap"].append(gap)
        self.log["v_l"].append(self._leader_v())
        self.log["v_f"].append(self.follower.v)
        self.log["e_lat"].append(e_lat)
        return obs, float(reward), terminated, truncated, {}
