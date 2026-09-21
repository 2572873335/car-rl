"""Reviewer test T5: does the W5 verdict SURVIVE the V2V action-prior change,
AND the addition of the vehicle curvature speed cap?

W5 world (self-prior, NO curvature cap):
    v_cmd = v_self + 0.8*a
Frozen env (V2V prior, WITH curvature cap):
    v_cmd = min(v_opponent + 0.8*a, 0.8*w_max*R_min)

The W5 world ALSO omits the frozen env's curvature speed cap
(_selfplay_design_probe.World.step has no R_min term).  So there are TWO
differences, not one.  Test three worlds:
  arm J-V2V   : joint self-play, V2V prior, no cap   (author's own test)
  arm F-SELF  : frozen opponent, self prior, no cap  (reproduce W5 run3)
  arm F-V2V   : frozen opponent, V2V prior, no cap    (the plan's chosen prior)
  arm F-FROZ  : frozen opponent, V2V prior, WITH cap  (the actual W6 target)
"""
import sys
import time
import tempfile
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from car_following_sim import Car, PurePursuit, A_START, DT
from _selfplay_design_probe import build_paths, wrapL

OBS_DIM, ACT_DIM = 6, 2
V_MAX, ACT_GAIN = 1.3, 0.8
COLLISION_2D, OFFTRACK, SWITCH_COOLDOWN, T_MAX = 0.12, 0.4, 1.5, 30.0


class WorldX:
    """Two-car world, parameterised by action reference + curvature cap."""

    def __init__(self, outer, inner, v2v=True, cap=False):
        self.outer, self.inner = outer, inner
        self.L = outer.length
        self.v2v, self.cap = v2v, cap

    def reset(self, rng):
        self.rng = rng
        s_A = self.outer.nearest(A_START)[0]
        gap0 = float(rng.uniform(0.15, 0.60))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        self.gap0 = gap0 * sign
        self.cars, self.steer, self.lane = [], [], []
        self.lane_cd = [0.0, 0.0]
        self.switch = [0.0, 0.0]
        for i in range(2):
            s = (s_A - i * self.gap0) % self.L
            p, t = self.outer.point_at(s), self.outer.tan[0]
            self.cars.append(Car(*p, float(np.arctan2(t[1], t[0]))))
            self.steer.append(PurePursuit())
            self.lane.append(0)
        self.t, self.done, self.reason = 0.0, False, "timeout"
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

    def _min_turn_radius(self, path, s, window=0.8):
        N = len(path.pts)
        i0 = int(np.searchsorted(path.s, s % path.length, side="right") - 1) % N
        m = int(window / 0.005)
        idx = (i0 + np.arange(m + 1)) % N
        ang = np.arctan2(path.tan[idx, 1], path.tan[idx, 0])
        dth = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
        k_max = float(dth.max()) / 0.005
        return 1.0 / k_max if k_max > 1e-6 else 1e6

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
            j = 1 - i
            v_ref = self.cars[j].v if self.v2v else self.cars[i].v
            a_sp = float(np.clip(acts[i][0], -1, 1))
            v_cmd = float(np.clip(v_ref + ACT_GAIN * a_sp, 0.0, V_MAX))
            lp = self.inner if self.lane[i] == 1 else self.outer
            if self.cap:
                R_min = self._min_turn_radius(lp, lp.nearest(self.cars[i].pos)[0])
                v_cmd = min(v_cmd, 0.8 * self.cars[i].w_max * R_min)
            if self.switch[i] > 0:
                self.switch[i] = max(0.0, self.switch[i] - DT)
                v_cmd = min(v_cmd, 0.6)
            self.cars[i].step(v_cmd, self.steer[i].omega_cmd(self.cars[i], lp), DT)
        self.t += DT
        d0 = self.delta(0)
        rew = np.array([0.05 * d0, -0.05 * d0], np.float64)
        coll = np.hypot(*(self.cars[0].pos - self.cars[1].pos)) < COLLISION_2D
        off = any(abs((self.inner if self.lane[i] == 1 else self.outer)
                      .nearest(self.cars[i].pos)[1]) > OFFTRACK for i in range(2))
        if coll:
            rew = rew - 500.0
            self.reason, self.done = "collision", True
        elif off:
            rew = rew - 100.0
            self.reason, self.done = "offtrack", True
        elif self.t >= T_MAX:
            if d0 > 0.05:
                rew = rew + np.array([1.0, -1.0])
            elif d0 < -0.05:
                rew = rew + np.array([-1.0, 1.0])
            self.reason = "success" if abs(d0) > 0.05 else "tie"
            self.done = True
        return self.obs(), rew, self.done


class JointVec(VecEnv):
    def __init__(self, n_worlds=8, seed=0, v2v=True, cap=False):
        self.n_worlds = n_worlds
        self.paths = build_paths()
        self.worlds = [WorldX(*self.paths, v2v=v2v, cap=cap) for _ in range(n_worlds)]
        self.rngs = [np.random.default_rng(seed + 1000 * i) for i in range(n_worlds)]
        self._ret = np.zeros(n_worlds)
        self.ep_ret, self.ep_reason = [], []
        super().__init__(num_envs=2 * n_worlds,
                         observation_space=spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
                         action_space=spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32))

    def reset(self):
        self._ret[:] = 0.0
        return np.concatenate([w.reset(r) for w, r in zip(self.worlds, self.rngs)])

    def step_async(self, a):
        self._acts = np.asarray(a, np.float64).reshape(self.n_worlds, 2, ACT_DIM)

    def step_wait(self):
        obs_l, rew_l, done_l, inf_l = [], [], [], []
        for k, w in enumerate(self.worlds):
            o, r, d = w.step(self._acts[k])
            self._ret[k] += r.sum()
            obs_l.append(o); rew_l.append(r); done_l.extend([d, d])
            inf_l.extend([{"reason": w.reason}] * 2)
            if d:
                self.ep_ret.append(self._ret[k]); self.ep_reason.append(w.reason)
                self._ret[k] = 0.0
                self.rngs[k] = np.random.default_rng(int(self.rngs[k].integers(1 << 30)))
        return (np.concatenate(obs_l), np.concatenate(rew_l),
                np.array(done_l, bool), inf_l)

    def close(self): pass
    def env_method(self, *a, **k): raise NotImplementedError
    def get_attr(self, *a, **k): return [None] * self.num_envs
    def set_attr(self, *a, **k): pass
    def env_is_wrapped(self, *a, **k): return [False] * self.num_envs


class FrozenVec(VecEnv):
    """car 0 trains, car 1 = frozen policy."""
    def __init__(self, opp, n_worlds=8, seed=0, v2v=True, cap=False):
        self.opp = opp
        self.n_worlds = n_worlds
        self.paths = build_paths()
        self.worlds = [WorldX(*self.paths, v2v=v2v, cap=cap) for _ in range(n_worlds)]
        self.rngs = [np.random.default_rng(seed + 1000 * i) for i in range(n_worlds)]
        self._ret = np.zeros(n_worlds)
        self.ep_ret, self.ep_reason = [], []
        super().__init__(num_envs=n_worlds,
                         observation_space=spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
                         action_space=spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32))

    def reset(self):
        self._ret[:] = 0.0
        return np.stack([w.reset(r)[0] for w, r in zip(self.worlds, self.rngs)])

    def step_async(self, a):
        self._acts = np.asarray(a, np.float64).reshape(self.n_worlds, ACT_DIM)

    def step_wait(self):
        obs_l, rew_l, done_l, inf_l = [], [], [], []
        for k, w in enumerate(self.worlds):
            a1, _ = self.opp.predict(w.obs()[1], deterministic=True)
            o, r, d = w.step(np.stack([self._acts[k], a1]))
            self._ret[k] += r[0]
            obs_l.append(o[0]); rew_l.append(r[0]); done_l.append(d)
            inf_l.append({"reason": w.reason})
            if d:
                self.ep_ret.append(self._ret[k]); self.ep_reason.append(w.reason)
                self._ret[k] = 0.0
                self.rngs[k] = np.random.default_rng(int(self.rngs[k].integers(1 << 30)))
        return (np.stack(obs_l), np.array(rew_l, np.float32),
                np.array(done_l, bool), inf_l)

    def close(self): pass
    def env_method(self, *a, **k): raise NotImplementedError
    def get_attr(self, *a, **k): return [None] * self.num_envs
    def set_attr(self, *a, **k): pass
    def env_is_wrapped(self, *a, **k): return [False] * self.num_envs


class Trend(BaseCallback):
    def __init__(self, v, bucket=100):
        super().__init__()
        self.v, self.b, self.marks, self._n = v, bucket, [], 0
    def _on_step(self):
        n = len(self.v.ep_reason)
        if n - self._n >= self.b:
            self._n = n
            rec = self.v.ep_reason[-self.b:]
            coll = sum(1 for r in rec if r == "collision") / len(rec)
            ret = float(np.mean(self.v.ep_ret[-self.b:]))
            self.marks.append((self.num_timesteps, ret, coll))
        return True


def snap(m):
    p = tempfile.mktemp(suffix=".zip")
    m.save(p)
    return PPO.load(p)


def run_joint(steps_updates, label, v2v, cap):
    print()
    print("=" * 78)
    print(f"JOINT self-play | {label}")
    print("=" * 78)
    venv = JointVec(8, 0, v2v=v2v, cap=cap)
    m = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=512,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    cb = Trend(venv, 100)
    S = 16 * 256 * steps_updates
    t0 = time.perf_counter()
    m.learn(total_timesteps=S, callback=cb, progress_bar=False)
    print(f"  {S:,} steps in {time.perf_counter()-t0:.0f}s  "
          f"updates={S/(venv.num_envs*256):.0f}")
    print(f"  {'steps':>9s} {'mean_ret':>10s} {'collision%':>11s}")
    for s, r, c in cb.marks[::max(1, len(cb.marks)//10)]:
        print(f"  {s:>9,} {r:>10.1f} {100*c:>10.1f}%")
    if len(cb.marks) >= 3:
        f = np.mean([c for _, _, c in cb.marks[:3]])
        l = np.mean([c for _, _, c in cb.marks[-3:]])
        print(f"  collision first3={100*f:.1f}%  last3={100*l:.1f}%  "
              f"=> {'STILL FAILS' if l > 0.6 else 'DIFFERENT from W5'}")
    venv.close()


def run_frozen(updates, label, v2v, cap, warmup=20):
    print()
    print("=" * 78)
    print(f"FROZEN-OPPONENT | {label}")
    print("=" * 78)
    warm = JointVec(8, 0, v2v=v2v, cap=cap)
    m = PPO("MlpPolicy", warm, learning_rate=3e-4, n_steps=256, batch_size=512,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    m.learn(total_timesteps=16 * 256 * warmup, progress_bar=False)
    warm.close()
    opp = snap(m)
    venv = FrozenVec(opp, 8, 0, v2v=v2v, cap=cap)
    m2 = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=512,
             gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
             policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    cb = Trend(venv, 50)
    S = 8 * 256 * updates
    t0 = time.perf_counter()
    m2.learn(total_timesteps=S, callback=cb, progress_bar=False)
    print(f"  {S:,} steps in {time.perf_counter()-t0:.0f}s  updates={updates}")
    print(f"  {'episodes':>9s} {'mean_ret':>10s} {'collision%':>11s}")
    for n, r, c in cb.marks[::max(1, len(cb.marks)//12)]:
        print(f"  {n:>9d} {r:>10.1f} {100*c:>10.1f}%")
    if len(cb.marks) >= 4:
        f = np.mean([c for _, _, c in cb.marks[:3]])
        l = np.mean([c for _, _, c in cb.marks[-3:]])
        print(f"  collision first3={100*f:.1f}%  last3={100*l:.1f}%  "
              f"=> {'LEARNING HAPPENED' if l < f - 0.1 else 'NO LEARNING'}")
    venv.close()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "j"):
        run_joint(120, "V2V prior, no curvature cap", v2v=True, cap=False)
    if which in ("all", "f"):
        run_frozen(200, "V2V prior, no curvature cap", v2v=True, cap=False)
    if which in ("all", "c"):
        run_frozen(200, "V2V prior, WITH curvature cap (= frozen env)", v2v=True, cap=True)
