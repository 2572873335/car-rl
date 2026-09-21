"""Does the W5 verdict SURVIVE the action-semantics change?

W5's conclusions were measured under a SELF-relative speed prior:
    v_cmd = v_self + 0.8*a          (a=0 => keep current speed)
W6's plan switches to the FROZEN V2V prior:
    v_cmd = v_opponent + 0.8*a      (a=0 => match opponent speed)

These are different environments. The W6 plan treats the switch as an
implementation detail, but the whole Phase C strategy rests on W5's verdict
("joint self-play fails; frozen-opponent learns"). If that verdict flips
under the V2V prior, W6 is built on sand.

Test: rerun BOTH arms (joint self-play, and frozen-opponent) under the V2V
prior, at reduced but informative budget. Compare to the W5 numbers
(joint: collision pinned 88-100%; frozen: 100% -> 24%).
"""
import sys

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import tempfile
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from car_following_sim import Car, PurePursuit, A_START
from _selfplay_design_probe import build_paths, wrapL

OBS_DIM, ACT_DIM = 6, 2
V_MAX, ACT_GAIN = 1.3, 0.8
COLLISION_2D, OFFTRACK, SWITCH_COOLDOWN, T_MAX = 0.12, 0.4, 1.5, 30.0


class WorldV2V:
    """Two-car world with the FROZEN V2V speed prior.

    Only ONE thing differs from the W5 world: v_cmd = v_other + 0.8*a.
    Geometry, observation, collision/offtrack thresholds are identical.
    """

    def __init__(self, outer, inner):
        self.outer, self.inner = outer, inner
        self.L = outer.length

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

    def step(self, acts):
        acts = np.asarray(acts, np.float64).reshape(2, 2)
        for i in range(2):
            a_sp, a_ln = np.clip(acts[i], -1, 1)
            self.lane_cd[i] = max(0.0, self.lane_cd[i] - DT_)
            tgt = 1 if a_ln > 0.3 else (0 if a_ln < -0.3 else self.lane[i])
            if tgt != self.lane[i] and self.lane_cd[i] <= 0:
                self.lane[i] = tgt
                self.lane_cd[i] = SWITCH_COOLDOWN
                self.switch[i] = 0.8
        for i in range(2):
            j = 1 - i
            # *** the ONLY change vs the W5 world: opponent-speed prior ***
            v_ref = self.cars[j].v
            a_sp = float(np.clip(acts[i][0], -1, 1))
            v_cmd = float(np.clip(v_ref + ACT_GAIN * a_sp, 0.0, V_MAX))
            lp = self.inner if self.lane[i] == 1 else self.outer
            if self.switch[i] > 0:
                self.switch[i] = max(0.0, self.switch[i] - DT_)
                v_cmd = min(v_cmd, 0.6)
            self.cars[i].step(v_cmd, self.steer[i].omega_cmd(self.cars[i], lp), DT_)
        self.t += DT_
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


from car_following_sim import DT as DT_  # noqa: E402


class JointVec(VecEnv):
    def __init__(self, n_worlds=8, seed=0):
        self.n_worlds = n_worlds
        self.outer, self.inner = build_paths()
        self.worlds = [WorldV2V(self.outer, self.inner) for _ in range(n_worlds)]
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


def snap(model):
    p = tempfile.mktemp(suffix=".zip")
    model.save(p)
    return PPO.load(p)


class Trend(BaseCallback):
    def __init__(self, venv, bucket=100):
        super().__init__()
        self.v, self.b, self.marks, self._n = venv, bucket, [], 0

    def _on_step(self):
        n = len(self.v.ep_reason)
        if n - self._n >= self.b:
            self._n = n
            rec = self.v.ep_reason[-self.b:]
            coll = sum(1 for r in rec if r == "collision") / len(rec)
            ret = float(np.mean(self.v.ep_ret[-self.b:]))
            self.marks.append((self.num_timesteps, ret, coll))
        return True


def main():
    print("=" * 78)
    print("ARM A: JOINT self-play under the V2V prior (W5 said: fails)")
    print("=" * 78)
    venv = JointVec(n_worlds=8, seed=0)
    m = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=512,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]), device="cpu", seed=0, verbose=0)
    assert m.policy.features_extractor.features_dim == OBS_DIM
    cb = Trend(venv, 100)
    STEPS = 16 * 256 * 120          # 120 updates (W5 used 312; this is a probe)
    m.learn(total_timesteps=STEPS, callback=cb, progress_bar=False)
    print(f"  updates={STEPS/(venv.num_envs*256):.0f}")
    print(f"  {'steps':>9s} {'mean_ret':>10s} {'collision%':>11s}")
    for s, r, c in cb.marks[::max(1, len(cb.marks)//8)]:
        print(f"  {s:>9,} {r:>10.1f} {100*c:>10.1f}%")
    first = np.mean([c for _, _, c in cb.marks[:2]])
    last = np.mean([c for _, _, c in cb.marks[-2:]])
    print(f"  collision first={100*first:.1f}%  last={100*last:.1f}%")
    print(f"  => joint verdict under V2V: "
          f"{'STILL FAILS (pinned high)' if last > 0.6 else 'DIFFERENT from W5!'}")
    venv.close()


if __name__ == "__main__":
    main()
