"""Test the fix direction: require RETURN TO THE OUTER LANE after overtaking.

Diagnosis so far: in this world "get ahead" is purchasable with lane geometry
(the inner lane is ~0.94 m shorter per lap), so ANY relative-position
criterion -- including "overtook at some point" -- is satisfied by a
lane-parking policy. loiter (2 lines) scores ot_rate 0.93 while the rule
machine and a trained PPO score 0.00-0.23.

The real contest task is not "get ahead" -- it is "overtake, then RETURN to
the following position". The frozen env's own success condition embodies this
(ahead_flag AND delta > 0.3), and its scripted baseline dives in, commits,
then returns to the outer lane.

So the criterion's dependent variable should require the full manoeuvre:
    overtake AND return to the outer lane AND remain ahead.
That is something lane-parking cannot fake: a policy that never returns fails
by construction.

Test it against the same set, including loiter.
"""
import sys

# F18/F19 BOOTSTRAP: resolve helper modules from the REPO, never /tmp.
# (/tmp copies caused a stale-module contamination and made these scripts
#  unrunnable from a clean checkout.)
import os as _os, sys as _sys
_RESULTS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_REPO = _os.path.dirname(_RESULTS)
_SCRIPTS = _os.path.join(_RESULTS, '20260920_phaseC_probe', 'scripts')
for _p in ('/tmp', '/home/zy/car_rl/code0919'):
    while _p in _sys.path:
        _sys.path.remove(_p)
for _p in (_SCRIPTS, _REPO):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np

from _criterion_v2 import WorldV2, OVERTAKE_EPS
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

ZERO2 = np.zeros(2)


class WorldV3(WorldV2):
    """Adds: did the follower return to the OUTER lane after overtaking,
    and did it then hold the lead?"""

    def reset(self, rng):
        obs = super().reset(rng)
        self.returned_outer = False
        self.held_after_return = True
        return obs

    def step(self, acts, leader_action=None):
        obs, done = super().step(acts, leader_action)
        d = self.delta(0)
        if not np.isnan(self.t_first_ahead):
            if self.lane[0] == 0:
                self.returned_outer = True
            if self.returned_outer and d <= 0.0:
                self.held_after_return = False
        return obs, done

    def m_full(self):
        """The full contest manoeuvre: overtook, returned to the outer lane,
        and still ahead at the end."""
        if np.isnan(self.t_first_ahead):
            return 0.0
        if not self.returned_outer:
            return 0.0
        if not self.held_after_return:
            return 0.0
        return 1.0 if self.delta(0) > 0.0 else 0.0


def run(pol, n_ep=40, seed0=123000):
    outer, inner = build_paths()
    deltas, times, fulls, colls, reasons = [], [], [], 0, {}
    for k in range(n_ep):
        w = WorldV3(outer, inner, prior="v2v")
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0 = pol(obs[0], w.L)
            obs, done = w.step(np.stack([a0, ZERO2]), leader_action=ZERO2)
            if done:
                break
        deltas.append(w.m_delta())
        times.append(w.m_time())
        fulls.append(w.m_full())
        reasons[w.reason] = reasons.get(w.reason, 0) + 1
        if w.reason == "collision":
            colls += 1
    return (np.array(deltas), np.array(times), np.array(fulls),
            colls / n_ep, reasons)


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")
    pos = PPO.load(str(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "positive_inworld")))

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    print("=" * 90)
    print("CRITERION v3: require overtake + RETURN to outer lane + hold lead")
    print("=" * 90)
    print(f"  {'policy':>22s} {'mean_delta':>10s} {'ot_rate':>8s} "
          f"{'FULL':>7s} {'coll%':>7s}  reasons")
    print("-" * 90)
    for name, fn in [
        ("zero-action", lambda o, L: np.zeros(2)),
        ("full-throttle", lambda o, L: np.array([1.0, 0.0])),
        ("loiter (skill-less)", loiter),
        ("RULE MACHINE", lambda o, L: baseline_action_ot(frozen_layout(o, L))),
        ("ckpt (out-of-dist)", lambda o, L: ckpt.predict(
            frozen_layout(o, L), deterministic=True)[0]),
        ("in-world PPO", lambda o, L: pos.predict(o, deterministic=True)[0]),
    ]:
        d, t, f, c, rs = run(fn)
        fin = [x for x in t if np.isfinite(x)]
        print(f"  {name:>22s} {d.mean():>+10.3f} {len(fin)/len(t):>8.2f} "
              f"{f.mean():>7.2f} {100*c:>6.1f}%  {rs}")

    print()
    print("=" * 90)
    print("READING")
    print("=" * 90)
    print("  If loiter's FULL drops to ~0 while some policy keeps FULL > 0,")
    print("  the criterion finally measures the manoeuvre, not lane geometry.")


if __name__ == "__main__":
    main()
