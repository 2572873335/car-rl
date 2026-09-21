"""FIX the harness bug that contaminated every criterion-v2 measurement.

In _criterion_v2.WorldV2.step, the leader is only scripted when
`leader_action is not None`. My run() helper never passed it, so the leader
fell through to the V2V prior with a1=0 => the leader MATCHED the follower's
speed. Both cars locked, the gap was preserved, and every policy scored
~ -gap0. That is why "loiter tied zero-action" looked like a fix.

Correct harness: always pass leader_action so the leader holds its scripted
cruise speed. Then re-measure the degenerate set AND the rule machine.
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

from _criterion_v2 import WorldV2
from _selfplay_design_probe import build_paths
from _ckpt_as_opponent import frozen_layout
from overtake_env import baseline_action_ot

ZERO2 = np.zeros(2)


def run_fixed(pol, n_ep=40, seed0=123000):
    """LEADER IS SCRIPTED (leader_action passed). This is the correct harness."""
    outer, inner = build_paths()
    deltas, times, tasks, colls, reasons = [], [], [], 0, {}
    for k in range(n_ep):
        w = WorldV2(outer, inner, prior="v2v")
        obs = w.reset(np.random.default_rng(seed0 + k))
        while True:
            a0 = pol(obs[0], w.L)
            # *** pass leader_action so car 1 holds its cruise speed ***
            obs, done = w.step(np.stack([a0, ZERO2]), leader_action=ZERO2)
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


def show(name, pol, n_ep=40):
    d, t, k, c, rs = run_fixed(pol, n_ep)
    fin = [x for x in t if np.isfinite(x)]
    print(f"  {name:>22s} {d.mean():>+10.3f} {len(fin)/len(t):>10.2f} "
          f"{(np.mean(fin) if fin else float('nan')):>10.2f} "
          f"{k.mean():>8.2f} {100*c:>7.1f}%  {rs}")
    return d, t, k, c


def main():
    from stable_baselines3 import PPO
    ckpt = PPO.load("/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip")

    def loiter(o, L):
        return np.array([-0.25 if o[5] < 0.3 else 0.0, 1.0])

    print("=" * 92)
    print("CORRECTED HARNESS (leader scripted at 0.4-0.8 m/s, gap0 0.3-0.9)")
    print("=" * 92)
    print(f"  {'policy':>22s} {'mean_delta':>10s} {'ot_rate':>10s} "
          f"{'mean_t_ot':>10s} {'task':>8s} {'coll%':>8s}  reasons")
    print("-" * 92)
    show("zero-action", lambda o, L: np.zeros(2))
    show("full-throttle", lambda o, L: np.array([1.0, 0.0]))
    show("loiter (skill-less)", loiter)
    show("RULE MACHINE", lambda o, L: baseline_action_ot(frozen_layout(o, L)))
    show("ckpt (out-of-dist)", lambda o, L: ckpt.predict(
        frozen_layout(o, L), deterministic=True)[0])
    show("in-world PPO (trained)", lambda o, L: PPO.load(
        "/tmp/positive_inworld").predict(o, deterministic=True)[0])

    print()
    print("=" * 92)
    print("READING")
    print("=" * 92)
    print("  A valid positive pole must show high ot_rate / task and low coll.")
    print("  If the rule machine or the in-world PPO now overtakes, the 0%")
    print("  seen earlier was purely the harness bug, not a real failure.")


if __name__ == "__main__":
    main()
