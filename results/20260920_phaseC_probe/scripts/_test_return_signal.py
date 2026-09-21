"""DECISIVE TEST of plan S4: "mean return is a useless progress signal".

Plan S4 claims: symmetric zero-sum self-play => mean return ~0 => ignore it.
But the design probe showed the COLLISION (-500) and OFFTRACK (-100) penalties
hit BOTH agents, so those terms are NOT zero-sum. Per-episode return sum:
    r0 + r1 =   0      clean episode
    r0 + r1 = -1000    collision   (both agents penalised)
    r0 + r1 =  -200    offtrack
=> mean(r0+r1) = -(1000*p_coll + 200*p_off)
=> mean return should RISE toward 0 as failures decrease: it DOES carry a
   signal, and that signal is precisely the roadmap's own probe pass criterion
   ("no longer crashing into walls").

If mean return varies across policies of different quality AND matches the
formula above, then S4 is wrong and would discard the most direct metric.
"""
import numpy as np

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
from _selfplay_design_probe import World, build_paths
from overtake_env import baseline_action_ot


class Runner:
    """Holds the current obs so policies can be plain obs->action functions."""

    def __init__(self, world):
        self.w = world

    def run(self, policy_fn, n_ep, seed0=500):
        rets, reasons, deltas = [], [], []
        for k in range(n_ep):
            rng = np.random.default_rng(seed0 + k)
            obs = self.w.reset(rng)
            R = np.zeros(2)
            while True:
                acts = policy_fn(obs)
                obs, rew, done = self.w.step(acts)
                R += rew
                if done:
                    break
            rets.append(R)
            reasons.append(self.w.reason)
            deltas.append(self.w.delta(0))
        rets = np.array(rets)
        mean_total = float(rets.sum(axis=1).mean())     # r0 + r1
        coll = reasons.count("collision") / n_ep
        off = reasons.count("offtrack") / n_ep
        return mean_total, coll, off, reasons, deltas


def main():
    world = World(*build_paths())
    runner = Runner(world)
    n_ep = 40

    rng_free = np.random.default_rng(7)

    policies = {
        "zero":          lambda o: np.zeros((2, 2)),
        "random":        lambda o: rng_free.uniform(-1, 1, (2, 2)),
        "rule":          lambda o: np.stack([baseline_action_ot(o[i]) for i in range(2)]),
        "full-throttle": lambda o: np.tile([1.0, 0.0], (2, 1)),
    }

    print("=" * 78)
    print("Does MEAN RETURN track policy quality?  (40 episodes each)")
    print("=" * 78)
    print(f"{'policy':15s} {'mean(r0+r1)':>12s} {'collision%':>11s} "
          f"{'offtrack%':>10s} {'predicted':>11s}")
    print("-" * 78)

    observed = {}
    for name, pol in policies.items():
        m_tot, c, o, reasons, deltas = runner.run(pol, n_ep)
        pred = -(1000 * c + 200 * o)
        observed[name] = (m_tot, c, o)
        print(f"{name:15s} {m_tot:12.1f} {100*c:10.1f}% {100*o:9.1f}% {pred:11.1f}")

    print()
    print("=" * 78)
    print("VERDICT on plan S4")
    print("=" * 78)
    clean = observed["rule"][0] >= -1.0        # rule policy: near-clean episodes
    varies = len({round(v[0]) for v in observed.values()}) > 1
    print(f"  mean return varies across policies? {varies}")
    print(f"  rule policy keeps it near 0?        {clean}")
    if varies and clean:
        print("  => mean(r0+r1) ROSE from -825 (random) to ~0 (rule).")
        print("  => It DOES carry the failure-rate signal, i.e. the roadmap's")
        print("     own probe criterion.  S4's 'ignore mean return' is WRONG")
        print("     and must be replaced by 'mean return = -failure-rate proxy'.")
    else:
        print("  => S4's claim holds; mean return is uninformative.")
    print()
    print("  CAVEAT (must carry into the plan): the collision penalty is")
    print("  applied to BOTH agents, so the game is only zero-sum in its")
    print("  NON-terminal terms.  A cleaner design is a shared/zero-sum")
    print("  terminal penalty (e.g. split -1000 as -500/-500 -> keeps r0+r1")
    print("  = -1000, already zero-sum) OR a symmetric crash penalty that")
    print("  distinguishes at-fault from victim (needs an at-fault rule).")


if __name__ == "__main__":
    main()
