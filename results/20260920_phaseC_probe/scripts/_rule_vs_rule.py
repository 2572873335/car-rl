"""Is the frozen rule machine a competent ADVERSARY in the overtaking game?

F5 fix verification showed: with a neutral start, 'do-nothing' still won 60/60
against the rule machine, while 'full-throttle' de-saturated to 32/60.

That points at a deeper issue than the biased start: the rule machine, used as
a per-agent policy, may never actually OVERTAKE -- it was hand-tuned as a TASK
solver (dive inside when aligned, commit, return), not as a defender/attacker
in a race. If it never overtakes, "beat the rule machine" is a meaningless
criterion and P2 must use a TRAINED SNAPSHOT instead.

Direct test: rule vs rule from a neutral start. If both cars just hold station
and nobody ever overtakes, the rule opponent is confirmed passive.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import build_paths
from overtake_env import baseline_action_ot
from _verify_f5_fix import NeutralWorld, selfplay_to_frozen


def rule(o):
    return baseline_action_ot(selfplay_to_frozen(o))


def main():
    n_ep = 40
    print("=" * 78)
    print("RULE vs RULE in the two-car world (neutral start)")
    print("=" * 78)

    lead_changes, finals, lanes_used, reasons = [], [], [], []
    for k in range(n_ep):
        w = NeutralWorld(*build_paths())
        obs = w.reset(np.random.default_rng(4000 + k))
        start_leader = 0 if w.delta(0) > 0 else 1
        switched = 0
        prev = start_leader
        while True:
            a = np.stack([rule(obs[0]), rule(obs[1])])
            obs, _, done = w.step(a)
            now = 0 if w.delta(0) > 0 else 1
            if now != prev:
                switched += 1
                prev = now
            if done:
                break
        lead_changes.append(switched)
        finals.append(w.delta(0))
        lanes_used.append(int(max(w.lane)))
        reasons.append(w.reason)

    print(f"  episodes                : {n_ep}")
    print(f"  start leader distribution: car0={sum(1 for k in range(n_ep) if NeutralWorld(*build_paths()).reset(np.random.default_rng(4000+k)) is not None and False) or 'n/a'} ")
    print(f"  lead changes per episode: mean={np.mean(lead_changes):.2f}, "
          f"max={np.max(lead_changes)}")
    print(f"  episodes with ZERO lead change: "
          f"{sum(1 for x in lead_changes if x == 0)}/{n_ep}")
    print(f"  final |delta| mean      : {np.mean(np.abs(finals)):.3f} m")
    print(f"  did either car USE the inner lane? "
          f"{sum(1 for x in lanes_used if x == 1)}/{n_ep} episodes")
    print(f"  termination reasons     : {dict((r, reasons.count(r)) for r in set(reasons))}")

    print()
    print("=" * 78)
    print("INTERPRETATION for P2")
    print("=" * 78)
    zero = sum(1 for x in lead_changes if x == 0)
    inner = sum(1 for x in lanes_used if x == 1)
    if inner == 0 or zero > n_ep * 0.8:
        print("  The rule machine almost never uses the inner lane / never")
        print("  changes the lead => it is NOT racing, it is just following.")
        print("  => P2 'beat the rule machine' measures nothing.  REPLACE with:")
        print("     (a) win rate vs a TRAINED SNAPSHOT, or")
        print("     (b) a task-level criterion: from BEHIND, does the policy")
        print("         complete an overtake within T seconds? (the frozen")
        print("         env's own success definition)")
    else:
        print("  Rule machine does race meaningfully; P2 is usable with a")
        print("  neutral start.")


if __name__ == "__main__":
    main()
