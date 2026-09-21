"""Reviewer tests for W6 plan review.  T1-T4 (no training, cheap).

T1: _rule_vs_rule.py claims "0/40 episodes used inner lane" -- but it reads
    max(w.lane) AFTER the episode ends (final lane), not "ever used".
    Rerun with an EVER-USED tracker.
T2: Is the neutral start really symmetric?  Two IDENTICAL policies should
    give ~50% win rate for car 0.  (W2's [35,65] band depends on this.)
T3: Collision attribution: is a collision rear-end (attributable) or
    ambiguous side-swipe?  (W6 plan single-sided-failure scoring needs this.)
T4: frozen_layout adapter correctness: gap_ref = (L/2 - delta) % L is
    suspected WRONG (should be (-delta) % L, off by L/2=2.74 m).
    Measure: checkpoint in TASK mode (frozen env) vs in the peer world with
    (a) the plan's adapter, (b) a corrected adapter.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld
from overtake_env import baseline_action_ot, OvertakeEnv
from _verify_f5_fix import selfplay_to_frozen
from _ckpt_as_opponent import frozen_layout
from stable_baselines3 import PPO

CKPT = "/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip"


def rule(o):
    return baseline_action_ot(selfplay_to_frozen(o))


# ---------------------------------------------------------------- T1 ----
def T1():
    print("=" * 78)
    print("T1: does the rule machine EVER use the inner lane?  (vs final-lane bug)")
    print("=" * 78)
    n_ep = 40
    ever_inner = 0
    ever_inner_c0 = 0
    ever_inner_c1 = 0
    final_inner = 0
    lead_changes = []
    for k in range(n_ep):
        w = NeutralWorld(*build_paths())
        obs = w.reset(np.random.default_rng(4000 + k))
        prev = 0 if w.delta(0) > 0 else 1
        sw = 0
        c0_in = c1_in = False
        while True:
            a = np.stack([rule(obs[0]), rule(obs[1])])
            obs, _, done = w.step(a)
            if w.lane[0] == 1:
                c0_in = True
            if w.lane[1] == 1:
                c1_in = True
            now = 0 if w.delta(0) > 0 else 1
            if now != prev:
                sw += 1
                prev = now
            if done:
                break
        lead_changes.append(sw)
        if c0_in or c1_in:
            ever_inner += 1
        if c0_in:
            ever_inner_c0 += 1
        if c1_in:
            ever_inner_c1 += 1
        if max(w.lane) == 1:
            final_inner += 1
    print(f"  EVER used inner (either car) : {ever_inner}/{n_ep}")
    print(f"  EVER used inner (car0)       : {ever_inner_c0}/{n_ep}")
    print(f"  EVER used inner (car1)       : {ever_inner_c1}/{n_ep}")
    print(f"  FINAL lane == inner (the metric _rule_vs_rule.py reports): {final_inner}/{n_ep}")
    print(f"  lead changes: mean={np.mean(lead_changes):.2f} max={np.max(lead_changes)}"
          f"  zero-change episodes: {sum(1 for x in lead_changes if x == 0)}/{n_ep}")
    return ever_inner, final_inner


# ---------------------------------------------------------------- T2 ----
def T2():
    print()
    print("=" * 78)
    print("T2: neutral-start symmetry -- two IDENTICAL policies, car0 win rate?")
    print("=" * 78)
    for label, pol in [("both do-nothing", lambda o: np.zeros(2)),
                       ("both full-throttle", lambda o: np.array([1.0, 0.0]))]:
        wins = losses = ties = coll = 0
        for k in range(60):
            w = NeutralWorld(*build_paths())
            obs = w.reset(np.random.default_rng(5000 + k))
            while True:
                a = np.stack([pol(obs[0]), pol(obs[1])])
                obs, _, done = w.step(a)
                if done:
                    break
            if w.reason == "collision":
                coll += 1
            d = w.delta(0)
            if d > 0.05:
                wins += 1
            elif d < -0.05:
                losses += 1
            else:
                ties += 1
        print(f"  {label:20s}: car0 W{wins} L{losses} T{ties}  (collisions {coll}/60)"
              f"  -> car0 win frac = {wins/60:.2f}")
    print("  (symmetry requires ~0.50; a large deviation means the start is biased)")


# ---------------------------------------------------------------- T3 ----
def T3():
    print()
    print("=" * 78)
    print("T3: collision attribution -- rear-end or side-swipe?")
    print("=" * 78)
    rng = np.random.default_rng(0)
    rear = side = other = 0
    details = []
    for k in range(400):
        w = NeutralWorld(*build_paths())
        obs = w.reset(np.random.default_rng(9000 + k))
        prev = None
        while True:
            a = np.stack([rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)])
            obs, _, done = w.step(a)
            if done:
                break
        if w.reason != "collision":
            continue
        # geometry at the terminal step
        p0, p1 = w.cars[0].pos, w.cars[1].pos
        # lateral offset in the track frame: use each car's own e_lat
        e0 = (w.inner if w.lane[0] == 1 else w.outer).nearest(p0)[1]
        e1 = (w.inner if w.lane[1] == 1 else w.outer).nearest(p1)[1]
        same_lane = w.lane[0] == w.lane[1]
        # relative along-track: who is ahead
        d = w.delta(0)   # >0 car0 ahead
        heading0 = w.cars[0].theta
        heading1 = w.cars[1].theta
        # angle between headings (wrapped)
        dh = abs((heading0 - heading1 + np.pi) % (2 * np.pi) - np.pi)
        sep = np.hypot(*(p0 - p1))
        rec = dict(same_lane=same_lane, delta=d, dh=dh, sep=sep,
                   lane0=w.lane[0], lane1=w.lane[1])
        details.append(rec)
        if same_lane and dh < 0.25:
            rear += 1          # aligned headings, same lane -> along-track encounter
        elif not same_lane or dh > 0.5:
            side += 1          # cross-lane / angled -> side-swipe
        else:
            other += 1
    print(f"  collisions sampled: {len(details)}")
    print(f"  same-lane aligned-heading (rear-end-like): {rear}")
    print(f"  cross-lane or angled (side-swipe-like)   : {side}")
    print(f"  ambiguous                                : {other}")
    if details:
        d = np.array([x['delta'] for x in details])
        dh = np.array([x['dh'] for x in details])
        print(f"  |delta| at collision: mean={np.abs(d).mean():.3f} m "
              f"median={np.median(np.abs(d)):.3f}")
        print(f"  heading mismatch dh : mean={dh.mean():.2f} rad  "
              f"frac dh>0.5: {np.mean(dh > 0.5):.2f}")
    print("  NOTE: the world detects collision ONLY via 2D distance < 0.12;")
    print("        it stores no relative velocity / culprit info.")


# ---------------------------------------------------------------- T4 ----
def T4():
    print()
    print("=" * 78)
    print("T4a: checkpoint in TASK mode (frozen OvertakeEnv), seeds 2000-2009")
    print("=" * 78)
    ck = PPO.load(CKPT)
    env = OvertakeEnv(domain_randomize=False)
    succ = coll = 0
    for s in range(2000, 2010):
        o, _ = env.reset(seed=s)
        while True:
            a, _ = ck.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        if env.term_reason == "success":
            succ += 1
        if env.term_reason == "collision":
            coll += 1
    print(f"  success {succ}/10   collisions {coll}/10   (frozen baseline: 10/10)")

    print()
    print("=" * 78)
    print("T4b: adapter check -- gap_ref from frozen_layout vs corrected")
    print("=" * 78)
    L = build_paths()[0].length
    # construct a state where car1 (self) is 0.30 m BEHIND car0
    o_self = np.array([-0.30 / 2.5, 0.5, 0.5, 0.0, 0.0, 0.0], np.float32)
    delta = o_self[0] * 2.5
    gap_plan = (L / 2.0 - delta) % L
    gap_correct = (-delta) % L
    print(f"  true along-track gap of self behind other = 0.300 m")
    print(f"  frozen_layout  gap_ref = (L/2 - delta) % L = {gap_plan:.3f} m  "
          f"-> normalized {(gap_plan-0.2)/0.5:.2f}")
    print(f"  corrected      gap_ref = (-delta) % L     = {gap_correct:.3f} m  "
          f"-> normalized {(gap_correct-0.2)/0.5:.2f}")
    print(f"  L = {L:.4f} m")

    print()
    print("=" * 78)
    print("T4c: checkpoint as car1 (car0 = do-nothing), peer world")
    print("=" * 78)

    def corrected_layout(o_self, L):
        delta_n, v_self, v_other, e_lat_n, lane, clear_n = o_self
        delta = delta_n * 2.5
        gap_ref = (-delta) % L
        return np.array([
            np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
            delta / 2.5,
            v_self,
            e_lat_n,
            lane,
            v_other / 0.5 * 1.3,
        ], np.float32)

    for label, mp in [("plan's frozen_layout", frozen_layout),
                      ("corrected (-delta)%L", corrected_layout)]:
        coll = ended_ahead_c1 = used_inner_c1 = 0
        v1 = []
        for k in range(40):
            w = NeutralWorld(*build_paths())
            obs = w.reset(np.random.default_rng(12000 + k))
            c1_in = False
            while True:
                a0 = np.zeros(2)
                a1, _ = ck.predict(mp(obs[1], w.L), deterministic=True)
                obs, _, done = w.step(np.stack([a0, a1]))
                if w.lane[1] == 1:
                    c1_in = True
                if done:
                    break
            if w.reason == "collision":
                coll += 1
            if w.delta(0) < -0.05:
                ended_ahead_c1 += 1
            if c1_in:
                used_inner_c1 += 1
            v1.append(np.mean(w.cars[1].v))
        print(f"  [{label:22s}] coll {coll}/40  car1 ahead@end {ended_ahead_c1}/40"
              f"  car1 used inner {used_inner_c1}/40  mean v {np.mean(v1):.3f}")


if __name__ == "__main__":
    T1()
    T2()
    T3()
    T4()
