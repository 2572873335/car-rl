"""Reviewer tests T8-T10.

T8: ANCHOR GATE A feasibility.  Plan 5.1 says: in the NEW env (8-dim obs),
    the frozen checkpoint (6-dim obs) must reproduce ~10/10 in task mode.
    But a network trained on 6 inputs CANNOT consume 8 inputs.  Verify.

T9: W2 discriminating power.  Plan W2: win rate vs training snapshots in
    [35%,65%].  But a MIRROR of identical weak policies also lands near 50%.
    Test: win rate of do-nothing vs a WEAK pool; win rate of the CHECKPOINT
    vs do-nothing.  If weak/degenerate policies score in-band, W2 passes
    vacuously.

T10: single-sided-failure incentive (plan 3.4).  Under "collision => the
     colliding car loses, other wins", can a LOSING car improve its payoff
     by ramming?  Enumerate the payoff cases and measure collision geometry.
"""
import sys
import numpy as np
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/zy/car_rl/code0919")

from stable_baselines3 import PPO
from gymnasium import spaces
from _selfplay_design_probe import build_paths
from _verify_f5_fix import NeutralWorld
from overtake_env import OvertakeEnv, baseline_action_ot
from _verify_f5_fix import selfplay_to_frozen

CKPT = "/home/zy/car_rl/code0919/ckpt_ot/overtake_final_v1.zip"


def T8():
    print("=" * 78)
    print("T8: ANCHOR GATE A feasi8ility -- obs dim of the frozen checkpoint")
    print("=" * 78)
    ck = PPO.load(CKPT)
    print(f"  frozen checkpoint observation_space : {ck.observation_space}")
    print(f"  plan's new env observation_space    : Box(shape=(8,))  [plan 3.2]")
    print(f"  -> a policy trained on 6 inputs cannot accept 8 inputs.")
    print(f"     Anchor gate A as written ('frozen checkpoint reproduces 10/10")
    print(f"     in the NEW env') is UNDEFINED unless the first 6 slots are")
    print(f"     extracted into a separate 6-dim wrapper env.  The plan does")
    print(f"     not specify this; it lists the checkpoint as a drop-in anchor.")
    # demonstrate the failure
    from gymnasium import spaces as sp
    e = OvertakeEnv(domain_randomize=False)
    o, _ = e.reset(seed=2000)
    print(f"  frozen env obs dim = {o.shape[0]}")
    try:
        a, _ = ck.predict(np.zeros(8, np.float32), deterministic=True)
        print(f"  8-dim input to checkpoint: {a}  <- (SB3 may broadcast; check)")
    except Exception as ex:
        print(f"  8-dim input RAISES: {type(ex).__name__}: {ex}")


def T9():
    print()
    print("=" * 78)
    print("T9: W2 discriminating power -- do WEAK policies land in [35,65]?")
    print("=" * 78)
    ck = PPO.load(CKPT)

    def true_adapter(o, L):
        delta_n, v_self, v_other, e_lat_n, lane, clear_n = o
        delta = delta_n * 2.5
        gap_ref = (-delta) % L
        return np.array([np.clip((gap_ref - 0.2) / 0.5, -1.0, 6.0),
                         delta / 2.5, v_self, e_lat_n, lane,
                         v_other / 0.5 * 1.3], np.float32)

    def winrate(pol0, pol1, n_ep=60, seed0=20000):
        w = l = t = 0
        for k in range(n_ep):
            wd = NeutralWorld(*build_paths())
            obs = wd.reset(np.random.default_rng(seed0 + k))
            while True:
                a0 = pol0(obs[0], wd.L)
                a1 = pol1(obs[1], wd.L)
                obs, _, done = wd.step(np.stack([a0, a1]))
                if done:
                    break
            d = wd.delta(0)
            if d > 0.05:
                w += 1
            elif d < -0.05:
                l += 1
            else:
                t += 1
        return w, l, t

    dn = lambda o, L: np.zeros(2)
    ckpol = lambda o, L: ck.predict(true_adapter(o, L), deterministic=True)[0]
    rnd = np.random.default_rng(3)
    randpol = lambda o, L: rnd.uniform(-1, 1, 2)
    rulepol = lambda o, L: baseline_action_ot(selfplay_to_frozen(o))

    cases = [
        ("do-nothing (car0) vs do-nothing", dn, dn),
        ("do-nothing (car0) vs frozen CKPT", dn, ckpol),
        ("frozen CKPT (car0) vs do-nothing", ckpol, dn),
        ("do-nothing (car0) vs rule machine", dn, rulepol),
        ("random (car0) vs random", randpol, randpol),
    ]
    print(f"  {'car0 vs car1':38s} {'car0 win frac':>14s}")
    for label, p0, p1 in cases:
        w, l, t = winrate(p0, p1)
        print(f"  {label:38s} {w/60:>14.2f}   (W{w} L{l} T{t})")
    print("  => If do-nothing / random score within [0.35,0.65], W2's band is")
    print("     satisfied by DEGENERATE policies -> no discriminating power.")


def T10():
    print()
    print("=" * 78)
    print("T10: single-sided failure (plan 3.4) -- payoff table + geometry")
    print("=" * 78)
    print("  Plan: '一车碰撞/出界即该车判负，另一车判胜' (collision => THAT car loses).")
    print("  The world (both W5 and the plan's) detects collision ONLY by")
    print("  2D distance < 0.12 -- there is NO relative-velocity, NO heading,")
    print("  NO 'who intruded' signal.  => 'which car collided' is UNDEFINED.")
    print()
    print("  Payoff analysis (zero-sum terminal +1/-1, collision -500 both):")
    rows = [
        ("leader about to be overtaken: force contact",
         "leader deemed 'victim' -> +1, follower -500", "REWARDS blocking"),
        ("loser forces a collision",
         "if it is deemed the collider -> -500 + lose", "no gain (still loses)"),
        ("loser forces MUTUAL off-track",
         "both fail, no winner -> 0 for both", "denies opponent +1"),
        ("winner induces follower rear-end",
         "follower -500 + lose, leader +1",
         "EXPLOITABLE: brake-check pays"),
    ]
    for a, b, c in rows:
        print(f"  * {a:44s} | {b:36s} | {c}")
    print()
    print("  Key: the plan keeps collision at -500 for the colliding car but")
    print("  awards the OTHER car the win (+1).  If fault is mis-assigned (and")
    print("  the physics cannot assign it), the 'winner' can engineer contact.")
    print("  This is exactly the mirror image of W5 review I4 -- the plan swung")
    print("  from 'same-death removes blocking' to 'single-death rewards blocking'")
    print("  without adding fault attribution.")


if __name__ == "__main__":
    T8()
    T9()
    T10()
