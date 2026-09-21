"""Statistical power of the M1.2 criterion: "win rate >= 0.8 over >= 30 eps".

The plan says ">=30 局，3 seed 集" without justification. Check whether that
sample size can support a 0.8 threshold -- i.e. whether a policy whose TRUE
win rate is at or just above the threshold actually passes, and whether a
policy below it reliably fails.

A threshold criterion is only usable if:
  - a policy at the threshold passes with high probability (not a coin flip), and
  - a materially worse policy fails with high probability.
"""
import numpy as np
from math import comb

THR = 0.8


def p_pass(true_p, n, thr=THR):
    """P(observed win rate >= thr | true rate true_p), n episodes."""
    k_min = int(np.ceil(thr * n))
    return sum(comb(n, k) * true_p**k * (1 - true_p)**(n - k)
               for k in range(k_min, n + 1))


def main():
    print("=" * 74)
    print("M1.2 statistical power:  P(pass | true win rate)")
    print(f"  criterion: observed win rate >= {THR}")
    print("=" * 74)

    for n in [10, 20, 30, 40, 60]:
        k_min = int(np.ceil(THR * n))
        print(f"\n  n = {n} episodes  (need >= {k_min}/{n} wins)")
        print(f"    {'true rate':>10s} {'P(pass)':>9s}  {'':>4s}")
        for tp in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
            p = p_pass(tp, n)
            bar = "#" * int(p * 40)
            print(f"    {tp:>10.2f} {p:>9.3f}  {bar}")

    print()
    print("=" * 74)
    print("READING")
    print("=" * 74)
    for n in [10, 30]:
        p_at = p_pass(THR, n)
        p_below = p_pass(0.6, n)
        print(f"  n={n}: a TRUE-0.80 policy passes with P={p_at:.2f}; "
              f"a TRUE-0.60 policy passes with P={p_below:.2f}")
    print()
    print("  KEY POINT: at n=30 a policy sitting exactly AT the threshold")
    print(f"  passes only P={p_pass(THR,30):.2f} of the time -- a coin flip.")
    print("  So '>= 30 episodes' does NOT support a 0.8 threshold.")
    print()
    print("  Two ways out:")
    print("    (a) raise n so a true-0.8 policy passes reliably "
          f"(n=60 -> P={p_pass(THR,60):.2f})")
    print("    (b) keep n=30 but LOWER the threshold to ~0.65 "
          f"(P(true0.8 passes)={p_pass(THR,30,0.65):.2f}, "
          f"P(true0.6 passes)={p_pass(0.6,30,0.65):.2f})")
    print()
    print("  Recommendation: report the whole per-seed distribution, not just")
    print("  a pass/fail, and set the threshold from the positive control's")
    print("  observed spread rather than from a round number.")


if __name__ == "__main__":
    main()
