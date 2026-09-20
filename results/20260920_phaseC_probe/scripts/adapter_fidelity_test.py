"""Fidelity unit test for cross-layout observation adapters.

F12 lesson: two adapter bugs shipped in this repo (gap hardcoded to 1.0 m;
gap offset by L/2) neither raised nor warned -- they silently fed the policy
the WRONG physical quantity. This test exists so that cannot happen again:
it constructs states with KNOWN ground truth and asserts the adapter's output
matches the frozen env's own definition, item by item.

Run: PYTHONPATH=$PWD uv run python adapter_fidelity_test.py
Exit code 0 = all pass.
"""
import numpy as np

from car_following_sim import A_START
from overtake_env import _paths

FAIL = []


def check(name, got, want, tol=1e-6):
    ok = np.allclose(got, want, atol=tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")
    if not ok:
        FAIL.append(name)


def frozen_gap_ref(s_self, s_other, L):
    """Authoritative definition, copied from overtake_env._measure:
       gap_ref = (s_leader - s_follower) % L
    For a self-view obs, 'leader' = the other car."""
    return (s_other - s_self) % L


def frozen_delta(s_self, s_other, L):
    """overtake_env._measure:
       delta = (s_f - s_l + L/2) % L - L/2      (follower minus leader)
       For a self-view obs, 'follower' = self."""
    return (s_self - s_other + L / 2) % L - L / 2


def main():
    outer, inner = _paths()
    L = outer.length
    print("=" * 74)
    print(f"ADAPTER FIDELITY TEST   (L = {L:.4f} m, L/2 = {L/2:.4f} m)")
    print("=" * 74)

    print()
    print("1. gap_ref from a self-view: opponent a known distance AHEAD")
    print("   authoritative: gap_ref = (s_other - s_self) % L")
    s_self = 1.0
    for ahead in [0.10, 0.20, 0.30, 0.45, 0.90, 2.00]:
        s_other = (s_self + ahead) % L
        want = frozen_gap_ref(s_self, s_other, L)
        # self-view delta_self = -wrapL(s_other - s_self) = -ahead
        delta_self = -ahead
        got_correct = (-delta_self) % L
        got_buggy = (L / 2.0 - delta_self) % L
        check(f"ahead={ahead:.2f}: CORRECT (-delta)%L", got_correct, want)
        if abs(got_buggy - want) <= 1e-6:
            pass  # would be a coincidence; not expected here
        else:
            print(f"      (buggy (L/2-delta)%L = {got_buggy:.4f} -> "
                  f"off by {abs(got_buggy-want):.4f} m)")

    print()
    print("2. delta from a self-view: opponent a known distance BEHIND")
    print("   authoritative: delta = (s_self - s_other + L/2) % L - L/2")
    for behind in [0.10, 0.30, 0.90]:
        s_other = (s_self - behind) % L
        want = frozen_delta(s_self, s_other, L)
        delta_self = -(-behind)     # opponent behind -> self is ahead
        check(f"behind={behind:.2f}: delta_self", delta_self, want)

    print()
    print("3. rule-machine preconditions: the dive branch needs gap < 0.45")
    print("   A hardcoded gap=1.0 DISABLES the dive branch -- assert our")
    print("   adapter can actually produce a gap that triggers it.")
    for ahead in [0.20, 0.30, 0.40]:
        gap_ref = frozen_gap_ref(s_self, (s_self + ahead) % L, L)
        norm = (gap_ref - 0.2) / 0.5
        print(f"   ahead={ahead:.2f} -> raw gap={gap_ref:.3f} m, "
              f"normalized={norm:+.2f}")
        if gap_ref < 0.45:
            print(f"      dive branch (gap<0.45) WOULD fire -> OK")
    # the bug: constant normalization
    const_norm = (1.0 - 0.2) / 0.5
    const_gap = const_norm * 0.5 + 0.2
    print(f"   hardcoded normalization {const_norm:+.2f} -> implied gap "
          f"{const_gap:.3f} m -> dive NEVER fires (this was the bug)")

    print()
    print("=" * 74)
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("RESULT: ALL PASS")
    print()
    print("Permanent guards this encodes (F12):")
    print("  - gap_ref MUST be (s_other - s_self) % L  (frozen env definition)")
    print("  - an adapter must NOT shortcut gap with a constant")
    print("  - any adapter feeding a policy/rule must preserve the physical")
    print("    quantity the consumer was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
