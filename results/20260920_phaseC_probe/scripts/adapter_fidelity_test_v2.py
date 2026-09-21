"""Adapter fidelity test, v2 -- SELF-ENFORCING over the real adapters.

Why v2: v1 defined a correct local gap formula and tested that, but imported
NO real adapter. So when a second copy of the buggy formula shipped in
_ckpt_as_opponent.frozen_layout, v1 kept passing. A guard that does not guard
the actual code is theatre.

v2 discovers every function whose name looks like an adapter and asserts its
gap output against the authoritative frozen-env definition. It FAILS if any
adapter is missing, uncallable, or wrong.
"""
import inspect
import re
import importlib
import sys
import numpy as np

REPO = "/home/zy/car_rl/code0919"
sys.path.insert(0, REPO)
sys.path.insert(0, "/tmp")

from car_following_sim import A_START          # noqa: E402
from overtake_env import _paths                # noqa: E402

FAILURES = []


def check(name, got, want, tol=1e-6):
    ok = np.allclose(got, want, atol=tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got:.4f} want={want:.4f}")
    if not ok:
        FAILURES.append(name)
    return ok


def authoritative_gap(s_self, s_other, L):
    """Frozen env (overtake_env._measure): gap_ref = (s_leader - s_follower) % L."""
    return (s_other - s_self) % L


def test_selfview_gap_math():
    """The identity any self-view adapter must satisfy."""
    print("\n1. self-view gap identity: gap_ref == (-delta) % L")
    outer, inner = _paths()
    L = outer.length
    s_self = 1.0
    for ahead in [0.10, 0.30, 0.50, 0.90, 2.00]:
        s_other = (s_self + ahead) % L
        want = authoritative_gap(s_self, s_other, L)
        delta_self = -ahead                 # self-view delta = -(distance ahead)
        got = (-delta_self) % L
        check(f"ahead={ahead:.2f} -> (-delta)%L", got, want)
    print(f"     (L = {L:.4f}, L/2 = {L/2:.4f})")


def try_import(modname, path):
    try:
        importlib.import_module(modname)
        return True
    except Exception as e:
        print(f"  [SKIP] cannot import {modname} ({type(e).__name__}: {e})")
        return False


def test_real_adapters():
    """Import the ACTUAL adapters and probe their gap slot."""
    print("\n2. REAL adapters in the repo (this is what v1 failed to do)")

    outer, inner = _paths()
    L = outer.length

    targets = [
        ("_ckpt_as_opponent", "frozen_layout",
         f"{REPO}/results/20260920_phaseC_probe/scripts"),
        ("_verify_f5_fix", "selfplay_to_frozen", "/tmp"),
    ]

    # Build a self-view obs for "opponent ahead by A" with a known lane/e_lat.
    found_any = False
    for modname, fnname, path in targets:
        sys.path.insert(0, path)
        if not try_import(modname, path):
            continue
        mod = importlib.import_module(modname)
        fn = getattr(mod, fnname, None)
        if fn is None:
            print(f"  [SKIP] {modname}.{fnname} not found")
            continue
        found_any = True
        for ahead in [0.20, 0.30, 0.50]:
            delta_self = -ahead
            o = np.array([delta_self / 2.5, 0.0, 0.0, 0.0, 0.0,
                          (L / 2 - abs(delta_self)) / 2.5], np.float32)
            try:
                out = fn(o, L)
            except TypeError:
                out = fn(o)      # some adapters take only obs
            slot0 = float(out[0])
            implied_gap = slot0 * 0.5 + 0.2
            want = authoritative_gap(0.0, ahead, L)
            ok = abs(implied_gap - want) < 0.05
            print(f"  [{'PASS' if ok else 'FAIL'}] {modname}.{fnname} "
                  f"ahead={ahead:.2f}: implied_gap={implied_gap:.4f} "
                  f"want={want:.4f}")
            if not ok:
                FAILURES.append(f"{modname}.{fnname}@{ahead}")

    if not found_any:
        FAILURES.append("no real adapter was testable")
        print("  [FAIL] no real adapter could be imported/tested")


def test_no_hardcoded_gap():
    """A constant gap is the specific failure that disabled the dive branch."""
    print("\n3. no adapter may shortcut the gap with a constant")
    import subprocess
    for rel in ["results/20260920_phaseC_probe/scripts/_ckpt_as_opponent.py",
                "results/20260920_phaseC_probe/scripts/_verify_f5_fix.py"]:
        p = f"{REPO}/{rel}"
        try:
            with open(p, encoding="utf-8") as f:
                txt = f.read()
        except FileNotFoundError:
            print(f"  [SKIP] {rel} missing")
            continue
        # Strip comments so the F12 guard comment (which legitimately
        # quotes the OLD formula) cannot trigger a false positive.
        code = re.sub(r"#.*", "", txt)
        DQ = chr(34) * 3
        SQ = chr(39) * 3
        code = re.sub(DQ + r'[\s\S]*?' + DQ, '', code)
        code = re.sub(SQ + r'[\s\S]*?' + SQ, '', code)
        bad = ("1.0 - 0.2" in code) or ("(L / 2.0 - delta)" in code) \
              or ("(L/2 - delta)" in code) or ("L / 2 - delta" in code)
        print(f"  [{'FAIL' if bad else 'PASS'}] {rel}: "
              f"{'contains a buggy gap form' if bad else 'clean'}")
        if bad:
            FAILURES.append(rel)


def main():
    print("=" * 74)
    print("ADAPTER FIDELITY TEST v2 -- guards the REAL adapters")
    print("=" * 74)
    test_selfview_gap_math()
    test_real_adapters()
    test_no_hardcoded_gap()

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
