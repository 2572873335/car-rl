"""Quantify: does the V2V prior change the JOINT self-play verdict?

W5 (self-relative prior, 312 updates): collisions degrade and PIN at 88-100%
in the final phase (bucketed 800k-1.3M -> 100.0% / 99.8% / 88.1%).

V2V (opponent-relative prior, 312 updates): the run ends oscillating around
12-34% -- visibly NOT pinned high.

If this holds up on bucketing, the W6 plan's core assumption ("W5's verdict
transfers to the V2V prior, so we can adopt V2V for comparability") is WRONG,
because the V2V prior materially changes the joint self-play dynamics.

Report the same buckets for both series so the comparison is apples-to-apples.
"""
import re

# F19 BOOTSTRAP: resolve siblings from THIS directory, never /tmp.
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
while '/tmp' in _sys.path:
    _sys.path.remove('/tmp')
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import numpy as np

W5 = "/home/zy/car_rl/code0919/results/20260920_phaseC_probe/probe_run2_joint_selfplay_trend_raw.txt"
V2V = "/tmp/v2v_v4.txt"

BUCKET = 200_000


def load_w5(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.match(r"\s*([\d,]+)\s+(-?[\d.]+)\s+([\d.]+)%", ln)
            if m:
                rows.append((int(m.group(1).replace(",", "")),
                             float(m.group(3)) / 100.0))
    return rows


def load_v2v(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.search(r"\[jointV2V\]\s+steps=\s*([\d,]+)\s+ret=\s*(-?[\d.]+)\s+coll=\s*([\d.]+)%", ln)
            if m:
                rows.append((int(m.group(1).replace(",", "")),
                             float(m.group(3)) / 100.0))
    return rows


def buckets(rows, hi_max):
    out = []
    for lo in range(0, hi_max, BUCKET):
        sel = [c for s, c in rows if lo <= s < lo + BUCKET]
        if sel:
            out.append((lo, len(sel), float(np.mean(sel))))
    return out


def main():
    w5 = load_w5(W5)
    v2 = load_v2v(V2V)
    print(f"W5 samples={len(w5)} (steps to {w5[-1][0]:,})")
    print(f"V2V samples={len(v2)} (steps to {v2[-1][0]:,})")
    print()
    print(f"{'step range':>20s} | {'W5 self-rel':>12s} | {'V2V opp-rel':>12s}")
    print("-" * 52)
    b5 = dict((lo, m) for lo, n, m in buckets(w5, 1_400_000))
    b2 = dict((lo, m) for lo, n, m in buckets(v2, 1_400_000))
    for lo in range(0, 1_400_000, BUCKET):
        a = b5.get(lo)
        b = b2.get(lo)
        sa = f"{100*a:>11.1f}%" if a is not None else f"{'--':>12s}"
        sb = f"{100*b:>11.1f}%" if b is not None else f"{'--':>12s}"
        print(f"{f'{lo:,}-{lo+BUCKET:,}':>20s} | {sa} | {sb}")

    print()
    print("=" * 62)
    print("VERDICT")
    print("=" * 62)
    fin5 = [m for lo, m in b5.items() if lo >= 800_000]
    fin2 = [m for lo, m in b2.items() if lo >= 800_000]
    m5 = float(np.mean(fin5)) if fin5 else float("nan")
    m2 = float(np.mean(fin2)) if fin2 else float("nan")
    print(f"  final phase (>=800k) mean collision:")
    print(f"    W5  (self-relative) : {100*m5:.1f}%")
    print(f"    V2V (opponent-rel)  : {100*m2:.1f}%")
    if m5 > 0.7 and m2 < 0.5:
        print()
        print("  => THE PRIOR CHANGES THE OUTCOME. W5's 'joint self-play fails")
        print("     (pins high)' does NOT transfer to the V2V prior: under V2V")
        print("     the same budget lands at a much lower collision rate.")
        print("     The W6 plan cannot claim comparability AND inherit W5's")
        print("     verdict -- those two are in tension.")
    else:
        print()
        print("  => Outcome broadly comparable across priors.")


if __name__ == "__main__":
    main()
