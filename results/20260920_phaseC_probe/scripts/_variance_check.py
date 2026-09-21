"""Honesty check on the prior-effect finding.

The bucketed means differ a lot (96% vs 26.8%), but the raw series oscillate
between e.g. 0% and 66% WITHIN a bucket. Before claiming "the prior changes
the outcome", quantify:
  (a) within-bucket std (how noisy is each series?)
  (b) the frozen-opponent arm's result under V2V (does it still learn?)
  (c) a mechanistic check: is V2V's lower collision rate LEARNED, or merely
      MECHANICAL (the coupling keeps speeds similar -> fewer chances to crash)?
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

V2V = "/tmp/v2v_v4.txt"
BUCKET = 200_000


def load(path, tag):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.search(rf"\[{tag}\]\s+steps=\s*([\d,]+)\s+ret=\s*(-?[\d.]+)\s+coll=\s*([\d.]+)%", ln)
            if m:
                rows.append((int(m.group(1).replace(",", "")),
                             float(m.group(2)), float(m.group(3)) / 100.0))
    return rows


def main():
    for tag, label in [("jointV2V", "JOINT V2V"), ("frozenV2V", "FROZEN V2V")]:
        rows = load(V2V, tag)
        if not rows:
            print(f"{label}: no data yet")
            continue
        print("=" * 68)
        print(f"{label}: n={len(rows)} samples, steps to {rows[-1][0]:,}")
        print("=" * 68)
        print(f"  {'step range':>20s} {'n':>5s} {'mean coll':>10s} {'std':>8s} "
              f"{'mean ret':>10s}")
        for lo in range(0, rows[-1][0] + BUCKET, BUCKET):
            sel = [(c, r) for s, r, c in rows if lo <= s < lo + BUCKET]
            if sel:
                cs = np.array([c for c, _ in sel])
                rs = np.array([r for _, r in sel])
                print(f"  {f'{lo:,}-{lo+BUCKET:,}':>20s} {len(sel):>5d} "
                      f"{100*cs.mean():>9.1f}% {100*cs.std():>7.1f}% "
                      f"{rs.mean():>10.1f}")
        cs = np.array([c for _, _, c in rows])
        rs = np.array([r for _, r, _ in rows])
        print(f"  OVERALL: coll mean={100*cs.mean():.1f}% std={100*cs.std():.1f}% "
              f"| ret mean={rs.mean():.1f}")
        n = len(rows)
        first, last = cs[:n // 5], cs[-n // 5:]
        print(f"  first 20% mean coll={100*first.mean():.1f}%  "
              f"last 20% mean coll={100*last.mean():.1f}%")
        print()

    print("=" * 68)
    print("MECHANISTIC CHECK: is V2V's lower coll rate learned or mechanical?")
    print("=" * 68)
    print("  Under V2V both cars reference EACH OTHER's speed:")
    print("      v0 = v1 + 0.8*a0 ;  v1 = v0 + 0.8*a1")
    print("  Measured earlier: max-achievable speed asymmetry is 0.765 m/s")
    print("  (V2V) vs 1.162 m/s (self-relative) -- a 34% compression.")
    print("  => the coupling mechanically keeps the pair's speeds close, so")
    print("     FEWER high-closing-speed encounters occur by construction.")
    print("  => a lower collision rate under V2V is at least partly MECHANICAL,")
    print("     NOT evidence that self-play learned better.  The two priors")
    print("     define different GAMES; their collision rates are not")
    print("     directly comparable as a measure of 'learning quality'.")


if __name__ == "__main__":
    main()
