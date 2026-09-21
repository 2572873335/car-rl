# Deprecated: early M1.3 anchor scripts (Phase C)

**Status: deprecated. Kept for provenance only — do not use.**

These are the *early* attempts at anchoring the M1.3 threshold, preserved
because `ERRATUM4_f23_correction_wrong.md` cites them. They are superseded by
`results/20260920_phaseC_probe/scripts/_reanchor2.py`.

| file | what it did | why deprecated |
|---|---|---|
| `_baserate.py` | base-rate test under the ORIGINAL hit definition (`switch[1] > 0`) | the definition was wrong: it did not distinguish the re-entry transient |
| `_anchor_m13.py` | first attempt to anchor M1.3 by the completion-time lookback | measured the wrong instant (the counter dives at t~0.45 s, completes at 6-10 s) |
| `_reanchor_m13.py` | re-anchor using the dive-based definition | superseded once the state distribution showed the criterion was anti-correlated with winning |

**Why they are kept rather than deleted**: the erratum cites them, and a
retraction whose evidence cannot be found is not verifiable. Project rule
(iron rule 7): old files are archived, never deleted.

**See**: `ERRATUM4_f23_correction_wrong.md`, `reviews/20260921_phaseC_m1_review1.md`.
