# Broken state optimized from the first update

Joint solver: eliminate broken-gradient warmup. Reject nonzero legacy warmup
configs explicitly rather than silently freezing. Fixed per-piece broken cost
defaults to rejection_cost=10 times broken probability; no degree weighting by
default. Preserve explicit degree weighting for historical comparisons only.
Use the same weighting for optional calibrated sign-charge calculation.
No changes to completed outputs, categorical solver, or final argmax readout.

## Spec update
Broken participates continuously from update0, competing with all winding states.
Constraints remain softly retention-gated, no irreversible deletion during solve.

## Docs and tests
Record defaults/legacy behavior, exact expected rejection penalty, unchanged
readout and absence of a rerun. Test gradients at initialization and fixed cost
independence from degree; existing sign calibration and solver regressions.
