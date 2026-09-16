# Weighted-fit task log

New user-approved experiment: single-anchor iterative weighted fit. Prior placement benchmark planning archived. Initial implementation uses owned-cell observations and fixed sigmas; covariance adaptation is explicitly not part of this first experiment. Existing extraction remains default and comparison baseline.

Independent plan review passed with support-normalization/degenerate-state/NMS/
identity clarifications recorded in plan. Implemented weighted branch in shared
cell fitter, same batch sampler/extraction/export. Owned-only sample stencil,
no NMS neighbor expansion/suppression, no gradient evaluation in resolved config.
Experimental processing identity is isolated, default legacy fingerprint unchanged.
Added fitted-axis tangent angle to anchor metric and20unique+legacy sweep through
approved benchmark-full runner, per-variant immutable results and checkpointed matrix.
Build passed. Approved test suite running. No GT influences fitting.

First test invocation stopped at a new cache-identity fixture passing null JSON
instead of object metadata. Corrected fixture only, rebuilt. Native125cases passed
on rerun; Python/integration in progress. Added direction-qualified arc occupancy
at10/20/30degrees to distinguish positional coverage from useful orientation.

Validation complete:125nativecases,263Pythonpassed8skipped (86.40s).
Approved benchmark-full command with eval_short200_anchor_weighted_fit_20260916_v1.json
launched full21-row matrix; output short200_anchor_weighted_fit_20260916_v1.
Legacy first, same200pieces/model/protocol. Default extractor unchanged.

Interim12/21rows: legacy exactly reproduces69141accepted matches/0.676528%emptybins.
Single-anchor rows approximately47000matches; initialization-only gamma2:
47018/0.630662%emptybins,1.983718%emptybins requiring<=20degree anchor.
Legacy corresponding directional-empty fraction1.765853%. Early iterations
improve conditional angular agreement but not bin coverage. Full matrix still
running; no default promotion or downstream quality claim.

Completed all21 variants, runner exit0, completed.json confirms21. Sum variant
wall875.058s. No variant has an unanchored or single-position piece. Best positional
empty-bin score is center-only0.630662%; legacy0.676528%. Center-only gamma2 has
47018matches (31.997% fewer), directional-empty20deg1.983718% versus legacy1.765853%.
Two-iteration gamma1/q2:47215matches,0.688%empty,1.984%directional-empty.
More iterations worsen empty-bin coverage; conditional angular agreement improves
slightly. No experimental row wins20degree-qualified coverage. No default change.
Full21-row matrix and per-scroll representative comparison persisted in
docs/benchmarks/short200_anchor_weighted_fit_20260916.md, with exact build/test/run
commands and provenance location. git diff --check passed. Existing tests125native
and263Python/8skips passed before sweep; final changes are documentation only.

Scope/limitations: owned-cell-only, one anchor, no NMS, fixed Gaussian widths;
adaptive covariance deferred, not implemented. No downstream fiberlet tracing or
controlled speed benchmark in this placement-only experiment. Fitter/multiplicity/
support/NMS all change together. Preserve as opt-in experiment, not promoted.
