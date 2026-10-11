# Broken optimized from first update (archived)

- uv_broken_joint_plan.md independently reviewed and approved.
- Removed joint-solver broken-gradient suppression. Nonzero legacy warmup rejected
  before prepare/GPU allocation. Saved config explicitly records warmup0.
- Default per-piece penalty now rejection_cost times mean broken probability,
  independent of degree. Explicit rejection_weighting=degree retains legacy
  weighting only on request. Optional calibrated sign charge uses same weighting.
- Broken remains a continuous reversible state, not a training-time deletion.
  Final argmax readout unchanged. No solve rerun and completed outputs unchanged.
- Tests cover fixed cost, SparseAdam first-update broken movement both ways,
  rejected warmup, and fixed-charge consistency. Broader run found stale warm-start
  test fixture:303columns specified without shared_anchor layout; corrected fixture.
- Gray fit_reference pixels: retained samples without accepted chosen-winding
  projection (200base-vx limit), not broken; coverage79.0224%retained sample length.
- Validation:38focused tests passed; git diff --check clean. No production rerun.
