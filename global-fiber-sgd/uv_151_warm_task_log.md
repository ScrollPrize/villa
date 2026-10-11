# 151-winding continuation and contrasting visualization

- Plan: uv_151_palette_plan.md; independently reviewed and approved.
- Configurable categorical count, warm expansion, validation/readout implemented.
- 0–150 inclusive means 151 candidates plus broken. Geometry is frozen;
  appended widths repeat the last saved width. Broken probability is preserved;
  old surviving probabilities scale by 100/151, new mass is uniform over 51 buckets.
- Saturated golden-angle colors with OKLab interpolation, opaque stable overdraw,
  2px lines and 1000px-high equal-scale LZW TIFF maps implemented.
- Validation: 17 tests passed (physical coordinate preservation including boundary
  crossings; categorical costs; projection/cache; TIFF; actual line width/overdraw).
- Runs: run_uv_categorical_ccw_151.json and run_uv_categorical_cw_151.json,
  10 full sweeps, batch2048, shared 12GiB reservations on separate GPUs.
- Rerender uses run_reference_comparison100.json and validated projection cache;
  no correspondence search or reference-based optimization.
- Independent code review found no launch blockers. Additional layout/visualization
  regressions passed:35 total tests. External-cache reuse/rejection also tested.
- Both GPU grants active, distinct devices, observed8.4GiB each. First sweeps
  ~79s including audit. Step1 removes four sign violations in both conventions,
  but this is not evidence of correct global placement.
- Old100-sweep comparison rerender completed exit0; six fiber maps plus reference
  map and TIFF detail crops. New151 comparison config reuses same projection cache.
- Both151candidate solves completed exit0 in794.13s/797.55s. Final retained
  length25.5162%/25.4423%, zero retained sign violations, occupied indices1–97.
  No additional hypothesis became MAP. Rejection59.9263%/60.0207%. This warm
  expansion is not promoted; old100candidate solutions remain unchanged.
- Complete regression suite rerun:35passed. New-map renderer and full persisted
  source/constraint/geometry audit launched using approved configs.
- Persisted audit completed exit0 for both directions: all6,606,100observations
  present, identities/partition intact, geometry and widths unchanged. All four
  original sign violations per run removed by endpoint rejection, none corrected
  while retained. Final continuation links retained0/466. Audit:uv_151_audit.md.
- New maps complete exit0 in158.81s, all six fiber maps1000px high; same cached
  projections, no rematching. Old-map rerender330.90s. Final results and exact
  configurations documented in uv_151_results.md. GPUs released. No default
  solution promotion and no commit requested/performed.
