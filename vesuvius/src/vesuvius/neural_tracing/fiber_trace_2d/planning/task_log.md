# Task log: weighted reference winding diagnostics

## Findings

- The existing reference table counts observations within an inclusive `0.5`
  tolerance and intentionally ignores constraint confidence and distance
  decay. It therefore does not expose the objective that can prefer another
  winding despite a majority of correct observations.
- Joint-grid winding factors use weighted absolute residuals. Their effective
  coefficient is the relevant hypothesis score multiplied by
  `2^-floor(abs(canonical_step))`.
- The existing global sign and per-gauge offset already establish the common
  coordinate frame needed to compare true and inferred reference windings.
- Independent review identified that perpendicular finite-L1 residuals are in
  measurement coordinates and therefore divide latent-coordinate error by the
  solved measurement scale. A subsequent result review exposed that the
  prepared solver incorrectly retained parallel and perpendicular winding
  terms simultaneously. These are alternative hypotheses: preparation,
  reference inference, and diagnostics now retain only the dominant class.
- Parallel distance cutoffs suppress solver evidence but the requested
  parallel-one and parallel-two-plus groups remain diagnostically useful. The
  table will therefore expose raw and cutoff-admitted coefficient separately
  while inference uses only admitted evidence.
- The first implementation retained the old support-first, squared-residual
  `est_w` while group inference used weighted L1. A live result exposed the
  mismatch. `est_w` now comes directly from an `all` group evaluated by the
  same factor scorer. When signed ordering constraints are contradictory, the
  forced-active diagnostic minimizes their violation count before finite
  winding energy; BP itself would otherwise use its Defect state.

## Deviations

- None.

## Validation

- Built the production CLI and focused winding test:

  ```text
  cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiber_trace_winding_bp -j 16
  ```

- `test_fiber_trace_winding_bp` passed all 44 test cases. New coverage checks
  effective canonical bucket boundaries, score and distance decay, cutoff
  admission, non-unit perpendicular measurement scale, reversed global sign,
  one source spanning multiple gauges, deterministic flat weighted-L1 ties,
  empty groups, and invalid values.
- Ran `direction-ablation` against the established Paris4 1024-crop stored
  traces with 25% quality selection, 512-base-voxel pieces, the eight
  `hendrik_crop1` references, fixed phase/scale, 500 messages, and parallel
  cutoff `0.5`. The new five-row-per-reference table appeared after gauge
  calibration and immediately before `reference fiber errors`.
- Reran `direction-ablation` on the established Paris4 1024 crop after the
  dominant-only update. Every `all.infer_w` matched the following `est_w`.
  Prepared evidence partitioned exactly: 87,254 perpendicular plus 16,490
  parallel-same measured incidences, together with 1,722 continuity incidences,
  matched the 105,466 total; the configured cutoff admitted no parallel-other
  winding incidences.
