# Task Log

## Scope

- Measure final retained population specifically within the calibrated
  reference winding interval and compare it with all removed pieces.
- Report only split pieces because pruning is piece-level; source-fiber
  aggregation is not a valid removal statistic.
- Count unique graph constraints, not the multiple factor terms emitted from a
  single constraint.

## Decisions

- Use the final authoritative conditioned-result reference calibration for the
  retained winding population.
- Define the area as the inclusive virtual reference interval from the first
  to last loaded reference fiber (`0.0..0.5*(N-1)`).
- Count only active H/V assignments; Defect/Mixed pieces are not used.
- Define piece percentages as `100*removed/(removed+retained_in_range)` and
  `100*removed/retained_in_range`.
- A unique constraint is problematic when it touches a removed piece, is
  neutralized by a retained final Defect, or has any infringed factor term.
  Retained constraints with all terms fulfilled form the comparison population.
- Retained subset construction preserves original constraint indices. Ordinary
  conditioned graphs use those identities after their conditioned prefix;
  oracle-rebuilt graphs use compact local identities.
- Report final active pieces without a calibrated gauge separately.

## Independent Review

- Incorporated exact latent-coordinate gauge semantics, stable-ID subset
  membership, half-ladder validation, inclusive boundaries, uncalibrated
  counts, and validation for malformed reports.

## Validation

- `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiber_trace_winding_bp -j 16`
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 97 test cases
  passed.
- `/tmp/vc_direction_ablation_runner.sh reference-prune oracle 1`: 1024 crop,
  Release build, 57.83 s wall. The final working set contains 454 active pieces
  in reference windings `0.0..12.5`; all 363 removed pieces give 44.43%
  problematic and a removed/in-range percentage of 80.00%. No active final piece
  had an uncalibrated gauge.
- The same run has 69,172 original unique constraints: 33,299 removed-incident,
  10,985 retained but infringed, zero retained-Defect, and 24,888 retained and
  fulfilled. That is 44,284 problematic constraints, 64.02% of the combined
  problematic/fulfilled population, and 177.90% of fulfilled constraints.
- `/tmp/vc_direction_ablation_runner.sh reference-prune inliers`: the ordinary
  non-contiguous subset retained 1,054 pieces and 39,123 constraints. Of 69,172
  original constraints, 30,049 were removed-incident, 12,708 retained and
  infringed, zero retained-Defect, and 26,415 retained and fulfilled. Its
  problematic fraction is 61.81% and problematic/fulfilled is 161.90%.
