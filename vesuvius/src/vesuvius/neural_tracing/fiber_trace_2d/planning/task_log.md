# Task log: penalize split-piece Defect boundaries

## Findings

- Same-trace split boundaries already have one hard-continuity measurement,
  but any factor with a Defect endpoint is currently neutral. Consequently the
  existing continuity coefficient does not penalize active-to-Defect borders.
- The penalty must be attached to the prepared edge rather than its individual
  measurements so it is charged once per adjacent-piece boundary.
- Both alternating and joint-grid winding solvers share the prepared edge but
  use separate pair-potential implementations; both require the same term.
- Independent review clarified that the activity regularizer must use
  `orientationTemperature`, must live only on the two Defect-capable configs,
  and must be printed/stored for reproducibility. The prepared-edge continuity
  flag must be authoritative and OR-merged rather than inferred from numeric
  scores.

## Deviations

- None.

## Validation

- Built `vc_fiber_trace_chunk` and `test_fiber_trace_winding_bp` from the
  existing optimized build tree. All 38 winding-BP test cases pass.
- Ran the exact 500-fiber, 512-base-voxel-piece reference benchmark with
  piece-break costs `0`, `1`, `8`, and `32`. The default-zero run reproduced
  728 Defect pieces, 407/451 central Defects, 520 active/Defect continuity
  boundaries, and 89.570% reference accuracy.
- Cost `1` produced 718 Defects, 401/451 central Defects, 510 boundaries, and
  89.517% accuracy. Cost `8` produced 730 Defects, 389/451 central Defects,
  463 boundaries, and 89.548% accuracy. Cost `32` produced 834 Defects,
  367/451 central Defects, 268 boundaries, and 88.940% accuracy.
- The sweep confirms that the regularizer reduces split-piece transitions but
  does not by itself solve the central gap: stronger values increasingly turn
  whole non-central runs into Defect. The feature therefore remains opt-in
  with a zero default.
