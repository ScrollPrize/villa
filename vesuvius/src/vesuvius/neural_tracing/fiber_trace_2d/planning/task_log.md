# Task log: reference calibration of winding phase

## Initial findings

- Solver latent coordinates are `integer + sign*phase` for class B and just
  `integer` for class A.
- Phase `0.5` therefore makes both consecutive H/V crossings half a winding.
  Phase `0` makes same-sheet H/V coincident and puts the full unit step on the
  alternating V-to-next-H crossing, matching the proposed physical model.
- Parallel constraints connect equal orientation classes, so phase cancels
  exactly. Only perpendicular reference constraints identify phase.
- Existing reference gauge calibration aligns inferred output windings after
  solve; it does not estimate the physical H/V phase from reference-to-reference
  measurements.

## Plan review corrections

- Hard-sign-first ranking would make phase zero lose automatically because a
  nonzero measured sign times predicted zero is treated as a violation. The
  raw physical fit therefore excludes sign penalties and reports them only as
  diagnostics.
- The exact alternating coordinates are recorded in `task_plan.md`; direction
  is applied once and the existing BP-calibrated global sign is not reused.
- Only opposite-parity perpendicular observations identify phase. Same-parity
  perpendicular and parallel observations are classified separately, including
  opposite-parity parallel contradictions.
- An empty signed identifying set reports an unidentifiable phase.

## Scope correction

- The first implementation incorrectly reused production effective class
  weights. That hid 19 signed adjacent perpendicular reference rows at the
  current zero `perp_0.5` weight and did not directly answer how H-to-V and
  V-to-H steps differ.
- The corrected diagnostic uses every signed dominant reference perpendicular
  row with unit weight and reports direct directional distributions. No solver
  weight variants are part of the phase experiment.

## 1024 result

- Input: `data/workdir3/crop_traces.zarr`, eight ordered `hendrik_crop1`
  references, 512-base-voxel pieces, quality fraction 0.25, fixed solver scale
  0.822. The raw diagnostic is independent of the supplied solver weights.
- The selected gauge maps even references to V and odd references to H. Across
  all 56 signed perpendicular constraints the unweighted L1 phase is 0.459;
  phase 0.5 loss is 19.164 and fitted loss is 18.787, only a 1.97% reduction.
  Phase zero is much worse (loss 36.159).
- Adjacent raw steps are not close to a zero/one alternation: V-to-H has
  `n=17, mean=0.760, median=0.723`; H-to-V has
  `n=14, mean=0.855, median=1.007`.
- At nominal separation 1.5, V-to-H is `n=11, mean=1.912, median=1.858` and
  H-to-V is `n=8, mean=2.253, median=2.264`. Their differences from the
  adjacent group imply noisy full-winding increments of 1.15--1.40 raw units.
- Same-orientation parallel constraints independently put a nominal full
  winding near 1.2--1.4 raw units: V-to-V step-one median 1.220 and H-to-H
  step-one median 1.268; two-step values are about 2.66--2.70.

## Validation

- `cmake --build volume-cartographer/build --target test_fiber_trace_winding_bp vc_fiber_trace_chunk -j 16`
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 67 cases passed.
- Release 1024 reference run completed with one unweighted distribution table
  and one phase fit; no alternate solver-weight scenario was used.
