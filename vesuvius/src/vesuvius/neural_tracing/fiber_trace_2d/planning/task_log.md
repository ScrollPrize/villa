# Task log: separate prepass and winding Defect controls

## Findings

- The CLI currently copies `options.bp.mixedUnaryCost` into both winding
  backends, coupling the initial prepass and winding-stage Defect costs.
- The exact assignment consumed by fixed-orientation winding already exists as
  `fixedOrientationByPiece`; it is recorded in CSV but not exported as OBJ.
- The shared ternary OBJ writer can persist the prepass assignment without a
  new geometry/export implementation.
- Independent review clarified that non-fixed alternating winding consumes the
  orientation posterior as its prior and does not charge a second Defect
  unary. The new cost applies to late Defect in fixed mode and to the sole
  joint-grid Defect state in non-fixed joint mode.
- Joint-grid output currently copies its Defect unary into the ordinary BP
  report even when no prepass exists. The winding report needs its own named
  cost diagnostic to avoid conflating these two stages.

## Deviations

- None.

## Validation

- Release build:
  `cmake --build volume-cartographer/build --target test_fiber_trace_winding_bp test_fiberlet_crop_trace vc_fiber_trace_chunk -j 32`
  passed.
- Focused CTest:
  `ctest --test-dir volume-cartographer/build --output-on-failure -R 'test_fiber_trace_winding_bp|test_fiberlet_crop_trace'`
  passed both tests in `0.89 s` total.
- A representative 25%-quality fixed-prepass crop used distinct
  `--bp-mixed-cost 20 --winding-defect-cost 3`. Its console winding summary
  reported `defect_unary_cost=3`; the first consistency CSV row reported
  `bp_mixed_unary_cost=20` and `winding_defect_unary_cost=3`.
- The same run wrote all four exact fixed-prepass artifacts under
  `/tmp/fiber-winding-split-cost/fibers_prepass_{h,v,err,tie}.obj` and retained
  the separate final `/tmp/fiber-winding-split-cost/fibers_{h,v,err,tie}.obj`
  partition. The prepass tie OBJ was empty as specified.
- A second one-message run without `--winding-defect-cost` reported the
  independent default `defect_unary_cost=0.5` while retaining explicit
  `--bp-mixed-cost 20`.
- CLI parsing rejected negative and non-finite winding costs and rejected the
  option without `--bp-only --bp-inference sum-product-mixed`.
- `git diff --check` passed.
