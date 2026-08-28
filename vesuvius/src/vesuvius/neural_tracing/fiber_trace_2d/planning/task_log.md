# Task log: extended distance-weighted winding constraints

## Findings

- The raw finite cutoff is applied during shared constraint extraction and is
  currently exclusive.
- H/V-aware winding solvers quantize admitted signed observations to
  half-integer targets, but the prepared factor currently reuses one
  perpendicular score for both H/V orientation and signed winding magnitude.
- Distance decay therefore requires a separate prepared signed-winding weight;
  mutating the original perpendicular score would unintentionally weaken H/V
  classification.
- The independent integer-only winding diagnostic deliberately uses raw targets
  and will retain unscaled evidence.
- Independent review identified that scaling only the signed perpendicular term
  would bias a measured link toward zero winding through its unscaled parallel
  term. The implementation will scale the complete winding contribution while
  retaining the original H/V relation scores.
- The generic constraint config is also used by the legacy parity labeler,
  which cannot represent the widened range. Its default remains `1.5`; H/V CLI
  modes select `4.0` unless explicitly overridden.

## Deviations

- None.

## Validation

- Release build:
  `cmake --build volume-cartographer/build --target test_fiber_trace_winding_bp test_fiberlet_crop_trace vc_fiber_trace_chunk -j 32`
- Focused CTest:
  `ctest --test-dir volume-cartographer/build --output-on-failure -R 'test_fiber_trace_winding_bp|test_fiberlet_crop_trace'`
  passed both tests in `0.89 s` total. Direct binaries reported 24 winding-BP
  and 74 crop/constraint test cases passed.
- Representative fixed-calibration crop used the existing 25%-quality command
  on `data/workdir3/crop_traces.zarr`, with output under
  `/tmp/fiber-winding-cutoff4-smoke`.

  ```bash
  volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/crop_traces.zarr --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --output /tmp/fiber-winding-cutoff4-smoke/fibers --direction-dominance 0.9 --piece-length 512 --bp-only --bp-inference sum-product-mixed --bp-temperature 2.5 --bp-mixed-cost 5 --quality-fraction 0.25 --winding-fixed-phase 0.5 --winding-fixed-scale 1.0
  ```
- The widened cutoff retained `69,713` measured factors and produced
  `1,404,929` candidate states. Effective signed-target/multiplier counts were:
  `-3.5/0.125: 4,599`, `-2.5/0.25: 7,565`, `-1.5/0.5: 10,863`,
  `-0.5/1: 10,852`, `0/1: 861`, `0.5/1: 11,052`,
  `1.5/0.5: 10,316`, `2.5/0.25: 7,679`, and `3.5/0.125: 4,743`.
- The crop completed but did not converge within 500 messages: residual
  `1.8301`, discrete time `19.67 s`, and final H/V/Defect counts `3/5/1353`.
  The preceding `<1.5` hard-sign run had `35,673` factors, `724,843` states,
  residual `1.0636`, about `10.15 s`, and H/V/Defect `13/7/1341`. The requested
  wider experiment is therefore operational but currently worsens convergence
  and directional retention on this crop.
- `git diff --check` passed.
