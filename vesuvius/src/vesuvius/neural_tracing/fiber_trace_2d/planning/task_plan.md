# Task Plan: Cached Fused Augmentation Maps For Line And CP Warp

## Implementation

1. Refine `StripAugmentTransform` into a real fused map object.
   - Construct all shape/parameter/device-dependent tensors once in
     `__post_init__` or a factory.
   - Precompute scalar affine constants once.
   - Precompute deterministic smooth-offset controls or a source-x lookup
     tensor once per transform object.
   - Remove per-call `torch.Generator` construction and per-call smooth control
     regeneration from point mapping and grid mapping.

2. Make smooth control generation stateless or cached.
   - Replace generator-per-call behavior with either:
     - a deterministic vectorized hash function over control index + seed; or
     - one cached control tensor stored on the transform object.
   - Keep outputs deterministic for identical `(seed, amplitude, stride,
     source_width, device)`.
   - Keep smooth as a direct paired map:
     - backward/image sampling: `source_y += f(source_x)`;
     - forward/line mapping: `source_y -= f(source_x)` before affine forward.

3. Use one batched source-to-output call for line and CP.
   - In `FiberStrip2DLoader._line_and_cp_xy_for_params`, concatenate
     `source_line_xy` and `source_control_point_xy[None]`.
   - Call `transform.source_to_output_points(...)` once.
   - Split the returned tensor into line and CP.
   - Apply finite/in-bounds filtering only to the line rows, not by remapping
     the CP separately.

4. Reuse transformed line/CP across strip-z offsets.
   - Audit the training/loader batch path to find whether the same CP source
     and augmentation params are currently recomputing line/CP for every
     strip-z offset.
   - If yes, compute line/CP once per CP source + params and pass/reuse it for
     all offset patches derived from that source.
   - Preserve per-offset image coordinates and image sampling.

5. Route all relevant geometric consumers through the fused object.
   - `source_coordinate_grid_for_output(...)` should still call the transform's
     grid method.
   - Runner/tracing helpers should not rebuild a separate smooth/affine inverse
     path when a fused transform/map can be used.
   - Do not add a second implementation for TTA, line-trace, dir-vis, or
     augment-vis.

6. Add profiling visibility.
   - Keep the existing `line` profile column.
   - Add or preserve enough focused timing to confirm line/CP mapping time
     drops after batching/caching.
   - Do not insert extra CUDA syncs outside existing profile timing behavior.

## Spec Update

Update `planning/specs.md` to state:

- A fused geometric map object is constructed once per source/output shape,
  augmentation params, and device.
- Shape/params/device-dependent smooth controls and constants are cached in
  that object.
- Line and CP coordinates are transformed together in one vectorized call.
- Shared CP line/CP mapping is reused for all strip-z offsets that share the
  same source and augmentation params.
- No geometric stage may use brute-force inversion, nearest-grid search,
  rasterized image/mask transforms, or iterative solvers.

## Docs Updates

Update `docs/code_structure.md` if implementation changes public module
structure or loader data flow.

Update `planning/changelog.md`, `planning/status.md`, and
`planning/task_log.md` for this current task only.

## Tests

- Add a unit test that monkeypatches smooth control generation or counts calls
  to verify one transform object does not regenerate controls for line and CP
  separately.
- Add a loader test verifying augmented line and CP are transformed through one
  batched `source_to_output_points` call.
- Add/keep round-trip tests for affine and smooth direct maps.
- If line/CP reuse across strip-z offsets changes loader behavior, add a test
  that a multi-offset batch does not recompute line/CP per offset.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Validation

- Run compile checks for touched Python files.
- Run the focused pytest command above.
- If practical, run the existing training `--profile` or augment profile path
  on the same sample before/after and report the `line` column change.
