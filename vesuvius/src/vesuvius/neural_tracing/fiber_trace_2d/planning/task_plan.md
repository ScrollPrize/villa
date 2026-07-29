# Native C++ Trace2CP Inference Scaledown Argument Plan

## Implementation

1. Extend the native fiber prediction scale resolver:
   - accept `inferenceScaledownPower`, default `2`
   - validate the power is in `[0, 30]`
   - keep prediction channel discovery and same-scale validation
   - keep `prediction_to_base = source_to_base * 2**group.scaledown`
   - compute `trace_to_base = prediction_to_base / 2**inferenceScaledownPower`
   - compute `prediction_spacing_in_trace_voxels = 2**inferenceScaledownPower`
2. Extend `vc_fiber_trace_metric`:
   - add `--inference-scaledown-power`
   - default to `2`, matching current Python inference/tracing commands
   - pass it into the resolver
   - print it with the derived scale diagnostics
3. Keep the manifest schema unchanged:
   - no top-level trace-scale aliases
   - no per-group `inference_scaledown`
   - no fiber-coordinate scale override
4. Keep GUI/default resolver callers on the same default power for now.

## Spec Update

- Replace the stale rule that native tracing derives `trace_to_base` directly
  from manifest `source_to_base`.
- State that native precomputed tracing derives persisted prediction scale from
  manifest fields, then divides by the explicit/default inference scaledown
  factor to recover trace coordinates.

## Docs Updates

- Update `docs/code_structure.md` for `fiber_trace_3d/infer.py` manifest output
  and native `vc_fiber_trace_metric` scale derivation.
- Update `planning/status.md`, `planning/task_log.md`, and
  `planning/changelog.md`.

## Tests

- Add C++ resolver coverage for the current Python-written manifest shape:
  `source_to_base=1`, group `scaledown=4`, inference power `2`.
- Add C++ resolver coverage for explicit alternate power and invalid powers.
- Build and run focused native tests:
  - `cmake --build volume-cartographer/build --target test_fiber_trace3d vc_fiber_trace_metric VC3D -j 4`
  - `volume-cartographer/build/bin/test_fiber_trace3d`
