# Plan: Native 3D Trace2CP Shared Live Inference

## Goals

- Fix the current import failure when `fiber_trace_3d` is imported from the
  repo root / package-style `PYTHONPATH`.
- Make native 3D Trace2CP live inference use the same fiber inference adapter
  semantics as `python -m vesuvius.neural_tracing.fiber_trace_3d.infer`.
- Keep tracing behavior, block routing, trusted-core caching, and CLI behavior
  intact except where they currently duplicate shared inference behavior.
- Prepare a clean field-provider boundary for future precomputed inference
  support, without implementing that provider now.

## Non-Goals

- Do not remove, lazy-load, or otherwise hide `FiberTrace3DPredictAdapter` from
  `fiber_trace_3d.__init__` in this task.
- Do not add `--prediction-manifest` or any precomputed `.lasagna.json` tracing
  implementation yet.
- Do not change native 3D Trace2CP search/fusion/scoring semantics.
- Do not change 3D training, prefetch, or whole-volume fiber inference output
  schema except for shared helper imports.

## Implementation Steps

1. **Make Lasagna normal encoding package-safe**
   - Move the `_decode_dir_angle` / `_estimate_normal` helper logic currently
     embedded in `lasagna/preprocess_cos_omezarr.py` into a package-safe shared
     Lasagna module.
   - Add a small helper for Lasagna compact `nx/ny` encoding from raw 3x2
     direction channels so both Lasagna predict3d and fiber predict3d can call
     the same implementation.
   - Update `lasagna/preprocess_cos_omezarr.py` to import and use this shared
     helper instead of defining its own copy.
   - Update `fiber_trace_3d/inference_adapter.py` to import the shared helper
     directly, never from the Lasagna CLI/preprocess script.

2. **Move raw fiber prediction decoding into shared fiber code**
   - Extract the tracer-local grouped raw-output decoder from
     `trace2cp_tool.py` into a shared fiber 3D helper module, likely near
     `fiber_trace_3d.direction` or a new focused `prediction.py`.
   - The helper should decode any `N,C` raw model sample with `C == 6` or
     `C == 7 * branch_count` into:
     `directions_zyx[N,branches,3]`, `presence[N,branches]`,
     `valid[N,branches]`.
   - Reuse `decode_lasagna_direction_3x2_analytic`; do not create another
     direction decoder.

3. **Create a prediction-field boundary for native tracing**
   - Define a small internal protocol/base class for the tracer field object:
     `sample_point_choices_torch(points_zyx, progress_label=None)` plus block
     cache/debug properties used by reporting.
   - Rename or adapt the existing live `NativeTraceFieldCache` to implement
     that boundary.
   - Keep its existing sparse block routing, VC3D strict blocking sampling,
     trusted-core crop, CPU-resident cache, and progress output.

4. **Make the live field cache use `FiberTrace3DPredictAdapter`**
   - Construct `FiberTrace3DPredictAdapter` from the same raw config and
     checkpoint used by the CLI.
   - Use `adapter.load_model(device=...)` instead of directly calling
     `build_fiber_trace_3d_model` and `_load_snapshot`.
   - Use `adapter.preprocess_tile(tile, valid_mask)` instead of direct
     `_normalize_image(...)` in block inference.
   - Use `adapter.run_tile_inference(model, tile, device=...)` for recurrent /
     conditioned-decoder / mixed-precision behavior.
   - Use `adapter.product_tensors_from_output(...)` to normalize live output
     into per-option raw tensors, then concatenate/store those option tensors
     in the existing CPU block cache layout expected by tracing.
   - Use the shared grouped raw-output decoder from step 2 for point sampling.

5. **Preserve current live CLI behavior**
   - `trace2cp_tool.py --checkpoint ...` remains the live inference path.
   - Existing arguments for inference patch shape, core margin, cache size,
     cone/beam/smoothness, sample/fiber selection, and visualization should
     remain valid.
   - If the adapter introduces a mismatch in option count/channel layout, fail
     loudly with a shape/config error.

6. **Tests**
   - Add a package import smoke test that does not put `lasagna/` itself on
     `PYTHONPATH`; expected imports:
     `vesuvius.neural_tracing.fiber_trace_3d`,
     `vesuvius.neural_tracing.fiber_trace_3d.infer`, and
     `vesuvius.neural_tracing.fiber_trace_3d.trace2cp_tool`.
   - Add/adjust unit tests for the shared Lasagna normal encoding helper so the
     old Lasagna predict3d output math is unchanged.
   - Add/adjust unit tests for shared raw fiber branch decoding, including
     single 6-channel direction-only samples, one 7-channel branch, and
     multiple 7-channel branches.
   - Run the focused 3D neural tracing test module after implementation.

## Spec Update

- Add/adjust specs stating that native 3D Trace2CP live inference uses
  `FiberTrace3DPredictAdapter` for model load, tile preprocessing, recurrent /
  multibranch inference, mixed precision, and raw output splitting.
- Add a spec that native 3D Trace2CP prediction access is through a field
  provider boundary, currently with live-checkpoint support only; precomputed
  fiber `.lasagna.json` prediction fields are an intended future provider and
  are explicitly not part of this task.
- Add a spec that Lasagna 3x2 normal estimation and compact `nx/ny` encoding
  are shared package-safe helpers, not imports from
  `preprocess_cos_omezarr.py`.

## Docs Updates

- Update `docs/code_structure.md` to describe the shared Lasagna normal helper,
  the fiber inference adapter, and the native 3D Trace2CP live field provider.
- Note that the tracer still runs sparse live inference from checkpoints for
  now, while whole-volume fiber inference writes precomputed `.lasagna.json`
  outputs through `fiber_trace_3d.infer`.

## Validation Commands

- Import smoke:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q <new import/decode tests>`
- Focused 3D tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Optional CLI smoke if data/checkpoint are available:
  run the existing native `trace2cp_tool.py` command with `--trace-step-limit`
  on one known sample and verify it reaches block inference without import
  errors.

## Changelog Update

- Add one line noting that native 3D Trace2CP live inference now uses the
  shared fiber 3D inference adapter and shared Lasagna/fiber prediction decode
  helpers.
