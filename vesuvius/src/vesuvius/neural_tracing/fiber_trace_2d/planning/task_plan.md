# Training Batch Config Validation, Prep Slowdown, And Pipeline Depth Plan

## Implementation

- Remove the hard-coded default patch-count warning from `run_training`.
- Add a small train/benchmark validation helper that requires
  `training.control_points_per_step` to match loader `batch_size`, because
  training loads exactly one CP batch per step.
- Keep non-default flattened patch counts valid:
  `batch_size * strip_z_offset_count` may be any positive configured value.
- Replace per-patch CUDA Gaussian blur in batched value augmentation with a
  grouped batched blur so variable per-patch blur sigmas use two grouped
  convolutions for the whole patch batch.
- Increase the default CUDA training pipeline depth/worker count so normal
  runs keep several whole-batch loaders in flight.
- Add explicit `pipeline_depth` and `pipeline_workers` to the example config.
- Print effective pipeline settings once at startup.

## Spec Update

- Document that top-level `batch_size` is the CP-sample count and training
  `control_points_per_step` must match it.
- Document that changing `strip_z_offset_count` changes the flattened CNN patch
  count without warning.
- Document that batched value augmentation should avoid per-patch CUDA blur
  loops.
- Document the updated pipeline defaults and startup print.

## Docs Updates

- Update `docs/code_structure.md` config/training notes with the same meaning.
- Update augmentation implementation notes for grouped batch blur.
- Update training config docs for the loader pipeline defaults.

## Testing

- Add a focused regression test for rejecting mismatched
  `control_points_per_step` and loader `batch_size`.
- Add a regression test that batched blur matches the single-patch path for
  different blur sigmas.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

- Add a short entry for removing the default patch-count warning, validating
  training batch config, and batching value-augmentation blur.
