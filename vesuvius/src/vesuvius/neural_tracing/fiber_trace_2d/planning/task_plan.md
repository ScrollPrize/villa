# Task Plan: Full-Augmentation Training Prefetch

## Scope

Make prefetch reliable and easy to invoke before V0 2D fiber-strip training.

The key requirement is that prefetch must use the same addressed base-volume
coordinates that training will later load, including full coordinate-space
augmentation.

In scope:

- Verify the current `FiberStrip2DLoader.chunk_requests_for_sample_index` path
  covers:
  - augmentation padding / oversized source strips;
  - final coordinate-space geometric augmentation;
  - deterministic `random_combined_augmentation(sample_index, offset_index)`;
  - every configured strip-z offset.
- Add tests that fail if prefetch request generation diverges from actual
  augmented batch loading.
- Add a training-oriented prefetch command path so the user does not have to
  manually translate training steps into control-point sample count.
- Document the exact local prefetch command for this checkout.

Out of scope:

- Changing VC3D sampling semantics.
- Prefetching Lasagna manifest channels.
- Adding multiprocessing data loaders.
- Changing augmentation behavior, training labels, model code, or batch
  composition.
- Changing cache storage layout.

## Current-State Assessment

The existing loader appears mostly correct for full augmentation:

- `chunk_requests_for_sample_index` derives `patch_shape_hw` from
  `augmentation_padding(...)` when augmentation is enabled.
- It builds the same CP-local line window and local Lasagna normals as
  `build_sample`.
- It loops over all `strip_z_offsets`.
- It uses the same deterministic
  `random_combined_augmentation(config.augment, sample_index, offset_index)` as
  loading.
- It applies `_resample_coords_like_augmentation(...)` before asking the sampler
  for chunk dependencies.
- It only calls `record.sampler.chunk_requests_for_coords(...)` for the
  base-volume image sampler.

The missing pieces are:

- No direct training prefetch CLI that maps `training.max_steps` and
  `training.control_points_per_step` to the required deterministic sample-index
  range.
- No focused regression test proving prefetch and actual loading use matching
  final coordinates under full augmentation.
- Docs do not yet show a training prefetch command.

## Implementation Steps

1. Add training prefetch CLI support
   - Extend `vesuvius.neural_tracing.fiber_trace_2d.train` with:
     - `--prefetch`: run prefetch only, then exit;
     - `--prefetch-steps N`: prefetch the first `N` training steps instead of
       all configured `training.max_steps`;
     - `--prefetch-steps 0`: prefetch all configured training steps;
     - optional `--prefetch-start-step N` if it is straightforward and does not
       complicate the command path.
   - Compute:
     - `start_sample_index = (start_step - 1) * control_points_per_step`;
     - `effective_steps = training.max_steps` when `prefetch_steps == 0`,
       otherwise `prefetch_steps`;
     - `sample_count = effective_steps * control_points_per_step`.
   - Reject negative `--prefetch-steps` values with a clear error.
   - Reuse `FiberStrip2DLoader.prefetch(...)` unchanged for actual chunk
     fetching.
   - Print a short summary with generated/missing/downloaded/error counts.
   - Do not initialize the model, optimizer, TensorBoard writer, or snapshots
     in prefetch-only mode.

2. Keep runner prefetch behavior intact
   - Existing runner command remains valid:
     `runner config.json --prefetch --prefetch-samples <control-point-samples>`.
   - Do not remove or reinterpret `--prefetch-samples`; it is still useful for
     manual deterministic sample ranges.

3. Add full-augmentation prefetch regression test
   - Use fake/local Zarr arrays and the existing fake sampler path.
   - Enable augmentation with nonzero geometry settings.
   - Monkeypatch or wrap the sampler so:
     - `sample_coords(coords, valid)` records the actual final load coordinates;
     - `chunk_requests_for_coords(coords, valid)` records the prefetch
       coordinates.
   - Call `loader.chunk_requests_for_sample_index(sample_index)` and
     `loader.build_sample(sample_index)` for the same sample.
   - Assert that prefetch and load saw the same coordinate arrays and valid
     masks for every strip-z offset.
   - Include at least two strip-z offsets so `offset_index`-dependent random
     augmentation is covered.

4. Add training prefetch sample-count test
   - Use a small local config with a `training` section.
   - Monkeypatch `FiberStrip2DLoader.prefetch` or use a fake loader path to
     assert `--prefetch --prefetch-steps 3` calls prefetch with
     `sample_count = 3 * control_points_per_step`.
   - Add a second assertion that `--prefetch --prefetch-steps 0` uses
     `training.max_steps * control_points_per_step`.
   - Assert prefetch-only mode does not create snapshots or run a model step.

5. Optional dry-run/count mode only if needed
   - If implementation/testing shows actual prefetch is awkward to inspect
     without downloading, add `--prefetch-dry-run` to report generated/missing
     chunk counts without fetching.
   - Keep this optional; do not add it unless it materially improves usability
     or testing.

## Spec Update

Update `planning/specs.md` with:

- Training prefetch uses the same deterministic sample-index sequence as
  training.
- Training prefetch sample count is
  `effective_prefetch_steps * training.control_points_per_step`.
- `--prefetch-steps 0` means all configured training steps, i.e.
  `effective_prefetch_steps = training.max_steps`.
- Negative `--prefetch-steps` values are invalid.
- Prefetch coordinates must be the final augmented coordinates passed to the
  base-volume sampler, not unaugmented source-strip coordinates.
- Prefetch remains base-volume-only; Lasagna manifest channels are not fetched
  by this prefetch path.
- Prefetch-only mode must not initialize training state or write snapshots.

## Docs Updates

Update `docs/code_structure.md` with:

- The new `train.py --prefetch` mode.
- The relationship between training steps, control-point samples, strip-z
  offsets, and chunk requests.
- The distinction between runner prefetch sample counts and training prefetch
  step counts.

Update `planning/local_development.md` with:

- Exact local command for prefetching, using the VC3D binding path:

  ```bash
  PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --prefetch --prefetch-steps 10
  ```

- Exact command for prefetching all configured training steps:

  ```bash
  PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --prefetch --prefetch-steps 0
  ```

## Testing Plan

1. Unit/regression tests
   - Prefetch coordinate generation matches actual augmented load coordinates
     for a local fake-array config with augmentation enabled.
   - Training prefetch computes the expected deterministic sample count from
     `prefetch_steps * control_points_per_step`.
   - `--prefetch-steps 0` computes the expected sample count from
     `training.max_steps * control_points_per_step`.
   - Negative `--prefetch-steps` values are rejected.
   - Training prefetch-only mode does not write checkpoints.

2. Existing focused tests
   - Re-run:

     ```bash
     PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
     ```

3. Local command smoke
   - Run the train CLI help after changes.
   - If local cache/network conditions allow, run a small prefetch command with
     `--prefetch-steps 1`.
   - Do not use `PYTHONNOUSERSITE=1`.

## Changelog Update

Add a dated changelog entry because this changes the training CLI and
documented workflow.

## Review Checklist

- Prefetch uses final augmented coordinates, not raw/oversized source
  coordinates.
- No image-space geometric augmentation is introduced.
- No neural-tracing 3D crop loader is used.
- No Lasagna channel prefetch is added.
- Deterministic sample-index behavior remains unchanged.
- Runner prefetch remains backward-compatible.
- Training prefetch-only mode exits before model/TensorBoard/snapshot setup.
