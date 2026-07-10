# Task Plan: Batched Strip Augmentation And Line Mapping

## Implementation

1. Add batched map helpers in `augmentation.py`.
   - Keep `StripAugmentTransform` as the per-parameter owner of concrete
     `backward_map_xy` and `forward_map_xy`.
   - Add a helper to stack a sequence of transform maps into:
     - `backward_maps_xy`: `[B, Hout, Wout, 2]`;
     - `forward_maps_xy`: `[B, Hsrc, Wsrc, 2]`.
   - Validate that maps in one stack have compatible shapes/device/dtype.
   - Do not introduce formula-based runtime mapping as a fallback.

2. Replace sparse point `grid_sample` with batched bilinear gather.
   - Add `sample_xy_maps_bilinear(maps_xy, points_xy, valid_lengths=None)`.
   - Input shape should support padded point tensors:
     - maps: `[B, H, W, 2]`;
     - points: `[B, N, 2]`;
     - optional lengths/mask for ragged line lengths.
   - Compute `x0/x1/y0/y1`, gather four map values, blend weights, and return
     `[B, N, 2]`.
   - Mark points outside the source-map bounds as non-finite or return a valid
     mask so downstream filtering remains deterministic.
   - Rewire `StripAugmentTransform.source_to_output_points(...)` to use the
     same gather helper for the single-transform case.

3. Batch line/control-point mapping per source sample.
   - In `FiberStrip2DLoader.build_sample`, build all strip-z offset params and
     transforms before constructing patches.
   - Group offsets by identical `FiberStripAugmentParams` so existing reuse
     remains intact.
   - For each unique params group, map the source line and CP once; for all
     unique params in a source sample, use one batched gather where shapes
     match.
   - Store mapped `(line_xy, control_point_xy)` by params and pass them into
     patch construction as today.
   - Keep line/CP filtering after lookup, but implement it as batched mask work
     where practical. If per-line compaction remains per patch, profile it
     separately so it is not confused with map lookup.

4. Batch coordinate augmentation across offsets.
   - Add a batched coordinate-resampling helper that accepts:
     - coords: `[B, Hsrc, Wsrc, 3]`;
     - valid masks: `[B, Hsrc, Wsrc]`;
     - backward maps: `[B, Hout, Wout, 2]`.
   - Use one batched `grid_sample` for coordinates and one for valid masks.
   - In `build_sample`, create offset grids for all offsets first, stack them,
     apply batched coordinate augmentation for augmented patches, then split the
     result back for VC3D image loading.
   - Preserve the unaugmented path; it may simply stack and pass through without
     geometric resampling.

5. Keep VC3D volume sampling as the explicit I/O boundary.
   - Check whether the current sampler interface supports batched coordinate
     arrays. If not, keep `sample_coords(...)` in a per-patch loop.
   - Make sure all torch-to-NumPy conversion for coordinates happens once per
     patch at that boundary, not before batched coordinate work.
   - Do not add a separate image sampling path.

6. Batch value augmentation after image loading.
   - Extend `apply_value_augmentation` or add a batched wrapper for
     `[B, H, W]` / `[B, 1, H, W]` tensors.
   - Apply brightness/contrast/gamma/noise/blur across the image stack on the
     configured augmentation device.
   - Preserve per-patch augmentation params. For scalar params, use broadcasted
     tensors over `B`; for blur, group by sigma when needed or apply a batched
     kernel if all sigmas can be handled together without changing behavior.
   - Keep deterministic noise seeds per patch.

7. Update profiling to show where batching helps.
   - Split current aggregate loader timing into at least:
     - `map_build`;
     - `line_lookup`;
     - `line_filter`;
     - `coord_aug_batch`;
     - `volume_sample`;
     - `value_aug_batch`.
   - Keep the existing summary columns if required by callers, but record the
     finer-grained keys so the next profile can distinguish lookup cost from
     filtering/compaction and NumPy/VC3D boundaries.

8. Preserve runner/debug behavior.
   - `augment-vis`, `dir-vis`, `line-trace-vis`, benchmark/profile/load-only,
     and training must continue to use the same shared loader path.
   - Single-sample debug calls may use a batch size of one through the same
     batched helpers rather than a separate implementation.

## Spec Update

Update `planning/specs.md` to add:

- Runtime geometric augmentation work should be batched across strip-z offsets
  and patches where compatible.
- Sparse line/CP mapping must use direct bilinear gather against
  `forward_map_xy`; tiny `grid_sample` calls for sparse point lists are not the
  intended implementation.
- Coordinate augmentation should stack `backward_map_xy` tensors and run
  batched dense sampling.
- Value augmentations after VC3D image loading should run as batched tensor
  operations when possible.
- VC3D coordinate sampling remains the external I/O boundary unless the sampler
  exposes a true batched coordinate API.

## Docs Updates

Update `docs/code_structure.md` to describe:

- Batched transform-map stacking.
- Batched sparse line/CP gather.
- Batched coordinate augmentation.
- The remaining per-patch VC3D image-sampling boundary.
- Batched post-load image/value augmentation.

Update `planning/changelog.md`, `planning/status.md`, and
`planning/task_log.md` for this current task only.

## Tests

- Unit-test batched bilinear gather against the single-map lookup on
  representative affine and smooth transforms.
- Unit-test out-of-bounds point handling for batched line/CP lookup.
- Unit-test that `build_sample` maps line/CP data through the batched helper
  rather than one tiny sparse `grid_sample` per patch.
- Unit-test batched coordinate augmentation equivalence to the existing
  per-patch path on fake/local arrays.
- Unit-test batched value augmentation equivalence for deterministic params and
  seeds where exact equality is expected.
- Keep existing loader, augment-vis, deterministic ordering, and cache tests.

## Validation

- Run compile checks for touched Python files.
- Run focused pytest:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run the local profile command before and after implementation when possible:
  `PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --profile`
- Compare at minimum `line_lookup`, `line_filter`, `coord_aug_batch`, and
  `value_aug_batch` timing before/after.
