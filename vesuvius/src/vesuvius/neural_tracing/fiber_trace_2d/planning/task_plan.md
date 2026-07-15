# 3D Fiber GPU-Side Target Materialization Plan

## Scope

Move full dense 3D direction/presence target construction out of
DataLoader workers. Workers should return image data plus compact target
descriptors; the main training process realizes full target tensors on the
training GPU.

This task does not change model architecture, patch size, sample order,
augmentation semantics, VC3D volume sampling, or loss definitions.

## Current Problem

- `FiberTrace3DLoader.load_sample(...)` calls `_build_targets(...)` in the
  worker process.
- For NML samples, `_build_targets(...)` currently:
  - transforms line points to output patch coordinates;
  - rasterizes segment neighborhoods into full `positive` and tangent volumes;
  - encodes a dense six-channel Lasagna 3x2 direction target;
  - builds dense direction/presence masks.
- Benchmark profiling shows the worker-side encode/tensor target stage can be
  hundreds of milliseconds per patch and creates large IPC payloads.

## Implementation Plan

### 1. Split Batch Data From Dense Targets

- Introduce compact target-spec dataclasses in
  `fiber_trace_3d.loader`, for example:
  - `FiberTrace3DTargetSpec`;
  - `FiberTrace3DDenseLineSpec`;
  - `FiberTrace3DCpOnlySpec`.
- Update `FiberTrace3DSample` / `FiberTrace3DBatch` so worker-returned batches
  carry:
  - `volume`;
  - `valid_mask`;
  - sample/record/control-point metadata;
  - `cp_local_zyx` and `crop_origin_zyx`;
  - compact per-sample target descriptors.
- Do not include worker-built `direction_target`, `direction_weight`,
  `direction_mask`, `presence_target`, or `presence_mask` in the normal
  DataLoader batch payload.

### 2. Build Compact Target Metadata In Workers

- Replace worker `_build_targets(...)` calls with a compact metadata builder.
- For NML dense supervision:
  - reuse `_line_window_for_labels(...)`;
  - reuse `_source_points_to_output_np(...)`;
  - return finite transformed output-space `segment_start_zyx` and
    `segment_end_zyx`;
  - keep only segments that overlap or may overlap the patch after
    `presence_radius_voxels` expansion;
  - fail loudly if no patch-overlapping segment remains.
- For non-NML CP-only supervision:
  - return `cp_local_zyx`;
  - return the transformed local tangent used by the current CP-only path.
- Preserve all existing invalid-sample/data-failure behavior. This change
  only moves dense realization; it does not hide malformed fibers.

### 3. Represent Variable Segment Counts Compactly

- In `load_batch(...)`, pack per-sample segment arrays into a padded tensor or
  flat tensor plus offsets:
  - preferred: `segment_start_zyx_flat`, `segment_end_zyx_flat`,
    `segment_offsets`, `segment_counts`;
  - this avoids Python object lists across the hot training path.
- Keep the tensors on CPU in worker output.
- `FiberTrace3DBatch.to(device)` transfers compact segment tensors to the
  training device together with image data.

### 4. Add GPU Target Materializer

- Add a GPU-side materializer in `fiber_trace_3d.train` or a small sibling
  module, e.g. `fiber_trace_3d.targets`.
- API shape:

  `materialize_targets(batch, config, device) -> FiberTrace3DPreparedBatch`

- It returns or attaches:
  - `direction_target`;
  - `direction_weight`;
  - `direction_mask`;
  - `presence_target`;
  - `presence_mask`.
- Cache regular coordinate grids and edge-interior masks by
  `(patch_shape_zyx, device)` to avoid rebuilding them every batch.

### 5. GPU Dense-Line Rasterization

- Port `_rasterize_segment_targets(...)` semantics to torch on the GPU.
- Keep the existing search-free segment rasterization logic:
  - clip each transformed segment to the patch AABB expanded by radius;
  - compute a local bounding box;
  - update `min_dist2` and nearest tangent for voxels inside that bbox;
  - mark positives where `min_dist2 <= radius^2`.
- The first implementation may loop over samples and their finite segments in
  Python, but all per-voxel bbox math and tensor writes must run on the GPU.
- Do not perform a full voxel-by-all-segments dense distance tensor if it
  explodes memory.
- If segment-loop launch overhead remains large, batch/vectorize by grouped
  segment bbox size in a follow-up. Do not silently change semantics.

### 6. GPU CP-Only Target Realization

- Reimplement `_cp_only_targets(...)` on torch:
  - distance from cached grid to `cp_local_zyx`;
  - positive mask within `presence_radius_voxels`;
  - transformed local tangent broadcast to target voxels;
  - direction mask gated by `valid_mask & positive`.
- This path should not create NumPy grids.

### 7. Torch Lasagna 3x2 Encoding

- Add torch equivalents of:
  - `encode_lasagna_direction_3x2`;
  - `projection_magnitude_weights_3x2`.
- Match the existing NumPy analytic formulas exactly enough for tests.
- Add focused tests comparing torch outputs with the current NumPy helpers on
  representative directions, including degenerate projection cases.

### 8. Presence Mask On GPU

- Move `_presence_loss_mask(...)` realization to torch/GPU:
  - start from `valid_mask`;
  - apply the same negative edge margin rule;
  - cache the interior mask by shape/device/margin.
- Keep positive and negative presence-loss balance unchanged in
  `compute_losses(...)`.

### 9. Training Integration

- In `_next_training_batch(...)`:
  - receive CPU image/valid/metadata batch from DataLoader;
  - transfer compact batch to training device;
  - call GPU target materializer;
  - pass the prepared batch into `_forward_loss(...)`.
- Keep `compute_losses(...)` compatible with a prepared batch that has dense
  target tensors.
- Update TensorBoard visualization and dense test loss to call the same
  materializer before accessing target tensors.

### 10. Benchmark/Profile Output

- Update `--benchmark --load-only --profile` to report:
  - DataLoader wait;
  - batch transfer;
  - worker image/sample time;
  - compact target-spec build time;
  - GPU target materialization time.
- Remove or rename old worker-side target columns:
  - old `target_ms`, `raster_ms`, `encode_ms`, `mask_ms` should not imply
    dense target construction happened in workers.
- Add a clear `target_gpu_ms` column when profiling is enabled.

### 11. Tests

- Run syntax check:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py`
- Run focused tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Add tests for:
  - serial worker batch no longer carries dense target tensors before
    materialization;
  - materialized CP-only targets match old CPU semantics on a small synthetic
    patch;
  - materialized dense-line targets match old CPU semantics on a small
    synthetic line segment;
  - torch Lasagna 3x2 encoding matches NumPy helper outputs;
  - DataLoader multi-worker deterministic sample order remains unchanged.

### 12. Performance Validation

- Use the same approved command shape for the 3D loader benchmark:

  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 40`

- Also run normal benchmark/profile with model work after the load-only path
  improves enough to matter.
- Report mean/median/min/max after warmup and include worker overlap plus
  `target_gpu_ms`.

## Spec Update

- Update `planning/specs.md`:
  - 3D DataLoader workers return compact target metadata, not dense target
    tensors;
  - dense direction/presence targets are realized in the main process on the
    training GPU;
  - NML target descriptors are transformed output-space segments;
  - CP-only descriptors are local CP/tangent metadata;
  - Lasagna 3x2 encoding must have torch and NumPy-compatible semantics.

## Docs Updates

- Update `docs/code_structure.md`:
  - describe the split between worker image loading/compact descriptors and
    GPU target materialization;
  - document the target materializer module/function;
  - document updated benchmark columns.
- Update `planning/local_development.md` only if benchmark command usage or
  column interpretation changes.

## Changelog

- Add a 2026-07-15 changelog entry that 3D training moved dense target
  materialization from DataLoader workers to the main GPU path.

## Explicit Non-Goals

- Do not change label semantics.
- Do not change model architecture, precision, patch size, or augmentation
  ranges.
- Do not remove NML dense-line supervision.
- Do not hide invalid data as successful samples.
- Do not reintroduce worker-side dense target tensor construction as a silent
  fallback.
