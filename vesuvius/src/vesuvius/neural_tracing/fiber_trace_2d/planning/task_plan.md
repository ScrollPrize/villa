# 3D Fiber Sparse Line Supervision Target Plan

## Scope

Replace 3D NML dense-line target materialization with sparse centerline voxel
supervision. The target path should draw line voxel indices, supervise
direction only at those indices, and create dense presence only on the training
GPU in the main process.

This task does not change model architecture, VC3D patch loading,
augmentation semantics, deterministic sample order, or non-NML CP-only target
semantics.

## Better Option: GPU Sparse Indices

Use GPU-side vectorized line-index generation rather than CPU worker
rasterization.

Rationale:

- Workers already return compact transformed segment endpoints.
- CPU-generated ragged index tensors would increase DataLoader IPC payloads
  and still need transfer to the training GPU.
- GPU generation keeps workers compact, preserves the current main-process
  target realization split, and avoids dense distance-volume construction.
- It also lets direction loss gather predictions directly at supervised voxels
  instead of building a full `[B,6,Z,Y,X]` dense direction target.

## Current Problem

- `fiber_trace_3d.targets._materialize_dense_line_sample(...)` loops over
  samples and segments in Python.
- For each segment it builds a local 3D bbox grid, computes closest distance to
  the segment, and marks positives within `presence_radius_voxels`.
- That creates a tube-like target and does unnecessary dense work.
- Benchmark profiling now shows `target_gpu_encode_ms` as a fixed cost because
  the code still creates full dense six-channel direction targets over the
  whole patch.

## Implementation Plan

### 1. Keep Worker Output Compact

- Leave `FiberTrace3DLoader` worker output as compact metadata:
  - image volume;
  - valid mask;
  - CP/crop/sample metadata;
  - transformed output-space segment starts/ends for NML samples;
  - CP/tangent metadata for CP-only samples.
- Do not create dense presence or direction tensors in workers.
- Keep worker-side segment clipping to the patch AABB because it reduces
  metadata and preserves current invalid-data checks.

### 2. Add Sparse Target Representation

- Extend `FiberTrace3DBatch` target fields to support sparse direction
  supervision:
  - `direction_indices_bzyx`: integer `[N,4]` index tensor or equivalent
    separate `b,z,y,x` tensors;
  - `direction_target_sparse`: float `[N,6]` Lasagna 3x2 target values;
  - `direction_weight_sparse`: float `[N,6]` projection weights;
  - optional `direction_tangent_sparse_zyx` for debug/profiling.
- Keep `presence_target` dense because presence loss still uses dense
  negatives.
- Keep `presence_mask` dense, created from `valid_mask` and edge masking on
  the training GPU.
- Remove dependence on dense `direction_target`, `direction_weight`, and
  `direction_mask` in the normal training loss path.

### 3. Vectorized NML Line Rasterization

- For all NML segments in a batch:
  - compute segment deltas in output-space ZYX;
  - compute `steps = ceil(max(abs(delta_zyx))) + 1`;
  - build a padded/vectorized interpolation tensor
    `t = arange(max_steps) / max(steps - 1, 1)`;
  - compute points as `p0 + t * (p1 - p0)`;
  - round to integer voxel coordinates;
  - mask padded entries and out-of-bounds coordinates;
  - de-duplicate repeated voxel indices after rounding.
- Associate each drawn point with its segment tangent.
- Do not apply `presence_radius_voxels` to NML dense-line targets.
- Track timing counters:
  - `line_index_ms`;
  - `line_point_count`;
  - `line_segment_count`.

### 4. CP-Only Compatibility

- Keep CP-only samples radius-based for now:
  - generate CP positive voxels on GPU from the cached patch grid;
  - add those voxels to dense `presence_target`;
  - add their indices and tangent values to sparse direction supervision.
- This preserves existing non-NML semantics while still avoiding dense
  direction-target tensors.

### 5. Dense Presence On GPU Only

- In `materialize_targets(...)`, allocate dense presence tensors on the
  training device:
  - `presence_target = zeros([B,1,Z,Y,X], device=batch.volume.device)`;
  - scatter line/CP positive indices to `1.0`;
  - `presence_mask = valid_mask & cached_edge_interior_mask`.
- No worker path may create or serialize `presence_target` or `presence_mask`.
- Keep positive/negative presence loss balancing in `_forward_loss(...)`
  unchanged unless the dense target shape requires a small adapter.

### 6. Sparse Direction Loss

- Update `_forward_loss(...)`:
  - read predicted direction output `[B,6,Z,Y,X]`;
  - gather predictions at `direction_indices_bzyx`;
  - compute squared error against `direction_target_sparse`;
  - apply `direction_weight_sparse`;
  - reduce over sparse supervised entries only.
- Preserve Lasagna 3x2 ambiguous direction encoding by using the existing torch
  helper on the sparse tangent vectors.
- Remove normal-training reliance on dense `direction_target` and
  `direction_mask`.

### 7. Visualization And Test Paths

- TensorBoard target visualization still needs dense-like images for display:
  - use dense `presence_target`;
  - for direction error views, compute sparse/gathered error or scatter a
    debug-only sparse direction-error volume for the displayed slice only.
- Dense test/evaluation should use the same sparse direction loss path as
  training.
- Any compatibility helper that expects dense direction targets should either
  be updated or explicitly limited to tests/debug.

### 8. Remove Tube Rasterization

- Delete or stop using `_materialize_dense_line_sample(...)` distance-to-line
  bbox rasterization for NML targets.
- Remove benchmark fields that describe bbox voxels for the line-supervision
  path.
- Keep the cached grid only for CP-only radius samples and presence edge masks.

### 9. Benchmark/Profile Output

- Update 3D benchmark columns:
  - replace `gpu_raster`, `gpu_encode`, `bboxM`, and related tube wording with
    sparse target timings:
    - `line_idx_ms`;
    - `line_pts`;
    - `presence_scatter_ms`;
    - `dir_encode_ms`;
    - `dir_gather_ms` where applicable.
- Keep total `target_ms`.
- Continue reporting throughput after DataLoader worker startup; with
  `loader_workers=32`, rows 1-32 are startup-contaminated and should not be
  used as steady-state throughput.

### 10. Tests

- Run focused tests:

  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

- Add or update tests for:
  - raw worker batches still carry no dense presence/direction targets;
  - NML diagonal segment rasterizes expected integer line voxels;
  - NML off-line neighboring voxels are not positive just because they are
    within the old radius;
  - CP-only samples still produce radius-neighborhood positives;
  - sparse direction loss matches a small hand-built expected gather;
  - deterministic DataLoader sample order is unchanged.

### 11. Performance Validation

- Reuse the established benchmark command:

  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 40`

- Report:
  - overall throughput including startup;
  - steady-state throughput excluding the first `loader_workers` rows;
  - mean/median/min/max crop time;
  - target timing breakdown.

## Spec Update

Update `planning/specs.md`:

- NML 3D dense-line supervision is centerline-index based, not a
  radius-expanded tube.
- Dense `presence_target` and `presence_mask` are created only in the main
  process on the training GPU.
- Direction supervision for 3D training is sparse/gathered at supervised line
  or CP voxels; dense full-patch six-channel direction targets are not the
  normal training representation.
- `presence_radius_voxels` applies to non-NML CP-only supervision, not to NML
  dense-line centerline rasterization.
- Benchmark steady-state throughput must account for the first
  `loader_workers` startup-contaminated rows.

## Docs Updates

- Update `docs/code_structure.md`:
  - describe sparse line-index target generation;
  - describe dense GPU-only presence construction;
  - describe sparse direction gather loss.
- Update `planning/local_development.md`:
  - document how to interpret benchmark throughput with 32 workers;
  - document new benchmark timing columns.

## Changelog

- Add a 2026-07-15 entry that 3D NML supervision changed from
  distance-to-segment tube targets to sparse drawn centerline voxel targets,
  with dense presence created only on GPU.

## Explicit Non-Goals

- Do not change VC3D loading, augmentation coordinate semantics, or sample
  order.
- Do not change model architecture, precision, patch shape, or training
  optimizer settings.
- Do not silently fall back to worker-side dense target construction.
- Do not change non-NML CP-only radius semantics in this task.
- Do not add a custom CUDA/Triton kernel in the first implementation.
