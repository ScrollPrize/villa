# Task Log: 3D Fiber GPU-Side Target Materialization

## Planning Notes

- Stopped the in-progress 32-worker benchmark when requested.
- Restored the checked-in S1A 3D config from the temporary
  `loader_workers: 32` benchmark value back to `loader_workers: 8`.
- Inspected the current 3D target-generation path:
  - `FiberTrace3DLoader.load_sample(...)`;
  - `_build_targets(...)`;
  - `_rasterize_segment_targets(...)`;
  - `compute_losses(...)`.
- Confirmed that dense worker-side target construction currently creates:
  - `direction_target`;
  - `direction_weight`;
  - `direction_mask`;
  - `presence_target`;
  - `presence_mask`.
- Wrote a plan to move dense target realization to the main GPU path while
  keeping worker output compact.

## Deviations / Simplifications / Deferred Items

- No implementation has been done for this task yet. This is currently a plan
  for review.
- The proposed first GPU rasterizer may still loop over samples/segments in
  Python while doing all per-voxel bbox math on GPU. If that is too slow, the
  follow-up is grouping/vectorizing segment bboxes without changing label
  semantics.

## Validation

- No tests run yet for this planning-only step.
