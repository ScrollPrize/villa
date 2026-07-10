# Task Log: Batched Strip Augmentation And Line Mapping

## Planning Notes

- Current profiling indicates sparse line/CP mapping is dominated by many tiny
  PyTorch operations and tiny `grid_sample` calls, not by the amount of point
  data.
- The next implementation should batch work across offsets/patches and replace
  sparse point `grid_sample` with direct bilinear gather against
  `forward_map_xy`.
- VC3D volume sampling remains the likely per-patch external boundary unless an
  existing batched sampler API is found.

## Validation

- Not run yet for this planning-only task.
