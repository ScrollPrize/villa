# 3D Fiber GPU-Side Target Materialization

The current 3D fiber DataLoader worker path still materializes full dense
target tensors inside each worker:

- six-channel Lasagna 3x2 `direction_target`;
- six-channel `direction_weight`;
- `direction_mask`;
- `presence_target`;
- `presence_mask`.

That creates expensive CPU work, large per-worker tensors, and large IPC
payloads. The worker should only load/synthesize the image patch and return
compact metadata needed to build the targets. Full line/direction/presence
tensors should be realized only in the main training process on the GPU.

Required behavior:

- Keep VC3D coordinate sampling and deterministic sample order unchanged.
- DataLoader workers return CPU image/valid data plus compact target metadata,
  not full direction/presence tensors.
- NML dense supervision returns transformed output-space line segments that
  overlap or may overlap the patch; it does not rasterize them in workers.
- Non-NML CP-only supervision returns compact CP/tangent metadata only.
- Main training process transfers the batch to the training device and then
  materializes direction/presence targets on GPU before loss/visualization.
- Target semantics must match the existing CPU path:
  - NML: dense supervision along overlapping line segments;
  - non-NML: CP-neighborhood supervision only;
  - same presence radius, negative edge mask, valid mask handling;
  - same Lasagna 3x2 ambiguous direction encoding and projection weights.
- Benchmark/profile output should separate image loading from GPU target
  materialization so the remaining bottleneck is measurable.
- No silent fallback to worker-side dense target tensor construction.
