# Task: Accelerate Native 3D Trace2CP Hot Path

Use the native 3D Trace2CP profile to plan low-risk acceleration work.

Current measured hot spots:

- `src_read`: 195.995 s wall, 53.85 percent
- `src_coords`: 44.719 s wall, 12.29 percent
- `inference_forward`: 37.148 s wall, 10.21 percent
- `field_sample_lookup`: 33.384 s wall, 9.17 percent
- `lasagna_normal_sample`: 13.553 s wall, 3.72 percent

Constraints:

- Keep the trained inference block size and artifact margin setup:
  `--inference-patch-shape-zyx 128 128 128` and
  `--core-margin-voxels 48` remain supported and should be the target path.
- Do not change model outputs, trace scoring semantics, requested volume scale,
  normalization, or strict requested-level VC3D blocking behavior.
- Replace generic coordinate sampling for regular axis-aligned inference blocks
  with a proper shared VC3D/sampler block-read path.
- Batch missing block reads and model forwards where practical.
- Check whether Lasagna-normal sampling, field lookup, and related tracing
  stages can be further vectorized or batched.
