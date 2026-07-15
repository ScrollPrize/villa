# 3D Prefetch Must Follow 2D Streaming Prefetcher

The 3D prefetcher currently prints
`fiber_trace_3d prefetch: generating chunk requests ...` and then generates all
chunk requests serially before downloads begin. This differs from the 2D
prefetcher, which streams dependency generation and downloads concurrently with
progress output.

Required behavior:

- Rework the 3D prefetcher to follow the 2D prefetcher architecture as closely
  as possible.
- Use parallel dependency/sampler producer workers controlled by
  `prefetch_sampler_workers`.
- Use bounded parallel download workers controlled by `prefetch_workers`.
- Stream producer results into the same cache-hit/missing/download
  classification flow instead of generating the full request set first.
- Consume producer results in raw deterministic sample-index order before chunk
  classification/enqueueing, matching the 2D prefetcher.
- Prioritize downloads by the earliest raw deterministic sample index that
  requested each chunk, matching the 2D prefetcher.
- Report live sample/dependency progress and download progress during the whole
  run, including deterministic safe-prefix `idx`.
- Preserve VC3D-provided chunk metadata and Python atomic temp-file download
  behavior.
- Keep only the necessary 2D-vs-3D differences:
  - 3D samples generate one CP-centered 3D patch dependency request set;
  - 3D has no strip-z offset loop;
  - 3D has no 2D top-view prefetch branch;
  - 3D uses the 3D augmentation-envelope coordinate volume.
- Update specs/docs/tests/changelog for the 3D streaming prefetch behavior.
