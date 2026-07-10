# Task: Parallel And Batched Loader I/O

Reduce warm-path loader time by batching and parallelizing the remaining I/O
and cache work.

Required behavior:

- Add a sampler-level batch API for coordinate image loading.
- Prefer one VC3D call per CP sample by flattening a stack of strip-z patches
  into one larger coordinate image, then reshaping the returned image/valid
  data back to `[patches, H, W]`.
- This flattening is functionally valid because VC3D samples each requested
  output pixel from explicit coordinates; spatial adjacency in the request
  should only affect performance/chunk locality, not sampled values.
- If a sampler exposes true native batched coordinate support, use it.
  Otherwise, use the flattened single-call path before falling back to a loop.
- Parallelize CP-level `build_sample` work in `load_batch` with deterministic
  slot ordering.
- Parallelize strip-coordinate cache/descriptor/source loading across the CP
  samples in a batch through that CP-level executor.
- Worker count must be config-driven and default to the machine logical CPU
  count.
- Preserve deterministic sample ordering, skipping behavior, output tensor
  order, and existing augmentation/label semantics.
