# Task: Batched Strip Augmentation And Line Mapping

Reduce GPU overhead from many small per-patch augmentation operations by
batching the strip-coordinate augmentation, line/control-point mapping, and
post-load image value augmentations across all patches generated for one loader
sample or training batch.

Required behavior:

- Keep the current fused geometric map semantics: concrete `backward_map_xy`
  and `forward_map_xy` tensors are still the only geometric augmentation
  representation used by runtime coordinate and line/CP paths.
- Build all patch transforms for a source sample first, then stack compatible
  maps and tensors so coordinate augmentation runs as batched tensor work rather
  than many small per-patch calls.
- Map all line/control-point tensors for the same source sample in one batched
  sparse lookup operation where possible.
- Replace tiny `grid_sample` calls for sparse line/CP points with a batched
  bilinear gather over `forward_map_xy`.
- Keep image loading through the existing VC3D coordinate sampler. If VC3D only
  supports one coordinate patch per call, leave that loop explicit and document
  it as the external I/O boundary.
- After image loading, batch value augmentations over the loaded patch stack
  where possible.
- Preserve deterministic sample/augmentation ordering and existing output
  semantics.
