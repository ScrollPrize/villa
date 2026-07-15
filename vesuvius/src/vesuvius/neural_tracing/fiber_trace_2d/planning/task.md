# 3D Fiber Sparse Line Supervision Targets

The current 3D NML target materializer treats dense line supervision as a
radius-expanded distance-to-segment volume. That is the wrong primitive for the
NML line data. For NML fibers, supervision should be built by drawing the
fiber centerline through the output patch and supervising only those drawn line
voxels for direction and positive presence.

Required behavior:

- Keep DataLoader workers compact:
  - workers may return image/valid tensors and compact target metadata;
  - workers must not create dense `presence_target`, `presence_mask`,
    `direction_target`, `direction_weight`, or `direction_mask`;
  - dense presence must be created only in the main training process on the
    training GPU.
- For NML dense-line samples:
  - rasterize clipped output-space fiber segments into integer voxel indices;
  - set positive presence only at the drawn line voxels;
  - supervise direction only at the drawn line voxels;
  - do not use distance-to-segment tube/radius expansion for the line target.
- For non-NML CP-only samples:
  - keep the existing CP-neighborhood supervision unless explicitly changed in
    a separate task.
- Keep presence negatives dense over valid non-edge voxels, with the existing
  valid-mask and negative-edge-mask semantics.
- Keep Lasagna 3x2 ambiguous direction encoding semantics, but encode only the
  supervised line/CP direction samples rather than a full dense six-channel
  direction target where possible.
- Keep deterministic sample order, VC3D coordinate sampling, augmentation
  semantics, and invalid-data handling unchanged.
- Benchmark output should make the sparse line-index path measurable and show
  throughput after DataLoader worker startup.
