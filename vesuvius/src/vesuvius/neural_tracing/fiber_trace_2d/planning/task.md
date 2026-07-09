# Current Task

Unify 2D fiber-strip training, batch loading, prefetch, and augmentation
visualization around the same augment-vis-style loading path.

- The augment-vis path is the intended behavior and must become the shared
  implementation used by training and prefetch too.
- Remove mismatches where training or prefetch still use older NumPy/Python
  strip-coordinate generation instead of the torch-vectorized source geometry
  path tested by augment-vis.
- Remove cache/prefetch mismatches where prefetch uses a separate Python chunk
  store path that does not match VC3D blocking sample/cache behavior.
- Keep the one intended training difference: training needs multiple strip-z
  offsets. Generate the CP-local source strip once, then derive offset strips
  by applying each strip-z offset along the strip normals/frame instead of
  rebuilding unrelated coordinate machinery.
- Prefetch must be independent of the particular random augmentation draws used
  by a training step. For each CP, prefetch an area/chunk set sufficient for the
  maximum configured augmentation envelope, so later random augmentations within
  that configured range should be cache-covered.
- Preserve coordinate-space geometric augmentation: image pixels are sampled
  once from final augmented coordinates, never image-warped after loading.
- Add tests proving augment-vis, training batch loading, and prefetch use the
  same final coordinate generation and cache-aware VC3D sampling path.
