# Task: Cached Fused Augmentation Maps For Line And CP Warp

Fix the remaining geometric augmentation performance issue in line/CP mapping.

- Build one fused geometric augmentation map object per source/output
  shape + augmentation parameter set.
- The map object must contain all reusable tensors needed by the geometric
  transform, including deterministic smooth-offset control values or lookup
  tensors.
- Do not allocate a random generator or regenerate smooth controls in every
  line/CP transform call.
- Use the same fused map object for:
  - output-to-source coordinate-grid generation for image sampling;
  - source-to-output point mapping for line/CP coordinates;
  - any runner/tracing helper that needs geometric coordinate mapping.
- Transform line points and the CP together in one vectorized torch call.
- Reuse the transformed line/CP for every strip-z offset that shares the same
  CP source and augmentation parameters.
- Preserve coordinate-space geometric augmentation semantics: no raster/image
  warp, no dense nearest search, no brute-force inversion, and no iterative
  solver.
