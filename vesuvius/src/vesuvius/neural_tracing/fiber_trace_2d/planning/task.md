# Task: Real Fused Geometric Augmentation Map Tensors

Replace formula-based line/CP augmentation with actual fused map tensors.

The current `StripAugmentTransform` centralizes formulas, but still evaluates
smooth interpolation and affine math when mapping line/CP points. That is not
the required behavior.

Required behavior:

- Construct the complete geometric augmentation maps once per
  `(source_shape, output_shape, params, device)`.
- Bake all geometric stages into those maps at construction time:
  shift, flips, scale, shear, rotation, and smooth offset.
- The image/coordinate path uses a precomputed backward map:
  output pixel -> source pixel.
- The line/CP path uses a precomputed forward map:
  source pixel -> output pixel.
- Line/CP mapping must be lookup/interpolation against the forward map only.
- No smooth interpolation, affine formula stack, iterative solve, dense search,
  or raster/image warp is allowed during line/CP mapping.
- The same map object must be shared by coordinate augmentation, line/CP
  mapping, runner/debug views, and tracing helpers where relevant.
