# Task Log: Real Fused Geometric Augmentation Map Tensors

## Planning Notes

- The previous implementation centralized formulas in `StripAugmentTransform`
  but still evaluated transform math during line/CP mapping.
- The corrected requirement is to construct concrete fused map tensors:
  `backward_map_xy` and `forward_map_xy`.
- Smooth and affine stages must be baked into those maps during construction.
- Line/CP mapping must be lookup/interpolation against `forward_map_xy` only.
- `planning/specs.md` now defines "fused map" as actual precomputed coordinate
  map tensors, not a shared formula bundle.

## Validation

- Not run yet for this implementation task.
