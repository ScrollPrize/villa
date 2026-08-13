# Unoriented Normal Smoothing

## Problem

The volume data stores surface normals as hemisphere-encoded `(nx, ny)` with
`nz = sqrt(1 - nx² - ny²)`.  The normal orientation (sign) is arbitrary:
`n` and `-n` represent the same surface direction.

Naive component-wise smoothing (averaging `nx`, `ny` independently) fails
when neighboring normals have opposite signs, because `(n + (-n)) / 2 = 0`.

## Outer-Product Tensor Representation

Each unit normal `n = (x, y, z)` is represented as the symmetric rank-1
second-moment tensor `N = n nᵀ`:

    N = [[x²,  xy,  xz],
         [xy,  y²,  yz],
         [xz,  yz,  z²]]

Properties:
  - **Sign-invariant**: `n nᵀ = (-n)(-n)ᵀ`
  - Symmetric, positive semi-definite, rank 1, trace = 1
  - Stored as 6 unique elements: `(x², y², z², xy, xz, yz)`

## Smoothing Algorithm

Applied in `fit_data.blur_3d()` to the `nx`, `ny` volume channels:

1. **Decode**: `nx_f = (nx_u8 - 128) / 127`, same for `ny`. Reconstruct
   `nz = sqrt(max(0, 1 - nx² - ny²))`.

2. **Tensor field**: compute the 6 outer-product components at every voxel.

3. **Smooth**: apply separable 3D Gaussian blur to each of the 6 tensor
   components independently.  Because the tensor representation removes
   sign ambiguity, linear averaging is valid.

4. **Recover normal**: at each voxel, reconstruct the 3x3 symmetric matrix
   from the 6 smoothed components.  The smoothed normal is the **principal
   eigenvector** (eigenvector with largest eigenvalue) from `eigh`.

5. **Hemisphere convention**: if the recovered `nz < 0`, flip the normal
   (`n = -n`) to maintain the hemisphere encoding.

6. **Re-encode**: `nx_u8 = round(clamp(nx_f * 127 + 128, 0, 255))`.

## Confidence Measure

The eigenvalue ratio `λ_max / trace` indicates local coherence:
  - ≈ 1: neighborhood normals are consistent
  - < 1: conflicting normals (e.g. at folds or noise)

## Sign-Invariant Distance

For comparing two unoriented normals:

    d(n₁, n₂) = 1 - (n₁ · n₂)²

This is used by `opt_loss_dir.normal_loss_maps()` (the `dir` loss term).
