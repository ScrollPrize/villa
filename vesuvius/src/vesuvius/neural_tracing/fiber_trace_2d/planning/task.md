# Correct Fiber 3D Tiled Inference Output Format

The previous shared tiled inference task left fiber 3D inference in an
incomplete V0 format: raw seven-channel option bundles, no useful scale-space
pyramid, and a custom manifest. Correct this so fiber inference is a normal
Lasagna-style prediction product.

Requirements:

- Fiber inference must create the same OME-Zarr scale-space pyramid behavior
  as regular Lasagna `preprocess_cos_omezarr.py predict3d`.
- Fiber inference must write a `.lasagna.json` manifest, not only a custom
  `fiber_trace_3d_inference.json` side manifest.
- Fiber direction output must be stored in Lasagna's compact ambiguous
  hemisphere encoding:
  - decode the model's six Lasagna 3x2 ambiguous direction channels into one
    ambiguous 3D axis;
  - choose the equivalent sign with non-negative z component;
  - write `nx = round(axis_x * 127 + 128)` and
    `ny = round(axis_y * 127 + 128)` as `uint8`, matching current Lasagna
    normal output encoding;
  - downstream decode reconstructs `nz = sqrt(1 - nx^2 - ny^2) >= 0`.
- Fiber presence is the scalar confidence output:
  - store as `uint8` fixed point with `0 == 0.0` and `255 == 1.0`;
  - `0` is also the invalid/no-data encoding.
- Fiber inference output is only direction angles plus presence. It must not
  perform or write any fiber tracing result.
- Preserve crop-composable output and chunk-existence resume semantics from
  the shared tiled runner. This must be reuse of the existing shared runner
  resume path, not a fiber-specific reimplementation.
- Preserve the configured inference resolution:
  `--scaledown` selects the output resolution relative to the input zarr level,
  and the written data-level OME-Zarr resolution must be exactly that inferred
  output resolution. Coarser pyramid levels are derived from it.
- Keep existing Lasagna `predict3d` compatibility intact.

Clarifications:

- Existence-based resume is not a limitation; it is the intended durable
  resume state and matches Lasagna `predict3d`: chunks on disk are complete,
  missing chunks are recomputed, and no done markers are needed. Fiber only
  changes the adapter product-completeness definition to `presence/nx/ny`.
- Cubic tiled inference is not a limitation; it is the required current shared
  inference shape contract. Do not change it unless a separate future spec asks
  for non-cubic tiles.
- Single written data resolution for fiber is not a limitation if the output
  data level is exactly the configured inference resolution. The pyramid levels
  must still be written above that data level, matching Lasagna OME-Zarr
  behavior.
