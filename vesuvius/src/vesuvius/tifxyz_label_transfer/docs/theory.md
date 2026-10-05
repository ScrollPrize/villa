# Theory

TIFXYZ maps pixels on a flattened surface to 3D scan coordinates. Label
transfer uses those coordinates to move annotations between flattenings
of the same surface.

## Mapping

For each target pixel, the mapper finds a nearby source triangle in 3D
and uses it to calculate a position on the source label canvas. Labels
are sampled with nearest-neighbour interpolation, preserving their values.
Matches beyond the distance threshold are rejected. The default threshold
is derived from the target surface spacing.

Output size is inferred from the source label dimensions and TIFXYZ scales.
This assumes a complete, unrotated, unflipped source canvas. If the surfaces
use different scan coordinate frames, supply a registration transform.
A sampled geometry preflight checks coverage before full-resolution transfer.

## Canvas alignment

A label image can be shifted relative to its source TIFXYZ canvas. Geometry
alone cannot detect that shift. The alignment tools compare CT renders to
measure it; pass the correction explicitly with `--label-canvas-offset DY DX`.
Label pixel `(i, j)` then refers to source canvas position `(i + dy, j + dx)`.

## Validity and gaps

The `.valid.tif` sidecar records how each pixel was mapped:

- `255`: measured and verified in 3D through every transfer stage.
- `128`: filled across a gap.
- `0`: unmapped.

For strictly measured output, keep only `validity == 255` for each asset,
including supervision masks.

Optional seam filling continues the mapping into nearby gaps.
`--max-seam-distance` limits its reach in target-vertex units;
`--seam-anchor content` measures that distance from matched vertices with
nonzero annotation. Filling cannot correct mismatched surfaces.

The optional planar mode fits one global 2D affine from geometry samples.
It provides a simpler approximation but cannot follow local surface changes.

See [Reference](reference.md) for options and output details.
