# Theory

How the transfer establishes correspondence, what it refuses to guess, and
why the annotation-canvas offset needs its own measurement.

## Geometry, not image registration

Two flattenings of the same papyrus sheet are two different 2D
parameterizations of one 3D surface. The transfer therefore never warps a
label by matching image content: label correspondence is determined
entirely from TIFXYZ geometry. Each TIFXYZ stores, for every canvas pixel,
its XYZ coordinate in the scan volume, and those coordinates are the
correspondence. Phase correlation on CT renders appears only in the QA
tools — to audit a constant annotation-canvas offset and to measure
visualization residuals — and never estimates the 3D volume registration
or replaces the surface mapping.

Labels are also not voxelized into a 3D raster and rendered back to 2D.
The 3D surfaces are used once to construct a target-pixel → source-UV
warp; the categorical source image is then sampled directly in 2D with
nearest-neighbour interpolation. Output smoothness is therefore determined
by the requested destination canvas size, and label values are never
blended.

## What is inferred, and what cannot be

The source label shape and source TIFXYZ shape identify the effective
source render sampling:

```text
render_scale_y = label_height * source_meta_scale_y / source_stored_height
render_scale_x = label_width  * source_meta_scale_x / source_stored_width
```

Applying that sampling to the target TIFXYZ metadata determines the output
shape, which handles stored-resolution labels, full-resolution labels, and
other uniformly scaled full canvases without the original
`vc_render_tifxyz` command. The inference assumes the label covers the
complete, unrotated, unflipped, uncropped canvas. A crop, rotation, or
flip is unrecoverable from image dimensions alone and must be undone or
supplied explicitly.

## Per-pixel correspondence

For every output pixel, the target TIFXYZ provides an XYZ coordinate
(optionally passed through a rigid volume affine when the two surfaces
live in different scan frames). The nearest source-surface vertices
propose candidate triangles; the best triangle yields barycentric UV
coordinates on the source canvas. Interpolated correspondences are then
re-checked in 3D at every output pixel against a distance threshold, so
neighboring stored-grid vertices that land on different windings cannot
fabricate a valid band through the intervening source UV canvas — those
pixels are rejected and marked invalid instead. Before full-resolution
rasters are allocated, sampled vertices are checked against the exact
source triangles, and the run aborts below 1% coverage: that normally
means the volume frames differ or the affine/direction is wrong.

The pipeline intentionally never estimates registration affines. When a
registration JSON is supplied, `forward` means
`p_target = M * p_source`; by default both directions are scored via
point-to-surface distances and the unambiguous winner is chosen.

## The label canvas offset

Stage one assumes the label raster sits pixel-for-pixel on the old TIFXYZ
canvas. Annotations, however, are drawn on rendered surface volumes, and a
published render does not always correspond exactly to the TIFXYZ the
transfer reads: `vc_render_tifxyz` accepts `--crop-x/--crop-y` and
`--auto-crop` options that shift the render's canvas origin, and either
side may have been re-exported from a different generation of the
flattening. TIFXYZ metadata cannot detect this — `meta.json` carries only
`scale` and a 3D `bbox`, no canvas origin. The result is a constant
canvas-pixel offset on every transferred label.

Surface geometry cannot reveal the offset: the 3D surfaces still coincide
and point-to-surface distances stay tiny. Because the flattening's tangent
frame rotates across the sheet, no single 3D transform can repair it
either — the offset is constant in canvas space, so it must be measured
from image content and corrected in 2D. (On the PHerc 0139 demo cases the
same canvas shift appears in segments on opposite sides of the scroll,
which rules out a volume-frame translation.)

The estimator projects a source CT render onto the target canvas through
the TIFXYZ geometry, measures band-passed tile-wise phase correlation
against the native target render, robustly fits translation plus linear
spatial drift, and re-projects until the residual vanishes. Re-measuring
after every update means a wrong sign or scale cannot survive silently,
and once the remaining shift is known to be small the peak search is
restricted so it cannot lock onto the neighbouring-winding periodicity.
A fitted field whose corners drift apart by more than a threshold rejects
the constant-translation model outright.

The two corrections are distinguishable from the evidence: a canvas offset
measures the same shift with low scatter everywhere and in every segment,
while a genuine volume translation produces shifts that vary with the
local flattening orientation.

## Independent evidence, explicit approval

The blessed workflow measures the offset from two independent CT
comparisons when the selected pyramid level preserves Z plane indices:

1. the exact center layer from each surface Zarr, which is the TIFXYZ
   layer itself;
2. the shipped `*_max_FIRST_LAST.tif` annotation canvas against a target
   maximum composite with the same physical slab thickness.

The images are never blended before measurement: agreement between their
independently measured offsets is stronger evidence than a `center + max`
blend, because a blend can hide which input caused a bias. The consensus
is approved only when at least two comparisons converge, each supports a
constant-translation model, and their full-resolution offsets agree. The
approved value is then passed to the pipeline explicitly
(`--label-canvas-offset DY DX`, meaning label pixel `(i, j)` depicts
source canvas position `(i + dy, j + dx)`) — the correction is always
auditable, never applied silently.

Some private level-2 pyramids shorten 65 Z planes to 17 while retaining
early same-index planes; their level-2 midpoint is then not the level-0
surface midpoint. Evidence preparation detects any level whose Z count
differs from level 0 and omits the exact-center comparison rather than
producing misleading evidence.

## Validity semantics and seam filling

The `.valid.tif` sidecar records provenance per pixel: `255` for a
measured, 3D-verified correspondence in every stage, `128` for
seam-filled, `0` for unmapped. `--fill-seams` keeps the per-pixel
geometry mapping but closes its holes by continuing the UV field smoothly
from the nearest measured neighbours. The fill is not width-bounded —
every unmapped pixel whose continued field lands inside the label raster
is filled, including large regions — so filled pixels are always written
as `128`, the count is reported, and the `128` provenance survives later
pipeline stages. Mask `validity == 128` downstream to recover the strictly
measured transfer.

`--planar` is the opposite trade-off: one global 2D affine fitted from
geometrically mapped samples warps the whole raster at once, so nothing is
rejected and nothing punches holes, but every non-affine component of the
true mapping is ignored. The report's residual percentiles quantify
exactly how much; it lives in `planar.py`, separate from the production
mapper, so its weaker guarantees cannot be confused with it.
