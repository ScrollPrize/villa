# Fail-closed TIFXYZ/volume preflight

`vesuvius.surface_preflight` checks that a TIFXYZ surface is structurally
usable and, when provided, paired with the intended Zarr/OME-Zarr CT volume
before an expensive render, label transfer, or inference run starts.

Install the geometry I/O dependencies and run:

```bash
pip install "vesuvius[label-transfer]"
vesuvius.surface_preflight \
  --surface /path/to/segment.tifxyz \
  --volume /path/to/scroll.zarr \
  --output preflight.json
```

To validate a surface without opening a CT volume, omit `--volume`:

```bash
vesuvius.surface_preflight \
  --surface /path/to/segment.tifxyz \
  --output preflight.json
```

For an OME-Zarr with a nonstandard hierarchy, `--array-key` may explicitly
select its base-resolution array. Non-base arrays are rejected because TIFXYZ
coordinates are expressed in base-resolution voxel space. By default the
command selects the first dataset declared by OME-Zarr `multiscales`, then
falls back to array `0` or the only array in the group. A volume path that
selects a concrete array inside a `.zarr` group is resolved through that group
and must select its base dataset. Standalone single-array stores remain valid.

The command checks:

- required TIFXYZ files, metadata, coordinate shapes, and optional mask
  compatibility, including canonical multipage and integer-scaled masks;
- at least one valid vertex and connected quad;
- finite selected coordinates;
- consistency between metadata scale and adjacent grid spacing;
- consistency between metadata `bbox` and valid coordinate bounds when `bbox`
  is present;
- exact valid-coordinate bounds in the selected CT array, including an
  optional `--margin`;
- deterministic, evenly ranked surface samples for nonzero CT signal support.

The `tifxyz_scale_consistency` gate compares the median adjacent grid spacing
against the reciprocal metadata scale. The median is estimated from a
logarithmic histogram (64 bins per octave), so the report also records the
median bin bounds (`median_spacing_bounds_voxels`, `ratio_range`) and the gate
fails only when the whole bin falls outside the accepted ratio range; a grid
exactly at the tolerance boundary passes. Adjust the accepted range with
`--scale-tolerance` (default `2.0`, minimum `1.0`). When metadata contains a
`bbox`, the `tifxyz_bbox_consistency` gate checks that all valid coordinates
fit inside it. Adjust its coverage allowance with `--bbox-tolerance` (default
`0.01` voxels, minimum `0`).

The JSON report schema is version `2` and records every required gate and its
observed value. Exit code `0` means every gate passed; exit code `2` means at
least one gate failed or an input could not be read. The report is written
atomically, so downstream jobs can require a complete `PASS` report rather
than guessing from partial output.

Signal support defaults to at least 95% of 1,024 deterministic samples with
absolute CT value greater than zero. Use `--minimum-support-fraction`,
`--max-samples`, or `--support-threshold` when a volume has a documented
different fill-value convention. Threshold changes are recorded in the
report.

A signal-support failure means the selected volume does not contain enough
nonzero signal at the sampled surface coordinates. This can indicate a wrong
pairing, a cropped, sparse, or partially populated volume store, or genuinely
unsupported surface coverage; it is not by itself evidence that the surface
geometry is defective.

This is an input-pairing preflight, not a proof of surface correctness. It does
not replace geometric diagnostics such as self-intersection or local
orientation analysis.

## Depth orientation (advisory)

The report also carries `surface.depth_orientation`, a geometric hint for
whether `vc_render_tifxyz` needs `--flip-normals`.

The renderer stacks layers along the grid normal `dP/dU x dP/dV`.
`--flip-normals` makes the layer index grow toward the scroll centre, which
is the order of the published surface volumes and the training labels.
Which flag gives that order depends on the grid's handedness, and any
single mirror of the grid (VC3D's *Flip Surface* actions, for example)
reverses it. With the wrong order the model can return a plausible map at
chance level: in villa#1648, `ink_9um` on PHerc. 0139 w035 dropped from
AUC 0.95 to 0.51-0.56.

The hint needs no CT and no labels. Over a 4 mm stencil a wrapped scroll
sheet usually curves around the scroll axis, so its curvature vector points
inward. The report gives `inward_fraction`, the `|kappa . N|`-weighted share
of vertices whose grid normal points that way.

Stencils are left out when any vertex along them is invalid (including
vertices skipped by decimation), when any grid
edge is longer than 3 times the nominal spacing, or when a half-chord is
longer than 1.5 times the grid run. Single weights are capped at 3 times the
90th percentile.

`status` is one of:

- `determined`: `|2 * inward_fraction - 1|` is at least 0.5, and
  `vc_render_flip_normals` is `true` or `false`;
- `undetermined`: the vote is closer than that;
- `too_small`: fewer than 100 vertices with a clean stencil. The surface is
  shorter than about 8 mm, its grid spacing disagrees with the metadata
  scale (see `tifxyz_scale_consistency`), or the stencils cross holes and
  defects;
- `skipped`: the voxel size is unknown. Pass `--voxel-size-um`, or keep the
  published mesh name (`...-on-<volume>-9.362um.tifxyz`);
- `error`: the computation failed. This never changes the gates or the exit
  code.

Grids above 4 million vertices are decimated first, keeping the same
physical stencil length.

It is a heuristic, and the sheet's local curvature is not always the scroll's
curvature. Treat a `determined` result as a hint. When it disagrees with how
the segment was made, or when the result matters, run inference with
`--direction both`.

```text
depth orientation: the grid normal points outward (inward_fraction=0.13); render with --flip-normals so the layer index grows toward the scroll centre (assumes a wrapped scroll sheet, not a flat fragment)
```

Measured on the open-data catalogue (6 October 2026; details in villa#1648):

- **Scroll segments with a reference.** For 77 published scroll segments on
  8 scrolls, the published surface volume runs along `-N` (the
  `--flip-normals` order). The hint is `determined` on 55 of them and agrees
  on all 55. Because every one of these grids has the same handedness,
  "always pass `--flip-normals`" scores the same. What this shows is that no
  determined hint was wrong on these scroll meshes.
- **Mirrored grids.** For an undecimated grid, mirroring flips
  `inward_fraction` to `1 - inward_fraction` exactly.
- **Fragments.** On the PHerc. 0500P2 fragment the hint was `determined` on
  4 of 7 segments with a reference and disagreed with the published order on
  all 4; the other 3 were `too_small`. One of those, `0500P2-wrap11_0919`,
  shows why. It reaches `inward_fraction` 0.94, below the vertex floor. Its
  grid has the same handedness as the neighbouring wraps 10 and 12, but
  that part of the sheet curves the other way. Do not use the hint on
  fragments.
- **All scroll segments.** Of 238 published scroll segments, 147 are
  `determined`, all with the standard handedness; 81 are `undetermined`
  (mostly multi-wrap PHerc. Paris 4); 7 are `too_small`; 3 are `skipped`.

## Example: catching an empty `vc_obj2tifxyz` export

Round-tripping the public Scroll 1 segment
`https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s1/autogens/02201554/`
(196x225 grid, `scale = [0.05, 0.05]`) through `vc_tifxyz2obj` and then
`vc_obj2tifxyz --uv-non-metric` (VC3D build `f07d33b`) produced a 2x2 grid with
no valid points, yet the converter exited `0`:

```text
Final grid: 2 x 2
Valid grid points: 0 / 4 (0%)
Warning: no valid grid points were rasterized.
Saving to tifxyz format...
Successfully converted to tifxyz format
```

Before this change the preflight could not inspect that output without a CT
volume (`error: the following arguments are required: --volume`). Now:

```bash
vesuvius.surface_preflight --surface out_nonmetric --output preflight.json
```

```text
FAIL: preflight.json
valid_surface_vertices: surface has no valid vertices (all coordinates are sentinel/invalid); the producer emitted an empty grid
valid_surface_quads: surface has no connected valid quads
tifxyz_scale_consistency: grid has no positive columns spacing pairs to compare with meta.json scale
tifxyz_bbox_consistency: no valid vertices to compare against bbox
```

with exit code `2`. The same segment re-exported with
`--tifxyz-source=<segment>` (224x195 grid, `scale = [0.05, 0.05]`) passes: the
measured median spacing is `19.98` voxels per cell against the `20.0` implied
by the metadata.
