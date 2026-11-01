# snap_labels

Move a tifxyz surface label onto the papyrus recto face of the CT it lives in, and mark every vertex where it
could not.

```
vesuvius.snap_labels LABEL.tifxyz --volume SCAN.zarr --umbilicus UMB.json --out OUT
```

`--volume` is a local OME-Zarr path or an anonymous https URL, read chunk by chunk. The recto side is the side
toward the scroll axis: give `--umbilicus` (JSON points or `z, y, x` lines), or `--axis-point X Y Z --axis-dir X Y Z`
for a straight axis, or `--recto-sign 1|-1` when the side is known along the mesh normal.

## What it does

Per vertex, it reads a trilinear CT profile along the mesh normal (+-12 voxels, 0.25 voxel steps), smooths it
lightly and cuts it half way between its 10th and 90th percentiles. The runs above that level are papyrus bands; the
vertex moves to the recto face of the nearest band at sub-voxel precision. A vertex abstains when the band is
ambiguous (a second band nearly as near), runs off the profile, has too little contrast, is more than 4 voxels away
(`--max-move`), or the normal is nearly tangent to the radial direction. `--merged` also abstains where a band is
over 1.8 times the mesh's 25th percentile band thickness, which is how two sheets closer than about one voxel look.
An abstaining vertex may still move by an offset filled in from measured neighbors; nothing at that vertex vouches
for it, so it is flagged.

The optional --min-gap N rule abstains when either measured gap from the chosen band to a neighboring band is less
than N voxels. It marks the reason as "touching neighbor", leaves the vertex at its input coordinates, and excludes
its offset from filling and smoothing. The option is off by default. It changes coverage, and has not been shown to
increase the share of vertices within 1 voxel of a true face.

`--half 16 --fallback-half 12 --min-contrast 0` reads the same face on a +-16 voxel window, and again on +-12 where
+-16 abstains, with no contrast floor (`face.FACE_SNAP_WINDOW16`). The level comes from the window's 10th and 90th
percentiles, so where a vertex sits deep in a thick band a +-12 window can hold too little gap and the face lands
about a voxel inside the band; the wider window fixes most of those. These are the settings that were scored against
a second registered scan of the same object; the defaults are unchanged.

## Outputs

| File | Contents |
| --- | --- |
| `snapped.tifxyz/` | the moved label, same grid and voxel frame as the input; meta.json gets a new bbox and a `snap` record |
| `abstain.tif` | uint8: 0 measured its own face and moved onto it, 1 abstained, 255 not a valid vertex |
| `reason.tif` | uint8: why a vertex abstained (`face.FACE_REASONS`; 0 measured) |
| `confidence.tif` | float32, 0 to 1: contrast, separation from the next band and thickness, multiplied; 0 where it abstained. A ranking heuristic, not yet tested as one |
| `surface_label.zarr/` | OME-Zarr label on one image level (`--label-level`): 0 background, 1 surface (cells whose four corners measured their face), 2 ignore (cells with an abstaining corner, plus `--ignore-halo` voxels along the normal) |
| `review.json`, `review.tsv` | one VC3D point per region of connected abstaining vertices, largest first |
| `snap_labels.json` | inputs, settings, the snap report, counts per reason, the label summary, timings |

`--self-check` plants a smooth 2 voxel shift along the normals, snaps it back, and requires that a volume shifted by
3 voxels and an empty volume do not also "recover". On a mesh where most vertices abstain the check fails, since
abstaining vertices do not move back.

A snapped label is geometry, not a reading. Dependencies: numpy and zarr from the core install, and tifffile,
which `vesuvius.tifxyz` also needs (in the `render`, `models` and `all` extras). No scipy, no GPU.
