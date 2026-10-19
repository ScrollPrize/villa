# Surface placement audit

`vesuvius.surface_placement_audit` measures where a traced surface sits
relative to the papyrus sheet in the CT. For a random sample of surface points
it reads the CT along the stored surface normal, finds the nearest density peak
(the sheet) and reports the signed distance from the surface point to that
peak. A surface that tracks one face of one sheet gives offsets that are small
and nearly constant; a surface that wiggles through the sheet, drifts across
fibre layers or leaves the sheet gives offsets that spread, change sign from
patch to patch, or run to the search limit.

The audit does not need ground truth and runs on CPU over HTTP: a remote
VolCart `.obj` is sampled by byte range (the vertex block only, no download),
and the CT is read chunk by chunk from a zarr.

Needs `scipy`, `tifffile` and `zarr` (the `models` extra); `matplotlib` only
for `--plot`.

```bash
# one public segment against the public Scroll 1 volume, 20k samples
vesuvius.surface_placement_audit \
  --surface https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20231007101619/20231007101619.obj \
  --volume https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr \
  --out placement_out --plot

# several surfaces, local obj or tifxyz directories, fail (exit 2) if more than
# half of the points sit more than 2 voxels from the sheet peak
vesuvius.surface_placement_audit --surface seg_a.obj seg_b/ --volume ct.zarr \
  --out placement_out --fail-on offset_abs_gt_2_frac=0.5
```

## Inputs

- `--surface`: one or more VolCart `.obj` files (path or `http(s)://` URL;
  `v`/`vn` pairs are required, so `<id>_points.obj` files without normals are
  not accepted) or tifxyz directories (`x.tif`, `y.tif`, `z.tif`, `meta.json`;
  normals come from `Tifxyz.compute_normals()`).
- `--volume`: the CT as a `.tif`, a zarr array or an OME-Zarr group (path or
  URL); `--level` picks the multiscale level. Surface coordinates must be in
  the voxel grid of that level.
- `--samples` (20000): points per surface. An obj is read in `--blobs` (40)
  byte ranges of its vertex block, each a spatially compact patch since VolCart
  writes vertices in grid order, with the same number of points drawn from each.
- `--half` (8), `--step` (0.5): the profile runs from `-half` to `+half`
  voxels along the normal. `--sigma` (1.0) smooths the CT first.
- `--min-contrast` (8 grey levels), `--min-prominence` (0.15 of the profile
  range): a profile with less contrast, or whose nearest maximum is a shoulder
  rather than a peak, counts as no peak.
- `--workers` (8): threads for volume reads. `--seed` fixes the sample.

## Outputs

`<name>.json` with the metrics below, `<name>.samples.csv` with one row per
sample (`z y x nz ny nx patch offset contrast valid`, so the offsets can be
loaded into a viewer at their positions), `aggregate.csv` with one row per
surface, and `<name>.png` with `--plot` (offset as colour in two projections).

| metric | meaning |
| --- | --- |
| `n_samples`, `n_inside_volume`, `n_valid` | points drawn, inside the volume with the full profile, with a located peak |
| `no_peak_frac` | share of inside points with no qualifying peak (low contrast, air, or a shoulder) |
| `offset_median`, `offset_q25`, `offset_q75`, `offset_mean_abs` | signed offset from the surface point to the sheet peak, in voxels along the stored normal; positive means the peak lies in the normal direction |
| `offset_abs_gt_1_frac`, `_2_`, `_3_` | share of valid points more than 1, 2 or 3 voxels from the peak |
| `offset_at_limit_frac` | share at the search limit (`half`); a sheet further away than that |
| `off_sheet_frac` | share of valid points where the CT at the point is below the half-maximum of its peak, i.e. the point is outside the sheet |
| `sheet_fwhm_median` | half-maximum width of the located sheet, voxels |
| `contrast_median` | median profile range |
| `n_patches`, `patch_offset_*`, `patch_abs_gt_2_frac` | median offset per sampling patch (30+ valid points): quantiles, range, and the share of patches more than 2 voxels off; a large range with a small per-patch spread is drift, not noise |

The sign convention follows the stored normal and is not flipped to the CT
gradient, so a consistently negative median on a segment whose normals point
to the recto means the trace sits on the verso side of the peak. Compare
surfaces with the same normal convention.

## What it is for

- Checking a traced segment or a label set before it is used for training:
  a large `offset_abs_gt_2_frac` or `off_sheet_frac` says the trace is not on
  the sheet it claims, whichever sheet that is.
- Comparing tracers, tracer settings, or a trace before and after refinement,
  with `--seed` fixed so the same points are measured.
- Ranking the public segments of a scroll by how well they sit on the sheet,
  with `aggregate.csv`.

It measures distance to the nearest density peak, not winding identity: a
surface that jumped cleanly onto the neighbouring sheet reads as well placed.
Use it with the topology audit, not instead of it.
