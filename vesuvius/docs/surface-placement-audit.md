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
  byte ranges of its vertex block, with the same number of points drawn from
  each. The controls below measure a further 4 × 5000 points.
- `--half` (8), `--step` (0.5): the profile runs from `-half` to `+half`
  voxels along the normal. `--sigma` (1.0) smooths the CT first.
- `--min-contrast` (8 grey levels), `--min-prominence` (0.15 of the profile
  range): a profile with less contrast, or whose nearest maximum is a shoulder
  rather than a peak, counts as no peak.
- `--workers` (8): threads for volume reads. `--seed` fixes the sample.
- `--cache DIR`: keep every CT chunk read as a `.npy` under `DIR` (one
  subdirectory per volume source and level, plus its shape, chunking and
  dtype; a local file also keys on size and mtime). A rerun against the same
  remote volume, or the next surface on it, then reads from disk. A pass over
  the public Scroll 1 segments at 20000 samples fills roughly 20–40 GB.

## Outputs

`<name>.json` with the metrics below, `<name>.samples.csv` with one row per
sample (`z y x nz ny nx patch offset contrast valid`, so the offsets can be
loaded into a viewer at their positions), `aggregate.csv` with one row per
surface, and `<name>.png` with `--plot` (offset as colour in two projections).

| metric | meaning |
| --- | --- |
| `n_samples`, `n_inside_volume`, `n_valid` | points drawn, inside the volume with the full profile, with a located peak |
| `n_no_normal` | tifxyz input only: valid grid points with no computable normal (hole edges, grid border), which are never sampled. Vertex-level defects that sit there, such as the resampling spikes of villa PR #2067, are invisible to this audit; the count says how much of the grid it did not look at |
| `n_grid_spikes`, `grid_step_median`, `grid_step_max` | tifxyz input only, no CT read: adjacent valid grid points are joined when their distance is at most 4x the median step; the largest connected group is the surface, and any other group still tied to it by a long step is a spike (vertices thrown off the surface singly or as a run, the resampling defect of villa PR #2067, whose check inspired this one). Islands with no tie are left alone. Spike row/col pairs go to `<name>.grid_spikes.csv`. On segment 20231007101619 resampled on main it finds exactly the PR's 23 vertices and nothing on the fixed output |
| `n_grid_holes`, `grid_hole_points` | tifxyz input only, no CT read: interior holes of the grid (4-connected groups of invalid points that do not touch the grid border) and the points they cover. A flattening whose quality filter punched holes through the surface (villa PR #2128) passes a step check and exits 0; this counts it |
| `outside_volume_frac` | fraction of sampled points whose profile does not fit inside the volume. Above 0.1 the audit warns: a surface mostly outside the volume is in the wrong coordinate frame (missing downscale or physical scale, villa PRs #2112 and #2137), and its offsets mean nothing until the metadata is fixed |
| `no_peak_frac` | share of inside points with no qualifying peak (low contrast, air, or a shoulder) |
| `offset_median`, `offset_q25`, `offset_q75`, `offset_mean_abs` | signed offset from the surface point to the sheet peak, in voxels along the stored normal; positive means the peak lies in the normal direction |
| `offset_abs_gt_1_frac`, `_2_`, `_3_` | share of valid points more than 1, 2 or 3 voxels from the peak |
| `n_censored`, `censored_frac` | inside points with enough contrast whose profile rises to an end sample and has no accepted peak: the sheet may lie at or beyond the search limit (`half`) |
| `offset_at_limit_frac` | share of valid offsets at `half` exactly; with the default grid an end sample cannot be an accepted peak, so this stays 0 and `censored_frac` carries the signal |
| `off_sheet_frac` | share of valid points where the CT at the point is below the half-maximum of its peak, i.e. the point is outside the sheet |
| `sheet_fwhm_median` | half-maximum width of the located sheet, voxels |
| `contrast_median` | median profile range |
| `n_patches`, `patch_offset_*`, `patch_abs_gt_2_frac` | median offset per 32-voxel cell (30+ valid points): quantiles, range, and the share of cells more than 2 voxels off |
| `snap_remeasure_sd`, `snap_remeasure_abs_gt_2_frac` | control: each valid point moved to its located peak and measured again; the spread left is the precision of the peak location, not of the surface (0.5 voxels on Scroll 1) |
| `tangent_absdiff_median_{2,4,8}`, `tangent_corr_{2,4,8}` | a second point 2, 4 and 8 voxels away in the tangent plane: median absolute difference of the two offsets, and their correlation. Agreement at 2 and disagreement at 8 means the surface misses the sheet at a scale below its own triangles; agreement at all three means a displaced surface |

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

## Results on the public Scroll 1 segments

All 255 public segments with an obj (20230514182829 has only a points file)
against the public `54keV_7.91um_Scroll1A` volume, 20000 samples each,
defaults; 224 segments have 1000 or more valid samples. Quantiles across
segments (10th / median / 90th):

| | offset median | IQR | share > 2 vox | off sheet | no peak |
| --- | --- | --- | --- | --- | --- |
| across 255 segments | −1.5 / −0.5 / +1.5 | −2.5 … +1.5 (median) | 0.31 / 0.48 / 0.58 | 0.12 / 0.25 / 0.37 | 0.10 / 0.19 / 0.28 |

| control | 10th | median | 90th |
| --- | --- | --- | --- |
| `snap_remeasure_sd` | 0.25 | 0.34 | 0.42 |
| `tangent_absdiff_median_2` / `_4` / `_8` | 0.5 / 0.5 / 0.5 | 0.5 / 0.5 / 1.0 | 0.5 / 0.5 / 1.0 |
| `tangent_corr_2` / `_4` / `_8` | 0.79 / 0.62 / 0.41 | 0.85 / 0.72 / 0.54 | 0.89 / 0.79 / 0.66 |

The segments give the same picture, and the offset histogram is nearly
flat from −6 to +6, so this is not segments drifting the same way. What the
controls show: the snap control remeasures at a spread of 0.3 voxels, so the
peak location is repeatable once a peak is found; the offsets of two points 2
and 4 voxels apart agree (median difference 0.5, correlation 0.85 and 0.7), at
8 voxels they differ by 1 (correlation 0.5), and in a dense resample of one
crop they are uncorrelated by 16 voxels, so the offset varies along the
surface on a scale of a few voxels. On half the surface area the mesh is more
than 2 voxels (16 µm) from the sheet. What the controls do not show: that the
detected peak is the papyrus surface rather than another layer or an edge, or
what causes the offsets. One hypothesis is that the public meshes, with
roughly 18 voxels between vertices, cannot follow a sheet that wanders a few
voxels inside one triangle; this was not tested (for instance against a finer
mesh). Per-segment rows are in the evidence repository
(`data/scroll1_segment_placement.csv`). At 20000 samples over a segment of hundreds of cm² no 32-voxel
cell collects 30 points, so the `patch_*` metrics are NaN; they need a dense
sample (a small tifxyz patch, or `--samples` in the millions).
