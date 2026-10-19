# Label topology audit

`vesuvius.label_topology_audit` measures the voxel-level topology of 3D surface
label volumes (0 = background, 1 = surface, 2 = ignore; values configurable with
`--surface-value` / `--ignore-value`). Use it on label crops, refined labels or
binarised predictions to inspect holes, cavities, fragments, possible fused
sheets and a thickness proxy before training on or trusting them. These are
inspection signals, not verified physical defects or correct-winding judgments.

Needs `connected-components-3d`, `scikit-image`, `scipy`, `tifffile` (the
`models` extra); `matplotlib` only for `--plot`.

```bash
# one crop, JSON report
vesuvius.label_topology_audit --label sample_00001.tif --out audit_out

# overlay PNG, fail (exit 2) if any hole/handle is found
vesuvius.label_topology_audit --label sample_00001.tif --image img.tif --plot \
  --out audit_out --fail-on b1_total_big=0

# before/after refinement
vesuvius.label_topology_audit --label before.tif --after refined.tif --out audit_out

# a folder with labels/*.tif (and optional images/*.tif)
vesuvius.label_topology_audit --batch DATA_DIR --out audit_out

# a window of a zarr prediction mask (local path or URL), split into 256-voxel cubes
vesuvius.label_topology_audit \
  --label https://dl.ash2txt.org/community-uploads/yupeng/herc_PHerc1447_th0.3.zarr \
  --threshold 1 --roi 12288:13056,3840:4608,3584:4352 --tile 256 --out audit_out
```

## Input formats

`--label`, `--after` and `--image` accept a `.tif` file or a zarr array or
OME-Zarr group, given as a local path or an `http(s)://` URL (a name ending in
`.zarr`, a URL, or a directory holding `.zarray`, `.zattrs` or `zarr.json`).
For a group, `--level` picks the multiscale level (default `0`, the full
resolution; the pyramid levels of a binary mask are max-pooled and not
meaningful for topology). Zarr input is read lazily, so:

- `--roi Z0:Z1,Y0:Y1,X0:X1` reads only that window (empty bounds mean the whole
  axis; bounds are clipped to the array). Without it the whole array is read,
  which is only sensible for crop-sized arrays.
- `--tile N` splits the window into N-voxel cubes and audits each as its own
  crop named `<name>_z<z0>_y<y0>_x<x0>`, with `aggregate.csv` and `ranked.csv`
  as for `--batch`. Partial cubes at the far edge are dropped, so choose the
  window as a multiple of N. Topology is measured per cube: a sheet cut by a
  cube face is open at that face, and a hole that survives the cut is located
  against the cube face (`b1_face`), the same way crop faces do.
- `--threshold T` binarises the input as `value >= T`, for probability volumes
  (`128` for a uint8 probability at 0.5) and for `0/255` masks (`1`). It sets
  the surface value to 1 and disables the ignore value.
- `--threshold T1,T2,...` audits every cut of each window from a single read,
  as crops named `<name>_t<T>` (so `pred_z0_y0_x0_t128`), and writes
  `ranked.csv`. Use it to pick a binarisation threshold for a probability
  volume: plot `b1_total_big`, `b2_total_big` and `fused_voxel_frac` against `T`
  per tile; a cut that is too high fragments the sheets, one that is too low
  closes cavities between them.

`--batch` stays a folder of `.tif` crops.

Outputs: `<crop>.json` (full metrics), `aggregate.csv` (one row per crop),
`ranked.csv` with `--batch` (crops with topology/fusion/junction findings, worst first, with a one-line
reason), `<crop>.delta.json` (`metric: [before, after]`) with `--after`,
`<crop>.png` and `<crop>.holes.png` (three orthogonal slices per located hole,
label green, ignore blue, hole red) with `--plot`, `<crop>.hole_mask.tif` with
`--hole-masks`.

`ranked.csv` sorts on located interior plus unlocated holes, then cavities,
possible fused components, junction clusters, and finally located crop-face holes.
Unlocated holes may be near a crop face; their location is unknown. Tiny-fragment
and thickness findings appear in JSON/aggregate CSV but do not determine ranking.

Labels must be nonempty 3D arrays; singleton axes are supported. Images and
before/after labels must have matching shapes and refer to the same voxel grid
(the shape check cannot verify registration). `--erode` and `--min-piece` must
be positive, surface and ignore values must differ, and `--fail-on` limits must
be finite. A batch may leave reports for earlier valid crops if a later crop fails.

## Workflow: from a dataset to fixed crops

`ranked.csv` works as a proofreading queue: look at the top rows first, and fix
or drop those crops only after inspecting the supporting label and CT evidence.

```bash
# 1. audit the dataset, write hole tiles and hole masks
vesuvius.label_topology_audit --batch DATA_DIR --plot --hole-masks --out audit_out

# 2. open audit_out/ranked.csv; look at <crop>.holes.png for the top crops

# 3. fill the located holes with scripts/fix_holes.py, which expects
#    IN/<crop>/<crop>_surf.tif next to IN/<crop>/hole_mask.tif
mkdir -p fix_in/sample_00340
cp DATA_DIR/labels/sample_00340.tif fix_in/sample_00340/sample_00340_surf.tif
cp audit_out/sample_00340.hole_mask.tif fix_in/sample_00340/hole_mask.tif
# (run from vesuvius/; add --fg-value if your surface value is not 1)
python src/vesuvius/scripts/fix_holes.py --in-dir fix_in --ignore-value 2 --method 2d_hole_fill

# 4. compare topology counts, then inspect added voxels against CT;
#    this report alone does not establish that a repair is physically correct
vesuvius.label_topology_audit --label DATA_DIR/labels/sample_00340.tif \
  --after fix_in/sample_00340/sample_00340_surf_2d_hole_fill.tif --out audit_out
```

## Metrics

| metric | meaning |
| --- | --- |
| `n_components` | 26-connected foreground components |
| `n_tiny`, `tiny_voxel_frac` | components under 100 voxels (specks) and their share of foreground |
| `n_big_components` | components of at least 100 voxels; topology is computed for these only |
| `b1_total_big`, `n_big_with_b1` | Betti-1: holes/handles through a sheet (a flat sheet is 0, a sheet with a hole or a torus is 1) |
| `n_holes_located`, `holes` | holes found by location: background blobs that a 3-step closing fills and whose filling lowers b1 (centre, voxels, interior or crop face); wider holes are not located |
| `b1_face`, `b1_interior`, `b1_unlocated` | `b1_total_big` split by location: `b1_face` is located holes centred within 6 voxels of a crop face, `b1_interior` is the rest, including `b1_unlocated` (holes too wide to locate) |
| `b2_total_big` | Betti-2: enclosed cavities (a hollow shell is 1) |
| `n_closed_big`, `n_closed_with_b1` | big components not touching the crop border or ignore voxels; this does not establish physical correctness |
| `thick_*` | thickness proxy in voxel units (`2*EDT-1` at medial ridge voxels, with background padding); truncated at crop faces and not calibrated physical thickness; `frac_ge_7` is the share at 7 voxels or more |
| `n_fused_components`, `bridge_voxels` | components that split into several large pieces after `--erode` erosion; blind on 1-3 voxel sheets |
| `n_junction_clusters` | clusters (20+ voxels) where the structure tensor is not planar: merged sheets or bridges |
| `largest_component_frac` | share of foreground in the biggest component |

Betti numbers use the Euler characteristic (26-connected foreground, 6-connected
background), via `vesuvius.image_proc.topology.component_betti`, which the CPU
path of `scripts/detect_holes.py` also uses. Thresholds (tiny 100, thick 7,
piece 500, junction ratio 0.3) are heuristics.

## Results on the Kaggle surface labels

All 892 crops of `surfaces/kaggle/labels` (840 at 320³, 51 at 256³, one at 384³),
4 processes, 44 minutes:

| finding | crops | total |
| --- | --- | --- |
| holes/handles (`b1_total_big` > 0) | 152 (17%) | 571 |
| located, interior | 74 | 237 |
| located, on a crop face (`b1_face`) | 71 | 257 |
| not located (`b1_unlocated`) | 49 | 77 |
| interior (`b1_interior` > 0: located interior + not located) | 107 (12%) | 314 |
| cavities (`b2_total_big` > 0) | 69 (8%) | 206 |
| junction clusters | 281 (32%) | 651 |
| fused components | 22 (2%) | |
| tiny fragments, thickness ≥ 7 | ~0 | |

Holes cluster in the later crops: 1% of crops 0-99 have one, 49% of crops 800-899
and 81% of crops 900+. None of the holes are caused by ignore voxels. Filling
holes with only the ignore voxels inside them removes none of the 571. Filling
them with background voxels removes 495. Of the 571 holes, 494 (87%) were
located; the other 77 are too wide for a 3-step closing. Each located hole is
measured with the earlier holes already filled, so one loop is never counted
twice.

Provisional AI spot check: 29 located holes from 20 flagged crops, plus 10 clean
crops as controls. Each was shown as three orthogonal CT slices with the label overlaid.
- 12 holes were judged apparent label defects: a gap inside continuous papyrus, or a
  pocket inside two merged sheets labelled as one slab.
- 0 were judged to match a real tear in the papyrus.
- 17 could not be judged. Most sat on a crop face (20 of the 29 holes are within
  5 voxels of one), which is why `ranked.csv` puts interior holes first.
- All 10 clean controls were judged to follow the sheet.

The judging was done by an AI at thumbnail resolution, not by an expert
annotator. These counts do not establish precision or correct winding identity.

### Before/after with `fix_holes.py`

The 10 crops at the top of `ranked.csv`, with the audit's hole masks passed to
`scripts/fix_holes.py`, then audited again with `--after`:

| method | b1 | b2 | voxels added |
| --- | --- | --- | --- |
| `2d_hole_fill` | 111 → 49 | 19 → 12 | 3,731 |
| `close`, kernel 2 | 111 → 73 | 19 → 12 | 1,997 |
| `close`, kernel 3 | 111 → 67 | 19 → 12 | 4,689 |

- `2d_hole_fill` cleared 3 crops completely (13, 17 and 9 holes to 0), and
  took 3 more down to 1 hole each.
- It left 3 crops unchanged (26, 9 and 7 holes). One had no located holes,
  because they were all too wide. In the other two, the hole is not enclosed
  within any z-slice, and `2d_hole_fill` only fills holes in z-slices.
- Closing made one crop worse (6 holes to 7 with kernel 2, and to 8 with
  kernel 3). This is why `--after` is worth running on any fix.
- `fix_holes.py` only adds voxels. A pocket between two merged sheets needs a
  cut, which no tool here makes, and these were the most common defect in the
  spot check.

A lower b1 does not mean the repair is right. We rendered the voxels each method
added on the CT, for the four largest located holes per crop (20 tiles per
method), and judged each tile. The judge was an AI reviewer, not an expert, and
the tiles are small, so treat this as a warning sign rather than a measurement:

| method | fill on papyrus | fill on background or between sheets | no change | unclear |
| --- | --- | --- | --- | --- |
| `2d_hole_fill` | 5 | 3 | 5 | 7 |
| `close`, kernel 2 | 1 | 4 | 8 | 7 |
| `close`, kernel 3 | 1 | 6 | 8 | 5 |

Provisional inspection suggests some closing additions fill an inter-sheet wedge
and join sheets. Run
`--after` and look at the added voxels on the CT before keeping any automatic
fill. Most holes that were left unchanged looked like pockets between merged
sheets.

## Related tools

- `scripts/detect_holes.py` computes the same per-component Betti numbers and
  localizes holes with persistent homology. It writes hole masks.
- `scripts/hole_viewer.py` and `scripts/fix_holes.py` are for inspecting and
  repairing individual crops.
- This audit is the dataset-level pass that runs before them:
  - it adds cavities, fusion, junction and thickness checks
  - it separates holes on crop faces from interior ones
  - it adds before/after deltas and `--fail-on` for CI
  - it ranks a whole folder, so the per-crop tools go to the worst crops first
