# Label topology audit

`vesuvius.label_topology_audit` measures the voxel-level topology of 3D surface
label volumes (0 = background, 1 = surface, 2 = ignore; values configurable with
`--surface-value` / `--ignore-value`). Use it on label crops, refined labels or
binarised predictions to see holes, cavities, fragments, fused sheets and
thickness before training on or trusting them.

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
```

Outputs: `<crop>.json` (full metrics), `aggregate.csv` (one row per crop),
`ranked.csv` with `--batch` (crops with any finding, worst first, with a one-line
reason), `<crop>.delta.json` (`metric: [before, after]`) with `--after`,
`<crop>.png` with `--plot`.

`ranked.csv` sorts on interior holes, then cavities, fused components, junction
clusters, and finally holes that touch a crop face.

## Metrics

| metric | meaning |
| --- | --- |
| `n_components` | 26-connected foreground components |
| `n_tiny`, `tiny_voxel_frac` | components under 100 voxels (specks) and their share of foreground |
| `n_big_components` | components of at least 100 voxels; topology is computed for these only |
| `b1_total_big`, `n_big_with_b1` | Betti-1: holes/handles through a sheet (a flat sheet is 0, a sheet with a hole or a torus is 1) |
| `b1_interior` | `b1_total_big` recomputed with 6 voxels trimmed off every crop face; trimming can split a sheet, so it is occasionally higher |
| `b2_total_big` | Betti-2: enclosed cavities (a hollow shell is 1) |
| `n_closed_big`, `n_closed_with_b1` | big components not touching the crop border or ignore voxels; only these have untruncated topology |
| `thick_*` | sheet thickness in voxels (`2*EDT-1` at medial ridge voxels); `frac_ge_7` is the share at 7 voxels or more |
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
| interior holes (`b1_interior` > 0) | 96 (11%) | 342 |
| cavities (`b2_total_big` > 0) | 69 (8%) | 206 |
| junction clusters | 281 (32%) | 651 |
| fused components | 22 (2%) | |
| tiny fragments, thickness ≥ 7 | ~0 | |

Holes cluster in the later crops: 1% of crops 0-99 have one, 49% of crops 800-899
and 81% of crops 900+. None of the holes are caused by ignore voxels. Filling
holes with only the ignore voxels inside them removes none of the 571. Filling
them with background voxels removes 495.

Spot check: 29 located holes from 20 flagged crops, plus 10 clean crops as
controls. Each was shown as three orthogonal CT slices with the label overlaid.
- 12 holes were clear label defects: a gap inside continuous papyrus, or a
  pocket inside two merged sheets labelled as one slab.
- 0 matched a real tear in the papyrus.
- 17 could not be judged. Most sat on a crop face (20 of the 29 holes are within
  5 voxels of one), which is why `ranked.csv` puts interior holes first.
- All 10 clean controls followed the sheet.

The judging was done at thumbnail resolution, so treat the counts as indicative.

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
