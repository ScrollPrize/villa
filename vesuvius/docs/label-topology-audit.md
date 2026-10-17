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
`<crop>.delta.json` (`metric: [before, after]`) with `--after`, `<crop>.png` with `--plot`.

## Metrics

| metric | meaning |
| --- | --- |
| `n_components` | 26-connected foreground components |
| `n_tiny`, `tiny_voxel_frac` | components under 100 voxels (specks) and their share of foreground |
| `n_big_components` | components of at least 100 voxels; topology is computed for these only |
| `b1_total_big`, `n_big_with_b1` | Betti-1: holes/handles through a sheet (a flat sheet is 0, a sheet with a hole or a torus is 1) |
| `b2_total_big` | Betti-2: enclosed cavities (a hollow shell is 1) |
| `n_closed_big`, `n_closed_with_b1` | big components not touching the crop border or ignore voxels; only these have untruncated topology |
| `thick_*` | sheet thickness in voxels (`2*EDT-1` at medial ridge voxels); `frac_ge_7` is the share at 7 voxels or more |
| `n_fused_components`, `bridge_voxels` | components that split into several large pieces after `--erode` erosion; blind on 1-3 voxel sheets |
| `n_junction_clusters` | clusters (20+ voxels) where the structure tensor is not planar: merged sheets or bridges |
| `largest_component_frac` | share of foreground in the biggest component |

Betti numbers use the Euler characteristic (26-connected foreground, 6-connected
background). Thresholds (tiny 100, thick 7, piece 500, junction ratio 0.3) are
heuristics, not validated against annotated defects.
