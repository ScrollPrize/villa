# Label TIFF → lasagna fit pipeline

## Input

A label TIFF (ZYX, uint8): 0=background, 1=prediction, 2=ignore.

## Step 1 — Preprocessing (labels → lasagna normals zarr)

```bash
python lasagna/labels_to_lasagna_normals.py \
    --input labels.tif \
    --work-dir work \
    --output normals.zarr
```

This runs five substeps internally:

1. **Read TIFF, write binary zarr** — extracts the `==1` mask into a zarr Group with dataset `"0"`, consumed by vc_gen_normalgrids.
2. **vc_gen_normalgrids** — generates normal grid volumes from the binary mask.
3. **vc_ngrids --fit-normals** — fits local 3D normals, writes ngrids zarr with `x/0`, `y/0`, `z/0` (hemisphere-encoded uint8 normals).
4. **Compute pred_dt** — euclidean distance transform of inverted binary mask at full resolution (distance from each voxel to nearest foreground surface; 0 on surface, increasing away), mean-pooled to step resolution, raw distance clamped to 255 uint8.
5. **Assemble lasagna zarr** — Python reads ngrids `x/0`, `y/0` + binary prediction + pred_dt, writes a flat `zarr.Array` (5, Z, Y, X) uint8:

| Channel | Name     | Value                                              |
|---------|----------|----------------------------------------------------|
| 0       | cos      | 255 where binary pred at step resolution, 0 elsewhere |
| 1       | grad_mag | density (default 128) where pred, 0 elsewhere      |
| 2       | nx       | ngrids `x/0` (hemisphere-encoded)                  |
| 3       | ny       | ngrids `y/0` (hemisphere-encoded)                  |
| 4       | pred_dt  | distance to nearest foreground surface in voxels, clamped to 255 |

Optional flags: `--step` (default 4), `--density` (default 128), `--skip-gen-normalgrids`, `--skip-fit-normals`, `--no-pred-dt`.

## Step 1b — Winding volume (optional)

```bash
python lasagna/labels_to_winding_volume.py \
    --input labels.tif \
    --output winding.zarr
```

Computes a per-voxel winding number volume from the label TIFF:

1. Extracts foreground (`==1`) and runs 3D connected components.
2. Computes pairwise average distances between CCs via distance transforms.
3. Orders CCs into a winding chain (greedy: start with most isolated, append nearest unused).
4. Interpolates per-voxel winding numbers from the two nearest CCs (inverse-distance weighting).
5. Downsamples via mean-pooling and writes a float32 zarr with attrs `scaledown`, `n_components`, `winding_map`.

The output zarr has smooth values ranging from 1 to N (number of CCs), with integer values at CC centers and fractional values in between.

Optional flags: `--step` (default 4), `--connectivity` (6 or 26), `--min-voxels` (default 0), `--chunk-size` (default 64).

## Step 2 — Fitting

```bash
python lasagna/fit.py \
    lasagna/vc3d_configs/vc3d_labels_3d_straight.json \
    --input normals.zarr \
    --seed <cx> <cy> <cz> \
    --model-w <width> --model-h <height> \
    --windings <n> \
    --out-dir work/fit_output \
    --model-output work/fit_output/model.pt
```

To use the winding volume constraint, add `--winding-volume winding.zarr` and set `"winding_vol"` weight in the stages config:

```bash
python lasagna/fit.py \
    lasagna/vc3d_configs/vc3d_labels_3d_straight.json \
    --input normals.zarr \
    --winding-volume winding.zarr \
    --seed <cx> <cy> <cz> \
    --model-w <width> --model-h <height> \
    --windings <n> \
    --out-dir work/fit_output \
    --model-output work/fit_output/model.pt
```

The `winding_vol` loss penalizes deviation of each mesh depth layer from its expected winding number (depth d → winding d+1). Set the weight in the stages JSON `base` or per-stage `w_fac`, e.g. `"winding_vol": 1.0`.

The straight config models the sheet as a line in XY (center + angle + half-width) with perpendicular winding offsets — appropriate for small or flat regions. It runs two stages:
1. **straight_only** (1000 steps) — fits straight parameters (cx, cy, angle, half_w).
2. **opt** (2000 steps) — optimizes mesh and connectivity offsets.

The straight representation is baked into the mesh on save, so downstream steps are identical regardless of init mode.

For scroll-like geometry with significant curvature, use `vc3d_labels_3d.json` instead (arc init).

## Step 3 — Visualization (model → OBJ for MeshLab)

```bash
python lasagna/export_vis_obj.py \
    --model work/fit_output/model.pt \
    --input normals.zarr \
    --output-dir work/vis
```

This writes to `work/vis/`:
- `mesh.obj` — fitted surface mesh
- `connections.obj` — inter-winding connection lines
- `slice_xy_cos.obj` + `.png`, `slice_xz_cos.obj` + `.png` — volume cross-sections
- `loss_normal.obj` + `.png`, `loss_step.obj` + `.png` — loss heatmaps on mesh

Open all `.obj` files as layers in MeshLab. Textured slices and loss maps use accompanying `.mtl`/`.png` files (loaded automatically).

Optional flags: `--slices` (default: xy xz yz), `--channels` (default: cos pred_dt), `--losses` (default: normal step), `--no-mesh`, `--no-connections`, `--device` (default: cpu).

## Arguments

- `--seed`: center of the region to fit, in full-resolution voxel coordinates.
- `--model-w`, `--model-h`: mesh dimensions in fullres voxels.
- `--windings`: number of sheet windings to model.
- `--device`: compute device (default cpu, use `cuda` for GPU).
- `--downscale`: should match `--step` from preprocessing (default 4).
