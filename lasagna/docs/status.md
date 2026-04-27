# 3D Lasagna Model — Status

Scroll surface fitting for the Vesuvius Challenge.
The 3D lasagna model fits a multi-scale mesh pyramid to preprocessed volumetric data,
recovering papyrus sheet geometry from cosine, gradient-magnitude, and direction channels
produced by a 2D UNet run along each axis (z, y, x slices).

## Architecture

```
preprocess_cos_omezarr.py  (preprocessing pipeline)
  ├─ infer mode    (2D UNet per-slice inference → per-axis zarr)
  ├─ integrate mode (3-axis fusion → single OME-Zarr)
  └─ convert_fit_zarr_to_vc3d_omezarr.py  (flat zarr → per-channel OME-Zarr pyramid)

fit.py  (fitting entrypoint)
  ├─ cli_data / cli_model / cli_opt / cli_json   (argument parsing + JSON config merge)
  ├─ fit_data.py   (loads OME-Zarr → FitData3D, grid_sample_fullres)
  ├─ model.py      (Model3D: arc param + 5-level residual pyramid + modulation)
  ├─ optimizer.py  (stage-based Adam, per-scale LRs, arc auto-bake)
  │     ├─ opt_loss_dir.py   (quad-face normal vs per-axis dir channels)
  │     └─ opt_loss_step.py  (mesh-row spacing deviation)
  ├─ fit2tifxyz.py (export mesh slices → x/y/z tiffs + meta.json)
  └─ fit_service.py (HTTP wrapper: /optimize, /status, /results, mDNS)
```

## Implemented components

### Preprocessing

| Component | File | What it does |
|-----------|------|--------------|
| UNet inference | `preprocess_cos_omezarr.py` (default mode) | Per-slice 2D UNet inference along a chosen axis (z/y/x); tiled with overlap; outputs 5-channel uint8 zarr (cos, grad_mag, dir0, dir1, valid) |
| 3-axis fusion | `preprocess_cos_omezarr.py integrate` | Reads z/y/x per-axis volumes; normal-weighted fusion of cos + grad_mag; passes through per-axis dir0/dir1; optional distance transform from pred mask |
| OME-Zarr conversion | `convert_fit_zarr_to_vc3d_omezarr.py` | Converts flat (C,Z,Y,X) zarr to per-channel OME-Zarr with multi-level pyramid |

Integrated output channel layout (uint8):

| Channel | Content |
|---------|---------|
| 0 | fused cos (normal-weighted across axes) |
| 1 | fused grad_mag |
| 2–4 | dir0_z, dir1_z, valid_z |
| 5–6 | dir0_y, dir1_y |
| 7–8 | dir0_x, dir1_x |
| 9+ | pred_dt (if enabled) |

### Fitting

| Component | File | What it does |
|-----------|------|--------------|
| Model | `model.py` | `Model3D` — arc parameterisation, 5-level 3D residual pyramid, amp/bias modulation, HR bilinear upsample, `bake_arc_into_mesh()`, tifxyz init with masked inpainting, external surface intersection |
| Data pipeline | `fit_data.py` | `FitData3D` — loads preprocessed OME-Zarr (cos, grad_mag, dir0/dir1 per axis, valid, pred_dt); `grid_sample_fullres` for arbitrary 3D sampling with optional gradients |
| Direction loss | `opt_loss_dir.py` | Quad-face normals projected onto each axis plane, MSE against double-angle dir channels, weighted by surface-normal alignment |
| Step loss | `opt_loss_step.py` | Euclidean distance between adjacent height rows vs target `mesh_step` |
| Smooth loss | `opt_loss_smooth.py` | Smoothness regularization, normalized by `mesh_step²` |
| Bend loss | `opt_loss_bend.py` | Bend angle constraint — penalizes when angle between adjacent edges exceeds 60° from flat |
| Pred DT loss | `opt_loss_pred_dt.py` | Two-regime clamped L1 loss on distance-to-surface channel (outside + inner weighting) |
| Winding density | `opt_loss_winding_density.py` | Winding spacing via grad_mag strip integration (Huber loss), plus `ext_offset` loss for external surface offset optimization |
| Optimizer | `optimizer.py` | Stage list from JSON config; per-stage parameter groups, per-scale learning rates, automatic arc baking when arc params leave the optimised set |
| Export | `fit2tifxyz.py` | Writes per-winding x/y/z tiffs + `meta.json` with bbox, scale, uuid |
| HTTP service | `fit_service.py` | REST API for VC3D integration — start/stop jobs, stream progress, download results tar.gz, mDNS discovery |
| Tifxyz I/O | `tifxyz_io.py` | Loads tifxyz directories → `(xyz, valid, meta)` tensors with `-1,-1,-1` invalid detection |
| CLI / config | `cli_*.py`, `cli_json.py` | Argument parsing, JSON config merge, multi-stage config loading |
| Windowed opt | `fit.py` | Windowed tifxyz optimization — tiles large surfaces into overlapping windows, optimizes each independently, exports per-window tifxyz |

## Not yet implemented

Items described in the spec (`lasagna/lasagna_3d.md`) or the 2D model (`lasagna/old_2d/`) but not yet ported to 3D:

| Feature | Notes |
|---------|-------|
| `gradmag` loss | Sheet-density period-sum loss (exists in 2D, not yet in 3D) |
| Mesh growing | Spec describes grow operations in 6 directions (±D, ±H, ±W); not implemented |
| Mask scheduling | EMA / velocity-based mask expansion described in `lasagna/docs/mask_schedule.md`; not implemented |

## Recent changes

**3D preprocessing pipeline** — `preprocess_cos_omezarr.py` now supports the full
3-axis preprocessing workflow for the 3D model:

- **UNet inference**: per-slice tiled inference along z, y, or x axis with configurable
  tile size, overlap, and Gaussian blur. GPU-accelerated with torch autocast.
- **3-axis integration**: fuses cos and grad_mag from z/y/x volumes using estimated
  surface normal weights. Per-axis dir channels are passed through. Optional distance
  transform from prediction mask (CuPy GPU or CPU EDT fallback).
- **Two execution paths**: tile-parallel (numba JIT, releases GIL for thread parallelism)
  and slab-based (pipelined read/compute/write). Both use chunk-aligned iteration to
  prevent concurrent write races on shared zarr chunks.
- **OME-Zarr conversion**: `convert_fit_zarr_to_vc3d_omezarr.py` converts the flat
  integrated zarr into per-channel OME-Zarr with multi-level pyramids for the fitting
  pipeline.

**Chunk-alignment fix** — tile and slab iteration now aligns to zarr chunk boundaries
(rounding down the crop start), preventing race conditions where adjacent tiles would
concurrently read-modify-write the same underlying zarr chunk.

**Arc-bake-on-save** — arc parameterisation (cx, cy, radius, angle0, angle1) is now
automatically absorbed into the mesh pyramid at two points:

1. **`optimizer.py`** — when a stage does not include any arc params in its optimised set,
   `bake_arc_into_mesh()` is called before that stage runs.
2. **`fit.py` `_save_model`** — every checkpoint save calls `bake_arc_into_mesh()` if
   `arc_enabled` is still true.

After baking, `arc_enabled = False` and the saved checkpoint contains only the
self-contained mesh pyramid — no dangling arc parameters.

## Key files

| Path | Purpose |
|------|---------|
| `lasagna/lasagna_3d.md` | Full 3D model specification |
| `lasagna/preprocess_cos_omezarr.py` | UNet inference + 3-axis fusion preprocessing |
| `lasagna/convert_fit_zarr_to_vc3d_omezarr.py` | Flat zarr → per-channel OME-Zarr pyramid |
| `lasagna/model.py` | `Model3D`, pyramid ops, arc bake |
| `lasagna/fit.py` | Main fitting entrypoint |
| `lasagna/fit_data.py` | `FitData3D`, zarr loading, sampling |
| `lasagna/optimizer.py` | Stage-based optimisation loop |
| `lasagna/opt_loss_dir.py` | Direction alignment loss |
| `lasagna/opt_loss_step.py` | Step distance loss |
| `lasagna/fit2tifxyz.py` | Tifxyz export |
| `lasagna/fit_service.py` | HTTP service for VC3D |
| `lasagna/configs/` | JSON optimisation configs |
| `lasagna/docs/` | Design docs (losses, model, mask schedule) |
| `lasagna/old_2d/` | Legacy 2D model (reference for missing losses) |
