# 3D Lasagna Model — Status

Scroll surface fitting for the Vesuvius Challenge.
The 3D lasagna model fits a multi-scale mesh pyramid to preprocessed volumetric data,
recovering papyrus sheet geometry from cosine, gradient-magnitude, and direction channels
produced by a 2D UNet run along each axis (z, y, x slices).

## Architecture

```
fit.py  (CLI entrypoint)
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

| Component | File | What it does |
|-----------|------|--------------|
| Model | `model.py` | `Model3D` — arc parameterisation, 5-level 3D residual pyramid, amp/bias modulation, HR bilinear upsample, `bake_arc_into_mesh()` |
| Data pipeline | `fit_data.py` | `FitData3D` — loads preprocessed OME-Zarr (cos, grad_mag, dir0/dir1 per axis, valid, pred_dt); `grid_sample_fullres` for arbitrary 3D sampling |
| Direction loss | `opt_loss_dir.py` | Quad-face normals projected onto each axis plane, MSE against double-angle dir channels, weighted by surface-normal alignment |
| Step loss | `opt_loss_step.py` | Euclidean distance between adjacent height rows vs target `mesh_step` |
| Optimizer | `optimizer.py` | Stage list from JSON config; per-stage parameter groups, per-scale learning rates, automatic arc baking when arc params leave the optimised set |
| Export | `fit2tifxyz.py` | Writes per-winding x/y/z tiffs + `meta.json` with bbox, scale, uuid |
| HTTP service | `fit_service.py` | REST API for VC3D integration — start/stop jobs, stream progress, download results tar.gz, mDNS discovery |
| CLI / config | `cli_*.py`, `cli_json.py` | Argument parsing, JSON config merge, multi-stage config loading |

## Not yet implemented

Items described in the spec (`lasagna/lasagna_3d.md`) or the 2D model (`lasagna/old_2d/`) but not yet ported to 3D:

| Feature | Notes |
|---------|-------|
| `gradmag` loss | Sheet-density period-sum loss (exists in 2D, not yet in 3D) |
| `data` loss | Cosine MSE with amplitude/bias modulation (exists in 2D) |
| `data_plain` loss | Unmodulated cosine loss (exists in 2D) |
| `pred_dt` loss | Distance-transform supervision (exists in 2D) |
| Connection point computation | `conn_offset_ms` pyramid is allocated but connection vectors between windings in the ±D direction are not computed or used in any loss |
| Mesh growing | Spec describes grow operations in 6 directions (±D, ±H, ±W); not implemented |
| Mask scheduling | EMA / velocity-based mask expansion described in `lasagna/docs/mask_schedule.md`; not implemented |

## Recent changes

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
| `lasagna/model.py` | `Model3D`, pyramid ops, arc bake |
| `lasagna/fit.py` | Main CLI entrypoint |
| `lasagna/fit_data.py` | `FitData3D`, zarr loading, sampling |
| `lasagna/optimizer.py` | Stage-based optimisation loop |
| `lasagna/opt_loss_dir.py` | Direction alignment loss |
| `lasagna/opt_loss_step.py` | Step distance loss |
| `lasagna/fit2tifxyz.py` | Tifxyz export |
| `lasagna/fit_service.py` | HTTP service for VC3D |
| `lasagna/configs/` | JSON optimisation configs |
| `lasagna/docs/` | Design docs (losses, model, mask schedule) |
| `lasagna/old_2d/` | Legacy 2D model (reference for missing losses) |
