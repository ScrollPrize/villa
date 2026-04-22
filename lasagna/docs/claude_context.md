# Claude Context — Villa Codebase

## Overview

**Villa** is the Vesuvius Challenge monorepo for reading ancient Herculaneum scrolls from CT scans. The two core systems for surface extraction are **lasagna** (Python/PyTorch surface optimizer) and **volume-cartographer / VC3D** (C++/Qt segmentation GUI). They work together: VC3D provides interactive editing and correction point placement, lasagna runs gradient-based optimization, and results flow back to VC3D as tifxyz surfaces.

---

## Lasagna — 3D Surface Fitting Pipeline

PyTorch-based optimizer that fits multi-winding cylindrical meshes to preprocessed volumetric data.

### Key Modules

| File | Purpose |
|---|---|
| `preprocess_cos_omezarr.py` | UNet inference on CT volumes. Three modes: default (2D per-axis), `integrate` (fuse 3-axis results), `predict3d` (3D UNet single-pass). |
| `model.py` | `Model3D`: cylindrical mesh with arc parameterization, 5-level residual pyramid (x,y,z), amplitude/bias modulation. `bake_arc_into_mesh()` absorbs arc params. |
| `fit_data.py` | `FitData3D`: loads preprocessed OME-Zarr to GPU. `CorrPoints3D`: correction points as (x,y,z,winda) in fullres. Custom CUDA uint8 sampling kernel. |
| `optimizer.py` | Stage-based Adam optimization from JSON configs. Per-scale LRs, automatic arc baking, parameter groups (mesh_ms, amp, bias, arc_*). |
| `opt_loss_dir.py` | Quad-face normals vs per-axis direction channels. |
| `opt_loss_step.py` | Row-to-row spacing deviation from target mesh_step. |
| `opt_loss_corr.py` | Correction point loss — snap mode finds nearest quad, penalizes distance + winding mismatch. |
| `opt_loss_smooth.py` | Smoothness regularization. |
| `opt_loss_winding_density.py` | Winding spacing constraints. |
| `fit2tifxyz.py` | Exports fitted mesh as tifxyz surfaces (x.tif, y.tif, z.tif, d.tif, meta.json). |
| `fit_service.py` | HTTP REST API for VC3D integration (/optimize, /status, /stop, /export_vis). mDNS discovery via ~/.fit_services/*.json. |
| `fit.py` | Main fitting orchestrator — loads data/model/config, runs optimizer stages, manages losses. |

### Preprocessing Modes (`preprocess_cos_omezarr.py`)

Three CLI modes, all producing uint8 zarr with `preprocess_params` metadata for `fit_data.load_3d()`:

**Default (2D per-axis)** — Runs a 2D UNet on each slice along a chosen axis (z/y/x). Produces 5 channels: cos, grad_mag, dir0, dir1, valid. Must be run 3 times (once per axis) then fused with `integrate`.

```
python lasagna/preprocess_cos_omezarr.py \
    --input volume.zarr --output preprocessed_z.zarr \
    --unet-checkpoint model_2d.pt --axis z
```

**`integrate`** — Fuses three 2D per-axis volumes into a single 4-channel output (cos, grad_mag, nx, ny) via `_estimate_normal()` weighted normal estimation. Optionally adds a pred_dt channel.

```
python lasagna/preprocess_cos_omezarr.py integrate \
    --z-volume preprocessed_z.zarr --y-volume preprocessed_y.zarr \
    --x-volume preprocessed_x.zarr --output fused.zarr
```

**`predict3d`** — Single-pass 3D UNet inference. Predicts all 8 channels (cos, grad_mag, 3x2 dir encoding) at once, then converts to 4-channel output (cos, grad_mag, nx, ny) matching the `integrate` output format. Replaces the 3-axis + fusion pipeline. Uses CUDA by default when available.

```
python lasagna/preprocess_cos_omezarr.py predict3d \
    --input volume.zarr --output preprocessed.zarr \
    --unet-checkpoint model_3d.pt
```

Key `predict3d` options: `--tile-size` (tile cube size), `--overlap` (tile overlap), `--border` (hard-discard border), `--scaledown` (output downsample factor), `--crop-xyzwhd` (process a sub-region), `--pred-dt` (add distance-to-surface channel).

The 3D pipeline: tiled inference with 3D linear blending → sigmoid → avg_pool3d downsample → `_estimate_normal()` → uint8 encoding → zarr. The accumulator lives on CPU; for large volumes use `--crop-xyzwhd` to bound memory.

### Coordinate Spaces

- **Fullres**: raw voxel coordinates from the original volume
- **Model pixel space**: fullres / sd_fac (e.g. scaledown=4 → sd_fac=2^4=16 → 1 model pixel = 16 fullres voxels)
- **scaledown**: always an OME-Zarr pyramid level (power of 2). Actual factor = `2^scaledown` = `ChannelGroup.sd_fac`. This applies everywhere: CLI flags, `.lasagna.json`, zarr metadata.
- **z_step_vx**: z-stride between slices in **model pixels**, NOT fullres. Fullres z-stride = z_step_vx * sd_fac
- **3D model**: uniform sd_fac for ALL axes. 1 zarr voxel = sd_fac fullres voxels in x, y, z. z_step/z_step_eff are 2D-model-only concepts.

### Correction Points (`opt_loss_corr.py`)

Correction points are the user-in-the-loop mechanism. Each point has (x, y, z, winda) in fullres coordinates:

- **winda** = winding annotation = depth index from d.tif channel, assigned when point is placed in VC3D
- Points belong to collections; points in the same collection should land on the same depth layer
- Two modes: legacy and **snap mode** (current)
- Snap mode: brute-force init finds nearest quad for each point, then local updates track the closest quad as the mesh moves during optimization
- Loss penalizes: distance to surface + winding mismatch within collection

---

## Volume Cartographer (VC3D) — Segmentation GUI

C++/Qt GUI for interactive papyrus surface tracing and editing.

### Key Components

| File | Purpose |
|---|---|
| `CVolumeViewer.cpp/hpp` | Main 3D volume rendering widget. Handles mouse clicks, shift-click correction point placement, scene↔volume coordinate transforms. |
| `CWindow.cpp/hpp` | Main application window. |
| `SegmentationModule.cpp/hpp` | Core segmentation logic: growth, corrections, surface management. |
| `SegmentationWidget.cpp/hpp` | UI panel for segmentation controls. |
| `QuadSurface.hpp/cpp` | Quad-grid surface representation. Grid of 3D points stored as cv::Mat. Channels for mask, generations, d.tif. Lazy loading from tifxyz directories. |
| `SegmentationCorrections.cpp/hpp` | Manages correction point collections. |
| `SegmentationLasagnaPanel.cpp` | Integration with lasagna service. |
| `LasagnaServiceManager.cpp/hpp` | Manages Python service lifecycle, launches fit_service.py. |

### QuadSurface Coordinate Methods

Three coordinate systems:
1. **Nominal (volume) coordinates**: Physical 3D voxel space
2. **Internal relative (ptr) coordinates**: Surface-centered, _center is at (0,0)
3. **Internal absolute (_points) coordinates**: Grid indices, upper-left is (0,0)

Key methods:
- `loc_raw(ptr)` = `internal_loc(_center, ptr, _scale)` = `ptr + _center * _scale` → raw grid coords (col, row)
- `coord(ptr, offset)` = `internal_loc(offset + _center, ptr, _scale)` = `ptr + (offset + _center) * _scale` → 3D volume position
- `ptrToGrid(ptr)` → `(ptr.x / scale.x + center.x, ptr.y / scale.y + center.y)` — ptr-space to absolute grid
- `scale()` → `[sx, sy]` grid spacing in surface units
- `center()` → surface center in grid coordinates
- `lookupDepthIndex(surface, row, col)` → reads d.tif[row, col], returns NAN if invalid

### Tifxyz Format

Directory-based surface format:
```
winding_0000.tifxyz/
├── x.tif, y.tif, z.tif   # Float32 coordinates
├── meta.json              # scale, uuid, bbox
├── mask.tif               # Optional validity (uint8, 255=valid)
├── d.tif                  # Optional winding depth indices (float32)
└── generations.tif        # Optional growth tracking
```

Pixel (row, col) → 3D point from (x.tif, y.tif, z.tif). Points with z <= 0 are invalid (-1,-1,-1).

### Winding Concept

"Winding" = a layer of wrapped papyrus in the scroll. Mesh width (W) dimension = winding/circumferential direction. d.tif stores continuous depth index (float) per vertex — which winding layer a point belongs to. Used for correction point coupling: points in a collection should match depths.

---

## Lasagna ↔ VC3D Integration

```
VC3D → lasagna:  tifxyz seed + corrections (JSON with wind_a)
lasagna → VC3D:  optimized tifxyz (with updated d.tif channel)
```

1. VC3D creates seed surface (manual tracing or growth)
2. User places correction points (shift-click) with winding annotations
3. VC3D sends optimization request to lasagna HTTP service
4. Lasagna optimizes (stage-based Adam, streams progress)
5. VC3D downloads and imports results

---

## Recent Fix: wind_a Lookup via 2D Click Position

### Problem

When shift-clicking to place correction points in CVolumeViewer, wind_a (winding annotation from d.tif) was determined via `pointTo()` — a 3D nearest-surface search on the active segment. With a combined single-segment tifxyz containing multiple disconnected windings, `pointTo()` can't handle the disconnected geometry and returns wrong grid positions, giving wrong d.tif values.

### Root Cause

`CVolumeViewer.cpp` line ~675 used `seg.surface->pointTo(ptr, p, ...)` to reverse-map a 3D world position back to a grid position. This 3D nearest-neighbor search fails for combined tifxyz with multiple disconnected windings because it can find nearest points on the wrong winding.

### Fix

Replaced `pointTo`-based lookup with direct 2D computation from the scene click position. In segmentation view, the 2D pixel position directly maps to the grid position — no 3D reverse lookup needed.

```cpp
// Compute grid position directly from 2D scene coordinates
cv::Vec3f surf_loc = {static_cast<float>(scene_loc.x()/_scale),
                      static_cast<float>(scene_loc.y()/_scale), 0};
cv::Vec2f ss = seg.surface->scale();
cv::Vec3f fake_ptr(surf_loc[0] * ss[0], surf_loc[1] * ss[1], 0);
cv::Vec3f raw = seg.surface->loc_raw(fake_ptr);
int row = static_cast<int>(std::round(raw[1]));
int col = static_cast<int>(std::round(raw[0]));
float wind_a = lookupDepthIndex(seg.surface, row, col);
```

The math: setting `fake_ptr = surf_loc * scale` makes `loc_raw(fake_ptr) = fake_ptr + center * scale = surf_loc * scale + center * scale = (surf_loc + center) * scale` — equivalent to `coord(ptr=0, offset=surf_loc)`, giving correct grid coordinates.

Guard: only runs when `seg.viewerIsSegmentationView` (not slice views where there's no 2D surface context).

Reference: SegmentationModule.cpp already does this correctly at line ~1326 — uses grid position directly without pointTo.

### Cleanup

- Removed debug prints from CVolumeViewer.cpp
- Removed diagnostic `if dbg:` block from opt_loss_corr.py (lines ~634-661) that scanned all D layers comparing wind_d vs best_d distances
- Removed unused `#include <limits>` from CVolumeViewer.cpp
