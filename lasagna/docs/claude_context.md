# Claude Context — Villa Codebase

## Overview

**Villa** is the Vesuvius Challenge monorepo for reading ancient Herculaneum scrolls from CT scans. The two core systems for surface extraction are **lasagna** (Python/PyTorch surface optimizer) and **volume-cartographer / VC3D** (C++/Qt segmentation GUI). They work together: VC3D provides interactive editing and correction point placement, lasagna runs gradient-based optimization, and results flow back to VC3D as tifxyz surfaces.

---

## Lasagna — 3D Surface Fitting Pipeline

PyTorch-based optimizer that fits multi-winding cylindrical meshes to preprocessed volumetric data.

### Key Modules

| File | Purpose |
|---|---|
| `preprocess_cos_omezarr.py` | Runs 2D UNet inference per axis, produces 5-channel uint8 zarr (cos, grad_mag, dir0, dir1, valid). `integrate` mode fuses 3-axis results. |
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

### Coordinate Spaces

- **Fullres**: raw voxel coordinates from the original volume
- **Model pixel space**: fullres / scaledown (e.g. scaledown=4 → 1 model pixel = 4 fullres voxels)
- **z_step_vx**: z-stride between slices in **model pixels**, NOT fullres. Fullres z-stride = z_step_vx * scaledown
- **3D model**: uniform scaledown for ALL axes. 1 zarr voxel = scaledown fullres voxels in x, y, z. z_step/z_step_eff are 2D-model-only concepts.

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
