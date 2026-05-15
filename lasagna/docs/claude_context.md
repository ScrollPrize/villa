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
| `opt_loss_smooth.py` | Smoothness regularization (normalized by mesh_step²). |
| `opt_loss_bend.py` | Bend angle constraint — penalizes when adjacent edge angle exceeds 60° from flat. |
| `opt_loss_pred_dt.py` | Pred DT loss — two-regime clamped L1 on distance-to-surface channel. |
| `opt_loss_winding_density.py` | Winding spacing via grad_mag strip integration (Huber loss) + `ext_offset` loss for external surface offset. |
| `tifxyz_io.py` | Loads tifxyz directories → `(xyz, valid, meta)` tensors. Detects `-1,-1,-1` invalid sentinel. |
| `fit2tifxyz.py` | Exports fitted mesh as tifxyz surfaces (x.tif, y.tif, z.tif, d.tif, meta.json). |
| `fit_service.py` | HTTP REST API for VC3D integration (/optimize, /status, /stop, /export_vis). mDNS discovery via ~/.fit_services/*.json. |
| `fit.py` | Main fitting orchestrator — loads data/model/config, runs optimizer stages, manages losses. Supports windowed tifxyz optimization for offset mode. |

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

**Modes** (LasagnaMode enum in `SegmentationLasagnaPanel.hpp`):

| Mode | Enum | What it does |
|------|------|-------------|
| Re-optimize | 0 | Reload existing model.pt, run optimization stages |
| New Model | 1 | Create fresh model from seed point + arc/straight params |
| Expand | 2 | Grow existing model in selected directions |
| Offset | 3 | Create a parallel surface at a configurable offset from an existing tifxyz |

### Standard flow (modes 0-2)

1. VC3D creates seed surface (manual tracing or growth)
2. User places correction points (shift-click) with winding annotations
3. VC3D sends optimization request to lasagna HTTP service
4. Lasagna optimizes (stage-based Adam, streams progress)
5. VC3D downloads and imports results

### Offset mode (mode 3) — External Offset Surfaces

Generates a new surface at a configurable grad_mag integral offset from an existing tifxyz reference. Used to trace adjacent papyrus windings from a known surface.

**Data flow:**
1. VC3D sends the existing tifxyz as base64 (`tifxyz` in request body) — no model.pt needed
2. `fit_service.py` decodes files to temp dir, creates config with:
   - `tifxyz-init`: initialize model from the tifxyz (via `Model3D.from_tifxyz()`)
   - `external_surfaces`: same tifxyz registered as frozen reference with target offset
3. `fit.py` creates model (depth=1) initialized from the tifxyz vertices, then adds the frozen reference surface
4. Optimization: `ext_offset` loss (in `opt_loss_winding_density.py`) drives the model surface toward the target offset from the reference, using grad_mag integration along ray-bilinear-patch intersections (`model.py:_intersect_ext_surfaces()`)
5. Result exported as tifxyz back to VC3D

**Config keys** (injected by VC3D in offset mode):
- `offset_value` (float): target offset in winding-integral space (0 = reoptimize in place, ±1 = adjacent winding)
- `args.windings`: forced to 1

**Invalid vertices**: tifxyz surfaces use `(-1,-1,-1)` as invalid sentinel. `tifxyz_io.load_tifxyz()` returns a validity mask. Invalid vertices are inpainted via masked scale-space pyramid reconstruction in `Model3D.from_tifxyz_crop()`. The external surface validity mask is checked during ray intersection so `ext_offset` loss skips rays hitting invalid regions.

### Windowed optimization (offset mode)

Large tifxyz surfaces are too big to optimize as a single model. Windowed mode splits the surface into overlapping rectangular tiles, optimizes each independently, and returns multiple output tifxyz directories.

**Config** (set in VC3D Offset Settings panel, injected into `args`):
- `window-size` (int): window size in fullres voxels. 0 or omitted = no windowing (process whole surface).
- `window-overlap` (int): overlap between windows in fullres voxels. Default 500.

**Implementation** (`fit.py`):
1. Load full tifxyz to CPU (keeps GPU memory free)
2. Derive mesh_step from meta.json scale: `mesh_step = round(1/scale[0])`
3. Compute window grid via `_compute_window_grid()`:
   - `win_verts = window_size // mesh_step + 1`
   - `overlap_verts = overlap // mesh_step`
   - `stride = win_verts - overlap_verts`
   - Tiles H and W dimensions; last windows clamp to grid edges
4. Per window:
   - Crop `xyz[h0:h1, w0:w1]` and validity to GPU
   - `Model3D.from_tifxyz_crop()` creates model from cropped tensors
   - External surface cropped with margin (`max(4, 2*|offset|/mesh_step + 2)` extra verts each side) for correct ray intersection at boundaries
   - Auto-crop volume data for this window's spatial extent
   - Run all optimization stages
   - Export to `window_NNNN.tifxyz` with window metadata in meta.json
   - Free GPU memory (`del mdl, data; torch.cuda.empty_cache()`)

**Output**: each window is an independent tifxyz directory. `meta.json` includes:
- `window_index`: 0-based window number
- `window_origin_verts`: `[h0, w0]` position in the original grid
- `window_size_verts`: `[h, w]` of this window
- `source_grid_size_verts`: `[H, W]` of the original full surface
- `overlap_verts`: overlap in vertex units

**Service integration**: `fit_service.py` detects windowed output (`.tifxyz` dirs already present) and skips the standalone `fit2tifxyz` export step. The existing tar.gz packaging naturally includes all window directories.

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
