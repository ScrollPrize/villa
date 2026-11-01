# Shared 3D canvas

`vc3d::scene3d::Scene3DCanvas` is a display-only Qt OpenGL 3.3 widget. It
supports CT planes, volumetric CT slabs, and a surface mesh without a fiber
scene or annotation controller. `fibers3d::Fiber3DCanvas` extends it with fiber
geometry, selection, ribbons, filtering, and annotation overlays. Existing
fiber panels use the same shared camera and CT/surface rendering implementation.

Controllers supply immutable snapshots using `setCTSlices()` and
`setSurfaceMesh()`. Sampling and I/O stay outside the canvas. The existing
`slices::sample()` and `surface::build()` helpers can prepare CT data and a
mesh from a tifxyz surface's coordinate matrix. Their implementations and
snapshot types remain in `fibers3d/` for source compatibility; the shared
canvas exposes `scene3d::slices` and `scene3d::surface` namespace aliases.
All geometry uses level-0 CT voxel coordinates. Call `fitAll()` after replacing
content when a new camera framing is desired. Empty slices and a null mesh
remove those layers independently.

The camera provides orbit, pan, zoom, orthographic/perspective projection,
`focusPoint()`, and `viewScaleChanged()`. Controllers can handle
`positionClicked()`, `emptyContextRequested()`, and `clearSelectionRequested()`.
A subclass can extend painting using `beginFrame()`, `drawSurface()`, and
`drawCTSlices()` (the fiber view inserts its geometry before translucent CT).
GL resources must be released with their context current, before shared context
cleanup; see the fiber subclass for context destruction and recreation handling.
`Scene3DCanvasState.hpp` is internal to the rendering implementations.

## Standalone viewer

Open **View → 3D Viewer** in Main. The dock can also float using **Detach**.
It displays the active CT volume and the surface selected in Main's surface
list, independently of fibers. **Fit all**, **Fit surface**, and **Perspective**
control the camera. **CT** toggles CT visibility; the axis checkboxes and opacity
control the individual planes. **Display surface mesh** and its opacity control
the selected mesh. **CT texture** is on by default; turn it off for the shaded
mesh. **CT texture OME-Zarr level** selects the source pyramid level and defaults
to **3**. Larger level numbers use coarser CT data; level 0 uses the finest data.
Only available levels are listed. If the saved level is absent, the viewer uses
the nearest coarser level, or the coarsest available level for a shorter pyramid.
The selector shows the effective level without overwriting your saved preference.
Changing it cancels the previous request and reloads the texture. Preferences are
separate for the general viewer and Fibers 3D; slice sampling is unchanged.

Textures sample the active CT on the tifxyz grid using the shared surface
interpolation and four-corner validity rule. They follow CT window/level, are
bounded to 2048 pixels per edge, and keep the shaded mesh where CT is invalid.
Geometry and source files are unchanged.

Under **Settings**, **Attach CT volume…** uses the existing volume attachment
workflow. **Import tifxyz surface…** opens the existing object-store import
dialog; select the imported surface in Main's surface list afterward. The viewer
has separate display preferences from Fibers 3D. Dock visibility, placement,
and floating state use Main's existing saved layout.

While a mesh is visible, only the **XY** slice is shown, centered on the mesh's
bounding box in X, Y and Z. Its width and height cover the mesh bounds projected
onto the plane, with a 5% margin, independently of the 1024-pixel texture limit.
**XY slice Z** moves that plane along the volume Z
axis through the fixed mesh. Drag the slider or enter a voxel position;
**Center on mesh** returns it to the mesh center. Hiding or removing the mesh
restores your slice axis choices and the **Follow Main** control. Neither the
mesh nor Main's CT focus moves. The control spans the
active volume, supports single-voxel slider steps and fractional typed positions,
and resets to Follow Main when the volume changes. For a tilted XY plane, Z
specifies its center; its orientation is preserved. CT remains visible alongside
the mesh.

Select **CT volume** in the display-mode menu for a translucent, rotatable
preview of the volume's interior, with no tifxyz mesh or slice planes. The GPU
ray marcher uses a small 3D texture directly; no contour mesh is constructed.
Zero-valued CT background is transparent. **Volume density** adjusts absorption
without reloading data. Window/level still follows the active CT display.

Preview memory is automatic. The loader prefers the requested available Zarr
level (default 3), then coarser levels. If even the coarsest level is too large,
it resamples that level into at most 256 voxels per axis. CPU and GPU texture
storage together are capped at 64 MiB, with at most 256 KiB of sampler scratch
allowance; the application's shared chunk cache is separate. Source coordinates
and native volume dimensions are preserved, so reducing resolution never shrinks
or stretches the displayed volume. The status reports the source Zarr level and
preview dimensions. The 3D memory budget defaults to **2048 MiB (2 GiB)**. Type a larger whole-MiB
value directly; there is no preset ceiling, only the process address-size limit.
Existing saved budgets are preserved. The same editable budget is used in
Fibers 3D. Increasing it allows larger surface/fiber loads; the CT overview
continues to allocate only its bounded preview.

This is a display overview: coarser/resampled data can blur or miss small features.
Source data and tifxyz geometry are unchanged. Orbiting, zooming, and changing
density reuse the resident texture. Regression tests cover volumes with no
pyramid, an oversized coarsest level, small budgets, native coordinates,
cancellation, empty data, uint16 normalization, and internal-voxel rendering.

**Volumetric** in the surface/slice mode uses the
shared Composite View thickness and material settings, so changing it also
changes the linked CT views. This is bounded slab rendering, not a whole-volume
streaming renderer. Hidden panels cancel pending sampling/meshing; reopening
reloads the current content.

Build and run from `volume-cartographer/` with existing dependencies:

```sh
AGENTS_AGENT_MODE=1 ninja -C build VC3D test_fiber_3d_canvas
build/bin/VC3D
build/bin/test_fiber_3d_canvas
```

The Qt tests require a display with OpenGL 3.3. They cover standalone slices,
volumetric slabs, surface visibility/removal, and existing fiber rendering and
asynchronous scene replacement. These rendering fixtures are regression checks,
not a substitute for visual acceptance on real scroll data.

The panel integration test exercises texture level selection (including coarse
source values, saved preferences, and single-level fallback), XY navigation,
volumetric mode, mesh loading, hide/show, and data removal. Like the canvas tests, it requires a display that
supports Qt OpenGL widgets (the Qt offscreen plugin is not supported):

```sh
build/bin/VC3D --test-scene-3d
```

For a real-data rendering check, provide a CT volume directory and a tifxyz
surface in its coordinate frame. This test uses temporary settings and reads
source data without modifying it. It also captures a CT volume-only image at
`<screenshot-path>.volume.png`:

```sh
VC3D_SCENE3D_VOLUME=/path/to/volume.zarr \
VC3D_SCENE3D_SURFACE=/path/to/surface.tifxyz \
VC3D_SCENE3D_SCREENSHOT=/tmp/scene3d-panel.png \
build/bin/VC3D --test-scene-3d
```

These test entry points require a build with `VC_TESTING` enabled.

The loader-only benchmark uses temporary settings, performs three loads with the
decoded cache invalidated before each load, and prints load time, source level,
preview dimensions, and CPU texture bytes. Source files remain read-only:

```sh
VC3D_SCENE3D_VOLUME=/path/to/volume.zarr \
VC3D_SCENE3D_BENCHMARK=1 VC3D_SCENE3D_BENCHMARK_MIB=128 \
/usr/bin/time -v build/bin/VC3D --test-scene-3d
```

`/usr/bin/time -v` reports peak RSS on Linux; use `/usr/bin/time -l` on macOS.
The shared chunk cache and process/library memory are included in process RSS,
but not in the preview texture byte count.

Measured on 2026-11-01 with GCC 13.2, Linux arm64, RelWithDebInfo, using
`/mnt/nas/volpkgs/PHerc0139_2um_ds2.volpkg/volumes/0139_ds2_raw.zarr` through the
read-only `/tmp/scene3d-scale-volume.zarr` wrapper (available levels 0, 1, 2, 4,
5; level 3 metadata was inaccessible). Both loaders read level 5. Three timed
loads per process, warm filesystem cache, decoded cache invalidated each time;
no rendering is included. The command above used `BENCHMARK_MIB=512` for this
comparison and `128` for the failure regression.

| Metric | Contour-based loader | Bounded preview |
| --- | ---: | ---: |
| Load time, mean | 738 ms | 201 ms |
| Load time, min / median / max | 721 / 734 / 759 ms | 182 / 193 / 228 ms |
| Peak process RSS | 555.3 MiB | 159.1 MiB |
| Temporary triangle buffers | 432 MiB | 0 |
| CPU texture | 41.93 MiB | 3.87 MiB |
| Display grid | 206 × 184 × 580 | 89 × 89 × 256 |
| Loading with a 128 MiB budget | Failed building contour | Passed all 3 loads |

Phase timers attributed 640 ms on average (87% of the old load time) to contour
construction. The new output deliberately uses a lower-resolution display grid;
this comparison includes that quality tradeoff. Native volume coordinates and
source voxel values on disk are unchanged.
