# Fibers 3D

Fibers 3D shows fibers from the active `vc_object_store` attachment
beside VC3D's CT views. It is an inspection tool: it does not trace, optimize,
reconstruct or write back geometry.

## Main workspace

Open **View → Fibers → Fibers 3D**. The dock starts hidden and remembers its
layout. Resize it by dragging the dock separator; **Detach** floats or redocks
it. Drag to orbit, right/middle-drag to pan, and use the wheel to zoom.

Click a fiber to select it in the Fibers list. Selection
also follows the CT views. **Fit** frames the displayed geometry, including a
visible surface; **Fit selected** frames the selected fiber. Right-click a fiber
to center the CT views at that point or open it in Line Annotation. The 3D renderer is read-only; Line Annotation uses the store's editing permissions. Escape, an empty-space click, or **Clear selection** clears the
shared selection. Dragging does not deselect.

## Line Annotation

The **Fibers 3D** header button shows a separate dock containing the open fiber,
including its current generated geometry. A small cyan point marks the current
position. Clicking the curve moves Line Annotation to the picked position and
pauses automatic cut following. Navigating the flattened view updates the point
without rebuilding the 3D scene or moving its camera. The position readout uses
distance along the curve, not a percentage of point indices.

This dock keeps its own appearance preferences and does not load the whole store.
The current fiber retains its palette instead of the yellow selection highlight.
Its CT ribbon uses the actual generated Line Annotation surfaces, including
manual display-normal changes.

## Settings

**Settings** opens a movable, non-modal window. Changes apply immediately; the
3D view remains interactive. Close or Escape hides it.

| Tab | Controls |
| --- | --- |
| Fibers | Active fiber store, H/V/unclassified families, minimum length, whole volume or a box around CT focus |
| Surface | Selected surface mesh, opacity, nearby-fiber filter, Fit surface |
| Appearance | Colors, line width, perspective, cursor, region bounds, gap styles |
| Selection | Nearby fibers, other-fiber opacity, annotation markers/tags |
| Memory | Coordinate budget and maximum fibers per view |

The whole-volume filters and memory tab are absent from the single-fiber Line
Annotation dock. Controls specific to a color mode appear only in that mode.

### Colors and annotations

- **Custom:** one chosen color.
- **H / V:** the same default colors as Fibers CT overlays; unclassified is gray.
- **Direction:** XYZ orientation, or angle to one axis with a palette. Reversing
  a curve does not change its color.
- **Length:** complete source-fiber length in millimeters, with automatic or fixed
  range and a palette. Cropping the display does not change that length.

Palettes use VC3D/OpenCV's Viridis, Magma, Cividis and Turbo tables. Line width
defaults to **1 px**. Region bounds start off. **Other fibers opacity** is a
0–100% slider, applied only while a displayed fiber is selected.

Markers, tags and dashed gap styles start off. When enabled, existing annotation
metadata can mark control points, breaks, kollesis terminations, gaps, inferred
gaps and damaged spans. The renderer does not infer new tags. Geometry remains
unchanged when these styles are toggled.

### Proximity and surfaces

**Only fibers near selection** retains whole loaded fibers passing within the
chosen distance of any original segment of the selected fiber. Without a
selection it leaves all loaded fibers visible.

**Display surface mesh** shows the existing surface selected in Main in the same
CT coordinates. Valid grid cells become triangles; holes stay open. The mesh is
refreshed on surface changes and removed when the volume closes.

**Only fibers near surface** retains whole loaded fibers within the chosen
distance, on either side of the surface, even when the mesh is hidden. It checks
segment and triangle interiors. Without a surface it shows no fibers. This filter
combines with the other filters; it does not reload excluded source geometry.

Distances are in active CT voxels. Proximity is not evidence that a fiber belongs
to a particular sheet. These controls do not reconstruct/deform surfaces or
validate H/V non-crossing.

## CT in the 3D view

### Ribbon

Select a fiber and enable **CT ribbon**. Width is centered on the fiber, ranges
from 1–512 CT voxels, and defaults to **32 vx**. The grayscale window is shared
with CT views. There is one sample per texel, with no depth slab or MIP.

In Main, the ribbon uses the shared Line Annotation surface builder, Lasagna
resolution, normal orientation and annotation display overrides. A configured
but unreadable Lasagna dataset reports an error. Without Lasagna, the builder's
fallback frame is used and **Rotation** allows adjustment around the curve.
With Lasagna, rotation is locked to the Line Annotation surface. In the Line
Annotation dock it is always locked to the live generated surface.

Store geometry blocks do not restart the frame. Missing runs, invalid coordinates
and explicit gaps are not joined. Region clipping happens after construction of
the full curve's frame, so it cannot change the ribbon's orientation.

Sampling uses the existing CT chunk cache without blocking for missing chunks.
Coarse data can appear while the requested level loads; uncovered pixels stay
transparent. Zoom determines the requested pyramid level. Width/rotation changes
are debounced; selection changes and hiding cancel outstanding work. Contrast
changes reuse the image. Texture updates are staged across frames.

A ribbon is limited to 262,144 sampled rows and 8 million texels. Exceeding a
limit reports an error. Uint8 and uint16 CT are supported; uint16 is converted to
the viewer's 8-bit display range.

### Slices and volumetric slabs

**CT slices** displays the linked XY, XZ and YZ planes. XY/YZ start enabled and XZ
starts disabled. Their footprints follow the CT views' position, orientation,
normal offset, pan and zoom. Hidden CT panes use a bounded 1,024-voxel patch
around the shared focus. Slice opacity defaults to 70%.

**Volumetric** shares the mode, layer counts and transfer-function settings from
Composite View. Checked planes become slabs; the 3D camera supplies the viewing
angle. Overlapping slabs are sampled once. Without Volumetric, each plane is a
single slice. Slices and a ribbon can be displayed together.

Sampling runs on workers, first at preview resolution, then at higher detail.
Single-slice textures are limited to 1,024 pixels per edge; slabs are limited to
512 samples per axis and 32 MiB each. The 3D camera, opacity and contrast reuse
sampled CT. Missing or failed reads are reported by the controls.

## Loading and rendering

The defaults are **50,000 fibers** and **512 MiB of decoded coordinates per
view**. The selected fiber loads first, then the longest fibers in the active
region and filters, with object UUID as the tie-breaker. This order is independent of the
camera. A full-volume store load first publishes a bounded preview.

Memory presets range from 64 MiB to 32 GiB. Set **Fiber limit** to 0 for no count
limit; the memory limit still applies. A partial view is explicitly reported.
The same budget bounds the surface mesh buffers. It is not a total application
RAM/VRAM limit: snapshots, spatial indices, rendering buffers, CT and GPU copies
use additional memory.

Curve preparation and proximity queries run on cancellable workers owning value
snapshots. The GUI owns selection and live annotation state. Superseded results
are discarded. Drawing uses a spatial hierarchy and screen-space detail levels
(0.5 physical pixel error at rest, 1.25 during interaction). Source geometry and
picking retain original segments and indices. Simplification does not bridge
runs, styles, invalid points or region re-entry. GPU uploads are staged; very
large collections can still require smaller regions or budgets.

Fiber keys combine the store UUID and object UUID. Loading uses a pinned store
transaction and the store's transformation to the active CT volume; unbound
fibers are excluded rather than drawn in an assumed frame. Catalog pages are
bounded to 128 records. Complete-fiber temporary reads have a separate 64 MiB
budget; an object that exceeds it is reported rather than decimated. Source
geometry and annotation controls are retained at double precision.

Store-route, geometry and registration changes invalidate pending loads. Camera
movement reuses the displayed scene. The ribbon reloads the complete selected
object before constructing its frame, including when the displayed scene was
clipped or memory-limited.

## Verification

With dependencies already installed and CMake configured, build and run:

```sh
cmake --build build --parallel 5 --target VC3D \
  test_open_data_manifest test_lasagna_line_view_surfaces \
  test_fiber_3d_data test_fiber_3d_geometry test_fiber_3d_ribbon \
  test_fiber_3d_slices test_fiber_3d_volume test_fiber_3d_surface test_fiber_3d_canvas
ctest --test-dir build --output-on-failure \
  -R '^(test_open_data_manifest|test_lasagna_line_view_surfaces|fiber_3d_.*)$'
```

Tests cover frames/scales, store identities, registered coordinates, ordering/budgets/cancellation, original indices,
gaps and clipping, proximity against exhaustive search, native CT sampling,
ribbon seams, mesh holes, OpenGL rendering and queued scene replacement. The
canvas test requires a window system; OpenGL-specific tests report a skip when
the required context is unavailable.

Real-data review should include Paris 4 and PHerc1447 fibers imported into an object store, including a complete
collection, a dense local region and a long fiber crossing storage blocks:

1. Compare H/V colors and selection with the CT views; exercise clear, reopen,
   region/length filters, palette changes, orbit, detach and redock.
2. Open a fiber in Line Annotation. Compare ribbon orientation with its flattened
   surface and move the current position from both views. Toggle the ribbon and
   confirm that CT pixels actually appear in 3D. Reselect the current CT volume:
   the fiber, camera and position marker must remain visible and navigable.
   Enable annotation markers and gap styles on a tagged fiber; compare its gap
   and damaged spans, breaks and kollesis terminations with the flattened view.
3. Display one CT slice, then its volumetric slab. Move the linked CT view and
   confirm the 3D plane follows it.
4. Load a registered public surface; check proximity on both sides and with the
   mesh hidden. Remove the surface and switch volumes during pending work.

The implementation has been exercised on macOS arm64. Ubuntu validation remains
required before claiming coverage on both supported platforms. See
[CONTRIBUTING.md](../../CONTRIBUTING.md) for the real-data evidence and
human-commentary requirements for a PR.
