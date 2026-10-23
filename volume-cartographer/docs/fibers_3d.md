# Fibers 3D in VC3D

Open **View → Fibers → Fibers 3D** in the main workspace. The optional dock sits
beside the existing CT views. It can be resized, floated, and docked again; its
layout is saved with the workspace. It does not load another copy of the CT.

Drag to orbit, middle/right-drag to pan, and use the wheel to zoom. Click a fibre
to select it in the existing annotation or automated-fibre list. Right-click a
fibre to centre the CT views at the picked point or open Line Annotation.
**Fit** frames the displayed geometry; **Fit selected** frames its selected
portion. Changing filters or styles preserves the camera.

A white crosshair marks the shared CT focus at its actual 3D position. Moving
the focus in a CT view updates it immediately without moving the 3D camera;
in whole-volume mode this does not reload the fibres. It is a navigation
overlay drawn above the fibres for visibility, not a point snapped to a fibre.
The marker is hidden when its position is outside the camera view.
Toggle **Settings → More display options → Show cursor** to hide or show it.
It is enabled by default, and the choice is remembered across sessions.

The **Settings** button opens controls for:

- Sources: loaded project annotations and the attached read-only AFV collection.
- H, V, unclassified fibres and minimum full-fibre length. H/V and minimum length
  are shared with the automated-fibre controls.
- The whole volume (default, within the display budgets), or a box around the
  current CT focus (initial width: 1,024 current-volume voxels). Moving the 3D
  camera does not change this box.
- Custom colour; H/V colours; local axial direction (XYZ or angle to X, Y, Z);
  full-fibre length in millimetres with automatic or fixed range. Scalar modes
  offer Viridis, Magma, Cividis and Turbo with a legend. Reversing a fibre's
  point order leaves its direction colours unchanged.
  H/V defaults share the AFV slice palette: H blue (`#40a5ff`) and V orange
  (`#ffad43`). Both colours remain customisable in the 3D settings.
- Line width, unselected opacity, projection and region bounds.
- Optional annotation markers/styles. These are off by default, and selected
  fibres only is the default when enabling them. Whole-fibre tags appear in the
  selection details. Control points, `break` and `kollesis_termination` have
  distinct markers; `damaged` has a separate pattern. Link overlays are not
  drawn. **Dashed gaps** optionally marks known `gap` spans and AFV inferred
  connections; it is off by default. These are metadata-driven display styles,
  not additional gaps created in the source geometry.

Settings are grouped into Sources, Appearance and Annotations. Only controls
for the active colour mode are shown, and the CT box width appears when using
the focus region. **More display options** contains unselected opacity,
perspective, region bounds, the CT cursor and the fibre limit. The popup adapts
its height to the visible controls and scrolls when necessary.

## Geometry and loading

The view uses original stored polylines without resampling or joining separate
runs. Source indices survive region clipping so a pick opens the correct
original line point. CPU coordinates are double precision; rendering subtracts
a local origin before uploading floats. Native annotations are transformed from
the verified annotation frame to the current CT grid. AFV coordinates use the
existing collection identity validation and native-to-viewer scale.

Lengths and length filters use the complete source fibre, not the visible
clipped portion. Unavailable physical scale is not guessed. AFV selections use
collection UUID and fibre ID; native selections use an ID scoped to the loaded
project, and project changes clear the scene.

Loading is asynchronous, cancellable and coalesced. Workers own value snapshots
and independent read-only AFV readers. Stale results cannot replace a newer
scene. Orbit, zoom, selection and colour changes reuse GPU geometry; they do not
request data again. Native annotation snapshots are copied from the existing
controller on the GUI thread.

Each source defaults to 10,000 displayed fibres and a 16 MiB decoded-polyline
budget. The AFV reader also bounds candidate-block bookkeeping and its block
cache. Reaching a budget displays an explicit partial-view message; a region
with too many candidate blocks must be narrowed. These limits bound the new
AFV view's memory rather than pretending an entire scroll fits in GPU memory.

## Review and verification

The implementation is split into a display-only `Fiber3DCanvas`, value adapters
in `Fiber3DData`, and `Fiber3DPanel` for main-workspace UI, selection and loading.
It does not edit fibre geometry, perform inference, or reconstruct surfaces.
The canvas and adapters can be reused in another workspace later.

Antialiased segment caps use the actual framebuffer's MSAA coverage mask rather
than writing full depth for partially covered pixels. This prevents faint cap
edges from hiding the opaque interiors of neighbouring short segments. A
framebuffer without MSAA uses separate core and fringe passes; fringe pixels do
not write depth. Geometry and picking are unchanged.

Build targets: `VC3D`, `test_fiber_3d_data`, `test_fiber_3d_geometry`, and
`test_fiber_collection`. Run CTest names `fiber_3d_data`, `fiber_3d_geometry`,
`test_fiber_collection`.

Tests cover coordinate scaling and unknown frames, complete-source physical
lengths, disjoint runs, segments whose endpoints lie outside the region,
original point indices and span styles across AFV blocks, all family-filter
combinations, UUID mismatch, cancellation, geometry/candidate limits, and
sign-invariant direction colours. A real-collection GUI check should also
exercise orbit/pan/zoom, selection and CT navigation on the target platform.

Validated on macOS / Apple Silicon with Qt 6.11.2, Clang 22 and OpenCV 5:
all three test suites pass. The application check used a real PHerc1447 AFV
(2,373 fibres), including H/V filtering (1,287 V fibres), H/V, direction and
length colours,
orbit without data reload, picking, exact CT centring, hide/reopen and
float/redock. Full-window bridge captures composite the OpenGL framebuffer;
Qt's raster-only widget capture cannot capture this content correctly.

The settings popup was also checked on macOS in H/V, direction and fixed-length
modes, with advanced options expanded and collapsed. Its height follows the
visible controls; all controls remain reachable by scrolling with no horizontal
overflow. The Settings button uses the same text style as the other toolbar
buttons, without an icon or native menu indicator.

The dense-curve rendering regression was reproduced with every segment styled
solid, independently of gap metadata. The corrected renderer was checked on
the complete 2,373-fibre AFV, individual fibres and dense synthetic curves in
orthographic and perspective views, at line widths 2 and 4, and with 10/50/100%
unselected opacity. A standalone check also exercised an actual framebuffer
with `GL_SAMPLES=0`. On the complete collection, 30 orbit frames including
synchronous framebuffer readback measured 18.38 ms median / 28.41 ms p95 before
and 17.77 ms / 24.29 ms after; these are harness timings, not whole-application
frame-rate measurements.

The CT-focus crosshair was checked on the real AFV with the user's 6 mm
filter (29 displayed fibres): moving the CT focus A → B → A changed only
the marker pixels, and 12 focus moves caused no scene reload in whole-volume
mode. Canvas checks also cover a focus outside the fibre bounds but inside the
view, invalid/out-of-view positions, clearing the scene, and large coordinates.
These checks ran on macOS; the crosshair uses portable Qt painting.
