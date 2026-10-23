# Fibers 3D in VC3D

Open **View → Fibers → Fibers 3D** in the main workspace. The optional dock sits
beside the existing CT views. It can be resized, floated, and docked again; its
layout is saved with the workspace. It does not load another copy of the CT.

Drag to orbit, middle/right-drag to pan, and use the wheel to zoom. Click a fibre
to select it in the existing annotation or automated-fibre list. Right-click a
fibre to centre the CT views at the picked point or open Line Annotation.
**Fit** frames the displayed geometry; **Fit selected** frames its selected
portion. Changing filters or styles preserves the camera.

The gear menu controls:

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
- Line width, unselected opacity, projection and region bounds.
- Optional annotation markers/styles. These are off by default, and selected
  fibres only is the default when enabling them. Whole-fibre tags appear in the
  selection details. Control points, `break` and `kollesis_termination` have
  distinct markers; `damaged` has a separate pattern. Link overlays are not
  drawn. Known `gap` spans and AFV inferred gaps are dashed by default.

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
