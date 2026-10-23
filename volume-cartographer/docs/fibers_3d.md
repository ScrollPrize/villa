# Fibers 3D in VC3D

Open **View → Fibers → Fibers 3D** in the main workspace. The optional dock sits
beside the existing CT views. It can be resized, floated, and docked again; its
layout is saved with the workspace. It does not load another copy of the CT.

Drag to orbit, middle/right-drag to pan, and use the wheel to zoom. Click a fibre
to select it in the existing annotation or automated-fibre list. Right-click a
fibre to centre the CT views at the picked point or open Line Annotation.
Press **Escape** with the 3D view focused, click empty space without dragging,
or choose **Clear selection** in its right-click menu to deselect. This also
clears the selection in the fibre lists and CT overlays; it does not delete
fibres or close the collection. Camera dragging does not clear selection.
**Fit** frames the displayed geometry; **Fit selected** frames its selected
portion. Changing filters or styles preserves the camera.
Selecting an AFV fibre in a CT slice or the catalogue also highlights it in
3D, using the same yellow (`#ffff66`) as the slice overlay and a stroke
2 logical pixels wider than the other fibres, including on Retina displays.
The highlight respects the existing 3D depth and region/length filters.
Other fibres default to 20% opacity while a displayed fibre is selected; the
selected fibre stays opaque. With no displayed selection, every fibre is opaque.
Adjust this in **Settings → More display options → Other fibers opacity**,
using the 0–100% slider or numeric control (5% arrow steps). Existing saved
opacity preferences are preserved; the stored setting remains a 0–1 fraction.

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
- **Memory budget**: 64, 128, 256, 512 MiB, then 1, 2, 4, 8, 16 and 32 GiB,
  defaulting to **512 MiB**. This is the decoded-coordinate budget per source,
  not total process/GPU memory; rendering buffers, indexes and CT need additional
  memory. It is a limit, not a reservation. The chosen byte size is remembered.
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

The view retains original stored polylines and never joins separate runs.
Its display uses levels of detail: the chosen chords deviate by at most
0.5 physical screen pixel at rest, or 1.25 pixels during orbit/pan/zoom.
After 180 ms without movement, it returns to the finer bound. The selected
fibre always uses the exact original clipped segments. These approximations
only affect drawing, including the direction colour of a displayed chord;
they do not edit AFV coordinates, lengths, annotations or exported data.
Source indices survive region clipping so a pick opens the correct
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
scene. Geometry preparation also runs on a cancellable worker. A spatial
hierarchy rejects blocks outside the view before submitting draw calls and
restricts picking to blocks near the click. Picking still tests original
segments, in their original order, with the same depth/distance tie rules.
Orbit, zoom and colour changes reuse prepared geometry; they do not
request data again. Selecting a fibre uploads its exact geometry separately.
Native annotation snapshots are copied from the existing
controller on the GUI thread.

GPU uploads are staged in chunks of at most 8 MiB per repaint, coarse levels
first. During the initial **Refining display…** phase a coarser complete level
can be visible before the requested precision is available. No fibre is removed
by the level-of-detail system. Separate runs, span styles, invalid points and
region re-entry are never bridged. Curves are simplified against finite chords
so hairpins are preserved. Conservative perspective bounds fall back to exact
detail at the near plane. The CPU index references source runs directly instead
of duplicating every segment's double-precision endpoints for picking.

This bounds drawing error, not frame time or total application memory. Loaded
coordinates, prepared display levels, GPU buffers and CT still consume memory;
very large collections may require a smaller displayed region or data budget.
There is no guarantee of unlimited geometry at a fixed frame rate.

Each source defaults to 10,000 displayed fibres and a 512 MiB decoded-polyline
budget. **More display options → Fiber limit → No limit** removes the count
cap while retaining the memory budget; numeric caps up to 1,000,000 are allowed.
Within the region and active filters, the selected fibre is loaded first, then
fibres by decreasing complete-source length, with increasing ID for ties.
This ordering applies to AFV and native annotations. It does not change with
camera rotation or zoom. The CT slice renderer keeps its existing stable sample
ordering. Selection priority does not override region or family/length filters.
The AFV reader also bounds candidate-block bookkeeping (one sixteenth of the
coordinate budget, at least 16 MiB) and its block cache.
Reaching a budget displays an explicit partial-view message; a region
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

Deselection was checked on the published Paris4 AFV (512,698 fibres), using an
isolated profile and a bounded focus region. Escape and an empty-space click
clear both the AFV list and the 3D selection; dragging preserves selection.
An outstanding inspection request cannot restore a cleared selection. With no
selection, 87% and 100% other-fibre opacity produce identical framebuffer pixels.

Memory controls were checked on the full 80,887-fibre PHerc1447 AFV: a fresh
profile starts at 512 MiB; all ten presets are available; 32 GiB survives an
application restart without integer overflow; and the popup has no horizontal
overflow. With a 10,000-fibre cap, 64 MiB loads 5,599 fibres and 512 MiB reaches
the count cap. Lowering the budget preserves selection priority. A separate
data-loader check, without the count cap or length filter, loads all 80,887
fibres in descending length within 494,048,712 decoded coordinate bytes.
That check does not measure full-scroll rendering frame rate or total RAM.

### Full-scroll rendering measurement (24 October 2026)

Measured on an M2 Max / 32 GiB Mac, Qt 6.11.2, Clang 22, the existing
`QuickBuild` configuration (`-O1`), with `1447_5mm_v1.afv`: all 80,887 fibres,
20,472,671 source segments, no count/length filter, 800 × 600 logical pixels at
2× device scale. Each frame calls the actual canvas paint function followed by
`glFinish`; these are rendering latencies, not an unlimited-FPS claim. Five
warm-up frames are excluded, then 60 orbit frames and 10 clicks are measured.

| Operation | Before median / p95 | After median / p95 |
| --- | --- | --- |
| Whole-scroll orbit | 190.24 / 194.03 ms | 3.47 / 6.70 ms |
| Close orbit | 61.02 / 61.47 ms | 3.65 / 5.67 ms |
| Selection click | 286.73 / 302.41 ms | 1.22 / 2.13 ms |

The baseline macOS `sample` profile spent 1,166 of 1,184 main-thread samples
waiting for the GPU in `glFinish` / `GLDShareGroupRec::waitUsage`. The new
preparation/upload took 6.85 seconds once, asynchronously, after reading the
AFV. This does not occur on camera moves. Initial-load peak process footprint
was about 6.54 GB; the coordinate budget is still not a total-RAM bound.

A separate full-application check, with CT views open and all 80,887 fibres,
measured 60 presented frames in a 771 × 882 logical-pixel dock at 2× scale:
8.35 ms median, 16.83 ms p95, 16.97 ms maximum. Shared AFV selection and Escape
passed. Additional canvas checks verify exact original point/index picking in
orthographic and perspective modes, superseded loads, clearing during a load,
and empty scenes. Tests were run on macOS; Linux was not run for this change.

The local evidence and reproducible harnesses are under
`exports/VC3D_FIBERS3D_QA/render-lod-20261024` in the parent workspace:
`build-bench.py baseline` links the saved pre-change canvas,
`build-bench.py` links the current canvas, and `run-integrated.py` runs an
isolated application profile. Each harness records its exact inputs and
JSON timings. `lod-first.json` was an invalid preliminary measurement because
Qt deferred repaints; it is excluded. The reported comparison uses forced GPU
completion in both versions (`baseline-forced.json` and `lod-final.json`).
