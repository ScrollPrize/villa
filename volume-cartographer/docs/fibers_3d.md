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

**CT ribbon** displays the CT directly along the selected fibre in the 3D view.
The button opens a **Width** slider (full width, 1–512 active-CT voxels; default
32) and **Rotation** slider (−180° to +180° about the fibre). Both values are
remembered; the ribbon is initially off. It follows selection from either
the 3D view or the CT overlays. Clearing selection removes it, and the toggle
can still be switched off without selecting another fibre.

Attached Lasagna normals provide its initial orientation, with their signs
aligned along the fibre. Missing normals use the same parallel transport helper
as Line Annotation; if there are no usable normals, the initial plane is
arbitrary and the status says to adjust rotation. The rotation offsets that
baseline along the entire fibre. This is an inspection plane, not a reconstructed
papyrus surface or a claim about which sheet contains the fibre. It does not
modify the AFV or manual annotation. Gaps, invalid points and disconnected runs
are not bridged, and the displayed region still applies. Consecutive AFV storage
blocks sharing the same original point use one continuous orientation sequence;
storage boundaries do not restart the ribbon's frame.

The ribbon samples the active CT at level 0 with trilinear interpolation, one
sample per texel and no depth projection/MIP. Width is centred on the fibre.
The grayscale window follows the existing CT window controls. Sampling and
normal resolution run on a worker, with debouncing and cancellation; camera
movement and contrast changes never read CT or recompute the ribbon. A single
ribbon is cached for immediate hide/show. Texture tiles share one sampled border
row and are uploaded progressively. Requests are bounded to 8 million texels
(16 MiB of grayscale/validity pixels); larger requests show an explicit error
instead of silently reducing resolution. The active CT's existing chunk cache
is reused, with no additional full-volume copy. Uint8 and uint16 CT are supported
(uint16 is converted to an 8-bit display range); missing level 0 is reported.

Ribbon validation: `fiber_3d_ribbon`, `fiber_3d_geometry`, `fiber_3d_data` and
`test_fiber_collection` pass on the M2 Max Mac. The new suite checks curved
transported frames, sign-flipped/partly missing normals, gap/clipping boundaries,
rotation against a known CT intensity field, texture seams and cancellation.
The full application was checked with all 80,887 PHerc1447 fibres loaded, fibre
2,157,532 selected and 53 nearby fibres displayed. Its 873-row native CT ribbon
loaded in 4.03 s initially and about 0.48 s on subsequent cached runs (including
UI debounce/settling); event-loop callbacks continued during loading. Orbit and
contrast reused the same texture payload. Width/rotation changes, rapid selection
changes, hide/show, clear and shutdown during a request passed. No real Lasagna
dataset was attached in that project; normal handling was checked with controlled
normal fields and the existing dataset-resolution checks. The wider
`test_open_data_manifest` binary has one failure at `registeredCatalogRoot ==
sourceRoot` (line 1922), reproduced identically with the pre-change resolver;
its Lasagna pairing checks passed. QA inputs, screenshots, logs and the baseline
reproducer are in `exports/VC3D_FIBERS3D_QA/ribbon-20261024` in the outer workspace.

The AFV block-boundary regression was also reproduced on PHerc1447 fibre
1,433,216 (560 source points, three storage blocks, 20.98 mm). Restarting the
fallback frame per block introduced normal jumps of 87.44° and 96.33°.
The corrected 2,712-row ribbon has exactly the same frames as the unsplit
source polyline, including across its three texture tiles. Tests also retain
real gaps and disconnected runs. The native application check verifies the
32-voxel default, rotation, contrast, camera dragging, selection changes and
cancellation (`run-seam-integrated.py` and `ribbon-seam-integrated.json` in the
same QA directory).

With an existing configured build, compile using `cmake --build build --target
VC3D test_fiber_3d_ribbon -j 6`. Run the focused tests using `ctest --test-dir build
--output-on-failure -R '^(fiber_3d_ribbon|fiber_3d_geometry|fiber_3d_data|test_fiber_collection)$'`.

**CT slices** displays the linked **XY**, **XZ** and **YZ** CT planes in the
3D view, independently of fibre selection. Its controls sit below the toolbar;
each plane has a checkbox and the shared **Opacity** slider runs from 0–100%
(default 70%, numeric arrow steps of 5%). XY and YZ start enabled, XZ disabled;
these choices are remembered, while the CT slices button starts off.

The planes follow the existing CT views' position, normal offset, rotation,
pan and zoom. Their footprints match the visible CT areas. A hidden CT view
has no usable viewport, so its optional plane shows a 1,024-voxel square around
the shared focus instead. The grayscale window follows the existing CT window
controls. CT slices and a selected fibre's CT ribbon can be displayed together.
With **Volumetric** off these are individual inspection slices, with no slab projection or MIP.

Sampling runs on a cancellable worker. A preview (at most 256 texels per edge)
is followed by detail (at most 1,024 per edge), using the available CT pyramid
level appropriate for that sampling spacing. Zooming the linked CT view narrows
the footprint and reveals finer detail. The existing CT cache is reused; the
three final gray/validity textures occupy at most 6 MiB. Orbit, opacity and
contrast only redraw cached textures. Switching planes off and back on reuses
unchanged samples; changed positions discard stale images. A small BSP splits
intersecting planes so their transparency is ordered correctly during orbit.

**Volumetric**, beside the plane checkboxes, mirrors the mode in **Composite
View**. Enabling it from either control renders the checked 3D planes as CT
slabs with real thickness. The 3D XY/XZ/YZ checkboxes determine which slabs are
visible; the separate Plane View checkboxes in Composite View still control the
2D viewers. **Layers Front / Behind**, **Reverse**, the alpha intensity range,
**Material**, **Gamma** and high-pass cutoff are read from the linked views.
The existing 3D opacity slider fades the complete rendered CT. The 3D camera
controls the viewing angle, while each 2D viewer retains its own camera.

The worker extracts bounded 3D textures (at most 512 samples per axis and
32 MiB per slab). A lower-resolution preview precedes detail. The status shows
the physical layer range and sampled grid size; large footprints/ranges use a
coarser display grid without changing their world-space extent. There is no
additional CT disk cache. GPU uploads are limited to 4 MiB per frame. Camera,
window/level and material changes reuse resident textures. GPU ray marching
uses the existing VC3D unit-voxel opacity transfer function, with optical-length
correction so changing the integration step does not change homogeneous-volume
opacity. Up to 512 steps per intersected slab bound the display work. The same
CT in overlapping slabs is integrated once, choosing the finer sampled grid.
Resolved scene depth stops the ray at foreground fibre/ribbon geometry. This
is a world-space volume render, not a composited image pasted onto a plane.

Volumetric validation is split between `fiber_3d_slices` (known intensity
field, physical front/behind bounds, reverse direction, preview and cancellation)
and `fiber_3d_volume` (the real OpenGL renderer, constant-field extinction,
overlap counted once, step-size invariance, bounds, and foreground-depth
occlusion, including an MSAA framebuffer). The latter skips explicitly when
OpenGL 3.3 cannot be created. A separate run of the actual GLSL on an RTX 3090
passed these optical checks. Mac compilation passed; the native runtime checks
and relaunch are pending while the Mac SSH connection is unavailable. Local
QA scripts are in `exports/VC3D_FIBERS3D_QA/volume-20261024`.

Slice validation: `fiber_3d_slices`, `fiber_3d_ribbon`, `fiber_3d_geometry`,
`fiber_3d_data` and `test_fiber_collection` passed on the M2 Max Mac. Tests cover
known 3D intensity fields, axis-aligned/tilted planes, volume bounds, pyramid
preview/detail footprints, cancellation, and intersecting-plane area, UVs and
alpha ordering. The native app was also exercised with the complete 80,887-fibre
PHerc1447 AFV: all three planes, opacity/contrast without resampling, actual
camera drags, individual toggles, a 7-voxel CT normal offset, CT rotation/zoom,
concurrent ribbon display, and cancellation/resume. The QA harness and results
are in `exports/VC3D_FIBERS3D_QA/slices-20261024` in the outer workspace. Ubuntu
was not run. Add `test_fiber_3d_slices` to the build targets and `fiber_3d_slices`
to the focused CTest expression above to reproduce the slice tests.

Selecting an AFV fibre in a CT slice or the catalogue also highlights it in
3D, using the same yellow (`#ffff66`) as the slice overlay and a stroke
2 logical pixels wider than the other fibres, including on Retina displays.
The highlight respects the existing 3D depth and region/length filters.
Other fibres default to 20% opacity while a displayed fibre is selected; the
selected fibre stays opaque. With no displayed selection, every fibre is opaque.
Adjust this in **Settings → More display options → Other fibers opacity**,
using the 0–100% slider or numeric control (5% arrow steps). Existing saved
opacity preferences are preserved; the stored setting remains a 0–1 fraction.

**Settings → Selection → Only fibers near selection** hides distant fibres
in the 3D view. **Distance** starts at 64 active-CT voxels and is
remembered with the checkbox (off by default). A whole loaded fibre is kept
if any of its original segments passes within that 3D distance of any part
of the selected fibre. This includes close approaches between segment interiors,
not just stored vertices, and does not depend on zoom or display simplification.
It does not infer that neighbouring fibres belong to the same sheet.
The existing source, region, length and memory filters still apply: this option
does not load missing fibres or search outside the loaded region. Separate runs
are never joined for the distance query. CT overlays remain unchanged.

Selecting from the CT views or fibre lists updates the neighbourhood too.
Escape/clearing the selection, or disabling the option, restores all loaded
fibres immediately. Camera position, colours and opacity preferences are kept.
**Fit** frames the visible neighbours; **Fit selected** frames the selection.
The status shows the displayed and loaded counts. Neighbour searches reuse
the spatial index and run on a cancellable worker without rebuilding/uploading
the scene. Until a new search completes, only the selection is shown, with
**Finding nearby fibers…**; stale results cannot restore an old neighbourhood.
Hidden fibres and their annotation markers cannot be picked or drawn.

Proximity validation on the M2 Max Mac (QuickBuild `-O1`, Qt 6.11.2), using
the complete `PHerc1447/1447_5mm_v1.afv` (80,887 fibres): selecting fibre
2,157,532 retained 27 / 53 / 123 fibres at 32 / 64 / 128 voxels respectively,
including the selection. End-to-end settling took 114 / 121 / 125 ms in that
single UI run (includes a 50 ms test settling delay; not a throughput benchmark).
The event loop continued processing timer events during the searches. The
same loaded scene was retained. Rapid selection/radius changes, disable,
Escape, scene clearing and clean process shutdown passed in the full VC3D app.
The three suites `fiber_3d_geometry`, `fiber_3d_data` and `test_fiber_collection`
passed. Geometry tests cover interior approaches, parallel/degenerate segments,
Euclidean rather than box distance, disjoint runs, clipping, cancellation,
filtered drawing and picking. macOS was exercised; Ubuntu was not run.

Reproduce the build/tests in an already configured tree:

```sh
cmake --build <build-dir> --target VC3D test_fiber_3d_geometry test_fiber_3d_data -j 6
ctest --test-dir <build-dir> --output-on-failure -R '^(fiber_3d_data|fiber_3d_geometry|test_fiber_collection)$'
```

The local real-data QA harness and screenshots are in
`exports/VC3D_FIBERS3D_QA/proximity-20261024/` in the parent project. Its
`run-integrated.py` relinks the existing VC3D objects with `proximity-startup.cpp`
and runs a disposable copy of the settings/project; it does not modify the AFV.

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
- Line width (default: 1 logical pixel), unselected opacity, projection and
  region bounds (hidden by default). Existing saved preferences take priority.
- Optional annotation markers/styles. These are off by default, and selected
  fibres only is the default when enabling them. Whole-fibre tags appear in the
  selection details. Control points, `break` and `kollesis_termination` have
  distinct markers; `damaged` has a separate pattern. Link overlays are not
  drawn. **Dashed gaps** optionally marks known `gap` spans and AFV inferred
  connections; it is off by default. These are metadata-driven display styles,
  not additional gaps created in the source geometry.

Settings are grouped into Sources, Appearance, Selection and Annotations. Only controls
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
