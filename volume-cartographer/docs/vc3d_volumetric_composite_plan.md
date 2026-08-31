# Plan: Perspective volumetric compositing mode for the VC3D flattened-segment view

Status: **implemented, revised** (phases 1–4; see `apps/VC3D/volume_viewers/VolumetricCompositor.{hpp,cpp}`,
`CameraGizmoWidget.{hpp,cpp}`, the volumetric branch in `CChunkedVolumeViewer.cpp`, and
`core/test/test_volumetric_compositor.cpp`. Persistence was skipped — the existing composite
settings aren't persisted either. The optional Shift+drag rotate and row-parallel compositing
remain future work.)

Revision (2026-07): the original camera model below (output raster pinned to the w=0 plane,
per-layer parallax shifts, perspective-strength control) produced visibly wrong geometry
beyond a few degrees of tilt — layers shear but never foreshorten, so the slab doesn't read
as a rotating object. The implementation instead uses a **true orthographic render of the
rigidly rotated slab**: parallel rays along the tilted view direction with an orthonormal
screen basis, rotation centered on the view center at w=0. Per layer this is still one
affine map (a 2×2 matrix shared by all layers — the cos(tilt) foreshortening — plus a
per-layer w·tan(tilt) offset), so the cost is unchanged. The camera is a turntable:
azimuth spins the patch about the surface normal (a pure in-plane rotation of the image,
even at zero tilt) and tilt then tips the view about the screen-horizontal axis (always a
vertical foreshortening on screen; GUI-limited to 45°). The two are edited independently
by a two-pane gizmo: an azimuth compass dial and a true-angle elevation gauge.
Perspective is a true pinhole camera on the tilted view axis: the slider maps to a half-FOV
of `perspective · 45°`, and the focal length equals the camera distance so magnification is
exactly 1 on the plane through the view center perpendicular to the view axis — switching
ortho ↔ perspective preserves the central ray and the coverage at that anchor depth
(perspective then magnifies nearer layers and shrinks farther ones). A `wScale` relief
exaggeration factor stretches the slab along the normal before the render. Consequence: at
nonzero tilt the w=0 plane is no longer pixel-identical to the untilted view (content
compresses by cos(tilt) across the tilt axis). Scene overlays follow this exactly at w=0:
the viewer's screen↔surface funnel applies the camera's w=0 plane homography (see the
slice-view geometry note below), so overlays only misalign for content off the w=0 plane.

Extension (2026-07): the mode also works in the plane (slice) views, contrary to the
original flattened-view-only scope below. The slice slab is the same layer stack the plane
scalar composite samples (`planeLayersFront/Behind` around the slice plane), handed to the
same `VolumetricCompositor` — and since that slab is genuinely rigid, the tilted render is
exact there (no bendy-slab approximation). Gating follows the per-plane composite
checkboxes; the gizmo appears in each active viewer (shifted left of the plane tilt
handle). The camera is per-view: the panel's camera spinners/slider/reset map to the
flattened view only, while each slice view's camera is edited via its own gizmo
(double-click a pane to reset it). The transfer-function params (alpha window, opacity,
gamma) stay shared across viewers. `wScale` stays flattened-view-only — a slice slab has
no surface relief to exaggerate.

Slice-view geometry (2026-07 follow-up): a slice view's **azimuth is not applied by the
compositor at all** — it is folded into the slice plane's own basis
(`AxisAlignedSliceController` adds a handedness-corrected in-plane rotation on top of the
up-alignment; `CChunkedVolumeViewer::_volumetricAzimuthInSurface` zeroes the azimuth in
the render job). This keeps the sampled ROI matched to the viewport at any azimuth (no
rotated-strip coverage loss) and, because every plane-derived consumer (patch–slice
intersections, focus point, other-plane handles, scene↔volume mapping) reads the plane
object, they all agree with the rendered view for free. The plane spins about the world
view center (the viewer re-projects its pointer when its plane rotates in place). The
remaining tilt/perspective camera is exact for content **on** the slice plane: the w=0
screen↔surface map is a plane homography (ortho: y-foreshorten by cos tilt; perspective:
pinhole view of a plane), and `surfaceToScene`/`sceneToSurface` — the single funnel all
overlays and mouse paths go through — apply it, so on-plane overlays align exactly at any
tilt. Off-plane content still has no single correct screen position under a slab render;
it draws at its plane-projected (w=0) position as before. The flattened view keeps
compositor-side azimuth; its overlay funnel uses the same homography, so its overlays now
align at w=0 too.

## Goal

Add a new compositing mode to the flattened-segment (generated-surface) view in VC3D that
behaves like a true volumetric render of the "bendy slab" (the layers extracted along the
surface normals, as configured by the existing composite settings), instead of a purely
per-pixel orthographic reduction (max/mean/min/alpha/Beer–Lambert of corresponding pixels).

Requirements distilled from the request:

- **Coverage stays exactly as today.** The output raster remains the sorta-orthographic UV
  grid of the flattened view; the same UV region is visible at the same zoom/pan.
- **Volumetric look.** Front-to-back alpha compositing along actual view rays, driven by a
  transfer function, through the same slab (layersFront/layersBehind along the normal).
- **Tiltable camera.** Yaw/pitch up to ±90° away from the W axis (the surface-normal axis).
  Translation is always "flat" in UV: the camera lives on a plane parallel to the UV surface
  and slides on it (this is just the existing pan). Zoom moves that plane nearer/farther
  (existing zoom, plus a perspective-strength control).
- **GUI-only.** Implemented inside VC3D's viewer render path, not in the shared
  `core`/CLI compositing code (`utils::composite_*`, `vc/core/util/Compositing.hpp` stay
  untouched except for a small settings-struct extension, see below).
- **A small on-screen gizmo** in the flattened view to rotate the camera.

## Key geometric insight (what makes this cheap)

Work in **slab space** `(u, v, w)`: `u,v` are the flattened-view pixel coordinates, `w` is
the layer offset along the surface normal (the same per-layer offset the existing
`sampleCoordsComposite` uses: `coords + normals * offset`). The slab is a stack of
`numLayers = layersFront + layersBehind + 1` buffers, each aligned to the output raster.

Fix the **output raster to the w = 0 plane** (the "center" of the slab — see "Slab center"
below). Place the camera at

```
C = viewCenter(u₀, v₀, 0) + D · d(yaw, pitch)
```

where `d` is the unit view direction tilted away from +W by yaw/pitch, and `D` is the
camera distance (in layer units × scale). For an output pixel `p = (pu, pv, 0)`, the ray
`C → p` intersects the layer plane at height `w` at parameter `t = (C_w − w) / C_w`, giving

```
X_uv(p, w) = C_uv + t · (p_uv − C_uv)        // affine in p_uv!
```

So **each layer's screen→layer mapping is a uniform scale `t` about the camera footprint
`C_uv` plus nothing else** — an affine scale+translate, not a general homography. Two
consequences:

1. Resampling each layer buffer is trivial and fast (per-pixel bilinear fetch with a
   per-layer `scale, offset`; no `warpPerspective` needed).
2. As `D → ∞` at fixed direction, `t → 1` and the mapping degenerates to a **pure per-layer
   shift** `Δuv = −w · (d_uv / d_w)` — an oblique orthographic render. This is the cheapest
   and most predictable mode and should be the default ("Perspective" slider at 0 =
   orthographic-oblique; increasing it pulls `D` in).

Because the output raster *is* the w=0 plane, when yaw=pitch=0 and `D=∞` this mode
reproduces the current pixel-aligned compositing exactly (identity mapping per layer), and
all existing scene-coordinate machinery (`sceneToVolume`, overlays, cursor, intersections)
remains exactly correct at w=0 regardless of tilt. Coverage is unchanged by construction.

Caveat: treating the bendy slab as flat parallel planes in `(u,v,w)` is an approximation —
rays are straight in slab space, which means they bend with the surface in world space.
That is arguably the *desired* behavior for a flattened view (the render "follows the
page"), and it is what makes tilt well-defined despite the slab being bendy. Document this
in the UI tooltip.

## Rendering algorithm (per frame, in `renderFrame`)

New branch in the generated-surface path of
`apps/VC3D/volume_viewers/CChunkedVolumeViewer.cpp::renderFrame` (next to
`sampleCoordsComposite`), active when the composite method is `"volumetric"` and the
surface is not a `PlaneSurface` (plane views keep existing behavior; the mode is offered
for the segmentation/flattened viewer only).

1. **Generate layer buffers.** Exactly as in `sampleCoordsComposite` today — same
   coords/normals grid, same size, gen cache untouched. For each layer `i`:
   `layerCoords = coords + normals * offset_i`, sample via
   `ChunkedPlaneSampler::sampleCoordsFineToCoarse` into `layerValues[i]`,
   `layerCoverage[i]` (Nearest, as today). Out-of-bounds lookups during compositing are
   treated as **coverage 0 (fully transparent)** — no margin buffers. Consequence: near
   the viewport edge the camera leans toward, tilted rays exit the sampled region and
   composite an incomplete stack, giving a fainter band of width
   `≈ max|w| · tan(tilt) · scale` pixels (tens of px for typical slabs). The view center
   is unaffected and the degradation is graceful; if it ever matters, margin generation
   can be added later as a pure enhancement (larger `gen` size + gen-cache key extension)
   without architectural changes. UI clamps tilt to ±85°.
2. **Composite along rays.** For each output pixel `p`, walk layers **near-to-far from the
   camera** (i.e. from `w = +front` down to `w = −behind` when the camera is on the +W
   side; respect `reverseDirection`):
   - compute `q = t_i · (p_uv − C_uv) + C_uv` (two multiply-adds; `t_i`,
     `C_uv` precomputed per layer),
   - bilinear-sample `layerValues[i]` at `q` (skip if outside buffer or coverage 0),
   - apply the transfer function **per raw sample, before compositing**:
     - `emissionRGB = colorLut[value]` — the existing window-level + colormap LUT
       (`buildWindowLevelColormapLut`), applied to the raw voxel intensity. Colors are
       decided at the sample, then blended — not the other way around,
     - `alpha = opacityLut[value]` — a **separate nonlinear opacity mapping** from raw
       value: `alphaMin`/`alphaMax` window → normalized `ρ`, then
       `alpha = alphaOpacity · ρ^gamma` (256-entry LUT, built once per frame alongside
       the color LUT); `isoCutoff` stays as the hard highpass,
   - scale `alpha` by the **ray-segment length correction** `1/|d_w|` (per-frame constant
     for the orthographic-oblique case; per-layer for finite `D`), so tilted views don't
     look artificially transparent,
   - standard front-to-back accumulation in float RGB:
     `colorRGB += T · alpha · emissionRGB; T *= (1 − alpha)`; early-out when `T < ~0.004`.
   - Output `colorRGB` (clamped) into a `cv::Mat_<cv::Vec3b>` color buffer, coverage 1
     if any sample was valid.
3. **Blit path for this mode differs from the scalar modes:** the composited buffer is
   already colormapped, so the blit writes it straight into the framebuffer, skipping the
   scalar-LUT stage (overlay blending on top is unchanged — it operates on the final ARGB
   pixel). `RenderResult` gains an optional `cv::Mat_<cv::Vec3b> colorValues` used by
   this mode; the carry-forward fill (`fillFromPrev`) copies it alongside coverage the
   same way it copies scalar values today. One knock-on: window/colormap changes now
   require a re-composite rather than a re-blit in this mode — same render-job path as
   any composite-settings change, just noted for expectations. `renderJobsSameGeometry`
   already compares `compositeSettings` with `==`, so adding camera fields there
   automatically prevents stale carry-forward smearing across camera moves.

**Transfer function (phase 1):** two per-frame 256-entry LUTs from raw value —
`colorLut` (reuses the viewer's existing window low/high + colormap selection, so the
color TF is controlled by controls users already know) and `opacityLut`
(`alphaMin`/`alphaMax` window, `gamma`, `alphaOpacity` scale). A small editable
piecewise-linear opacity TF can replace `opacityLut`'s parametric form later without
touching the geometry machinery.

**Performance.** Layer sampling cost equals today's composite mode exactly. The
compositing loop is `O(W·H·numLayers)` with a bilinear
fetch — comparable to the existing per-pixel stack loop; add early-out on opacity
saturation. Runs on the existing render worker (QtConcurrent) like everything else; if
needed, split the pixel loop into row bands with `QtConcurrent::map` as a follow-up. Also
reuse the interactive downsampling that already exists (`startLevel`), nothing new needed.

## Camera model & controls

State (per segmentation viewer): `yawDeg`, `pitchDeg` ∈ [−85, +85] (0,0 = straight down
the W axis = current behavior), `perspective` ∈ [0,1] mapping to `D ∈ [∞ … ~2·slabThickness·scale]`.

- **Translation**: unchanged — existing pan already translates on a plane parallel to UV.
  The camera footprint `C_uv` follows the view center, so panning never changes the
  obliqueness.
- **Zoom**: unchanged — existing zoom (scale) is "the plane moves nearer/farther". The
  `perspective` control is separate and only affects foreshortening across the slab.
- **Gizmo**: a small (~72 px) circular pad drawn in a corner of the flattened view.
  Implementation: a lightweight `QWidget` child of `CVolumeViewerView` (simpler and
  crisper than `QGraphicsItem` overlays, and it must not pan/zoom with the scene — the
  existing `setOverlayGroup` items live in scene space, so a widget is the better fit).
  - Renders a disc with a dot/arrow showing the current `(yaw, pitch)` as a direction
    (dot at center = straight down; dot at rim = 85°); subtle W-axis tripod for context.
  - Drag on the pad sets yaw/pitch continuously (dot follows cursor, radially clamped);
    double-click resets to (0,0); scroll on the pad adjusts `perspective`.
  - Visible only when the volumetric mode is active.
  - Optional keyboard/mouse alternative: `Shift+Right-drag` in the view rotates the
    camera (wire through `CVolumeViewerView`'s mouse handling like existing modifiers) —
    nice-to-have, phase 2.
- While dragging, renders go through the existing debounced `submitRender` path
  ("composite changed" reason), same as slider changes today; the interactive
  level-of-detail path keeps it responsive.

## Slab center / layer range

"Render at the center of the slab": with `layersFront = F`, `layersBehind = B`, today's
stack spans `w ∈ [−B, +F]`. Keep the output raster anchored at `w = 0` (the surface
itself), matching today's coverage semantics exactly, and note in the plan review whether
anchoring at the geometric mid-plane `w = (F−B)/2` looks better; making the anchor the
surface (w=0) is the conservative default because that's where all existing overlays and
picking are defined. (If anchored elsewhere, `sceneToVolume` etc. would silently be
off by the anchor offset — avoid.)

## Settings & plumbing

- `core/include/vc/core/util/Compositing.hpp` (struct-only change, no behavior change for
  CLI): add to `CompositeParams` — method string `"volumetric"`, plus
  `float camYawDeg = 0, camPitchDeg = 0, camPerspective = 0; float tfGamma = 1;`
  (defaulted, `==`-compared for free). Alternatively a separate
  `VolumetricCameraParams` nested struct for clarity. The shared `utils` compositing enum
  is *not* extended — `isSupportedStreamingCompositeMethod` in the viewer gets
  `"volumetric"` added to its own allowlist and the volumetric branch is dispatched before
  `compositeLayerStack` is ever consulted.
- `ViewerCompositePanel` (`apps/VC3D/viewer_controls/panels/ViewerCompositePanel.cpp` +
  the corresponding UI refs in `CWindow`): add "Volumetric" to the mode combo; when
  selected show: opacity, alpha min/max (reuse existing rows), gamma, perspective slider,
  and a yaw/pitch readout with reset button (the gizmo is the primary control; the panel
  shows/edits numbers). Plane-composite checkboxes ignore this mode (plane views fall back
  to single-slice / existing methods — same pattern as the current
  `streamingCompositeUnsupported` handling but scoped to plane path only).
- Persistence via the same mechanism the panel already uses for composite settings
  (`VCSettings`), so mode + camera survive restarts; yaw/pitch reset to 0 on surface
  change is probably friendlier — decide during implementation.
- Status bar (`CChunkedVolumeViewer` status items, ~line 4980): show
  `composite volumetric yaw/pitch persp` so screenshots are self-describing.

## New files

- `apps/VC3D/volume_viewers/VolumetricCompositor.hpp/.cpp` — pure, Qt-free-ish core:
  input = vector of layer buffers + coverage + camera params + the two TF LUTs
  (color + opacity); output = `colorValues` (`Vec3b`) / `coverage` mats. Keeps
  `renderFrame` small and makes the math unit-testable.
- `apps/VC3D/volume_viewers/CameraGizmoWidget.hpp/.cpp` — the pad widget; emits
  `cameraChanged(yaw, pitch, perspective)`.
- `core/test/…` is for core; put compositor tests wherever VC3D app tests live, or (if
  none) add `apps/VC3D/test/test_volumetric_compositor.cpp` mirroring core's test setup.

## Interactions & edge cases to handle

- **Overlays under tilt**: correct at w=0 by construction. Items that represent 3D points
  off the surface (e.g. normals arrows, points with offsets) will not line up with the
  tilted volume rendering — acceptable; no change in phase 1.
- **Overlay volume compositing** (`OverlayCompositeSettings`): keep orthographic
  (unchanged) in phase 1; the blend still happens at w=0 raster so it stays aligned.
- **`zOff` / normal offset** (`adjustSurfaceOffset`): composes as today — it shifts the
  whole slab; volumetric mode treats the shifted coords grid identically.
- **Invalid coords** (`-1`/non-finite in `gen` output): already skipped by coverage; the
  bilinear fetch must treat coverage==0 texels as fully transparent, not as value 0
  (otherwise dark halos appear at patch borders under tilt).
- **prevResult carry-forward**: geometry equality includes camera params (via
  `compositeSettings ==`), so carry only happens for identical cameras. Good as-is.
- **Gen cache**: untouched — layer generation is identical to today's composite path.
- **Edge fade under tilt**: out-of-slab-region rays composite an incomplete stack near
  the leading viewport edge (see step 1). Accepted; margin buffers are a possible later
  enhancement.

## Implementation phases

1. **Settings + plumbing** — struct fields, panel combo entry + param visibility, viewer
   accepts the mode (initially renders with identity layer mappings at yaw=pitch=0,
   including the new RGB blit path). Verifies the pipeline end-to-end.
2. **VolumetricCompositor** — affine per-layer mapping, front-to-back
   TF compositing, segment-length correction, tests for the mapping math (t/shift
   formulas, degenerate cases) and for compositing invariants (yaw=pitch=0 &
   perspective=0 & opacity=1 ⇒ matches first-covered-layer; zero-opacity ⇒ ambient …).
3. **Gizmo** — CameraGizmoWidget, wiring, debounced re-render, reset, clamping.
4. **Polish** — status bar, persistence, tooltips, optional Shift+drag rotate, optional
   row-parallel compositing if profiling (`--profile`, `[vc3d-profile]` logs) shows the
   composite loop dominating.

## Explicitly out of scope

- Shared renderer/CLI compositing (`utils/compositing.hpp`, `vc_render` tools).
- True world-space (non-slab-following) perspective rendering.
- Lighting/shading (the removed lighting code stays removed) — emission-absorption only.
- Fully general editable transfer functions — phase 1's color TF is the existing
  window-level + colormap LUT and the opacity TF is parametric (window + gamma + scale);
  an editable piecewise TF is a possible follow-up.
