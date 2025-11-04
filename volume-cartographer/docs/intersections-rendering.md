# Intersection Rendering: End‑to‑End Flow and Controls

This document explains, in depth, how “render intersections” works in the VC3D viewers: when and how intersection overlays are computed, displayed, updated, and removed; what drives differences in appearance; and how UI and filters control behavior.

## High‑Level Overview

- Intersections are 2D paths drawn on a viewer to show where another surface intersects the current view.
- Plane viewers (e.g., `seg xz`, `seg yz`) compute intersections against target quad surfaces visible in the viewport and draw them directly. They also publish the computed intersections to a shared collection.
- The segmentation viewer (`segmentation`) does not recompute; it reads the published intersections and projects them onto the segmentation surface for display.
- A filter (“Current Segment Only”) and other filters drive which target surfaces are considered for intersection rendering per viewer.
- Opacity is user‑controlled, and colors/thickness/z‑ordering vary depending on which surface is involved.

Key components:
- `CVolumeViewer` renders images and overlays, including intersections, and manages per‑viewer state: `apps/VC3D/CVolumeViewer.hpp:32`, `apps/VC3D/CVolumeViewer.cpp:1551`.
- `CSurfaceCollection` stores surfaces, POIs, and computed intersections and emits change signals: `apps/VC3D/CSurfaceCollection.hpp:14`, `apps/VC3D/CSurfaceCollection.cpp:56`.
- `ViewerManager` wires multiple viewers, restores settings (e.g., opacity), and broadcasts UI changes: `apps/VC3D/ViewerManager.cpp:1`.
- `SurfacePanelController` applies filters and sets the per‑viewer target set (`setIntersects`): `apps/VC3D/SurfacePanelController.cpp:1263`.
- Intersection geometry is computed from surface data in `find_intersect_segments(...)`: `core/include/vc/core/util/Surface.hpp:263`, `core/src/Surface.cpp:1348`.

## What Causes Intersections to Be Displayed

Intersections are drawn when all of the following are true:

- The viewer has an active surface (`_surf`) and a valid volume: `apps/VC3D/CVolumeViewer.cpp:1553`.
- The surface is either a plane (`PlaneSurface`) or the special segmentation viewer (`_surf_name == "segmentation"`): `apps/VC3D/CVolumeViewer.cpp:1570`, `apps/VC3D/CVolumeViewer.cpp:1698`.
- The intersection target name is present in the viewer’s target set `_intersect_tgts` (set by UI filters via `setIntersects(...)`): `apps/VC3D/CVolumeViewer.hpp:221`, `apps/VC3D/CVolumeViewer.cpp:789`.
- For plane viewers, the candidate surface’s 3D bounding box intersects the current view’s 3D bbox (derived from the visible ROI): `apps/VC3D/CVolumeViewer.cpp:1608`, `core/include/vc/core/util/Surface.hpp:23`.

Triggers that call `renderIntersections()`:
- Intersection change signal from the shared collection: `apps/VC3D/CVolumeViewer.cpp:771` → invalidates items for the relevant key and rerenders.
- Changing the viewer’s intersection targets (filters): `apps/VC3D/CVolumeViewer.cpp:790`.
- Overlay refresh (e.g., pan/zoom/end of panning via timer): `apps/VC3D/CVolumeViewer.cpp:2255`.
- Viewer surface change and immediate rerender of slice: `apps/VC3D/CVolumeViewer.cpp:1000` (followed by overlay update which reruns intersections).

Where the target set comes from:
- The Surface panel applies filters and updates viewers via `viewer->setIntersects(intersects)` with a union of eligible surface IDs. “Current Segment Only” narrows this set to just the active surface and segmentation: `apps/VC3D/SurfacePanelController.cpp:1320`, `apps/VC3D/SurfacePanelController.cpp:1263`.

## Full Flow: Plane Viewers (seg xz / seg yz)

1) Establish visible ROI and 3D view bbox
- Visible QRectF from the viewport is converted to a 3D bbox on the plane by projecting ROI corners: `apps/VC3D/CVolumeViewer.cpp:1574`–`apps/VC3D/CVolumeViewer.cpp:1581`.

2) Select candidate targets
- Iterate `_intersect_tgts`; for each target surface that is a `QuadSurface` and not already drawn, check bbox intersection (`intersect(view_bbox, segmentation->bbox())`): `apps/VC3D/CVolumeViewer.cpp:1597`–`apps/VC3D/CVolumeViewer.cpp:1626`.
- If the bbox does not intersect, register an empty entry for the key to avoid recomputation until inputs change.

3) Compute intersections per candidate
- For each candidate, compute segments using `find_intersect_segments(...)`, passing:
  - The target’s raw points
  - The current `PlaneSurface`, ROI rect in plane space
  - Step size `4 / _scale` (denser at higher zoom)
  - A higher `min_tries` for the special key `"segmentation"` (1000) to ensure robust coverage: `apps/VC3D/CVolumeViewer.cpp:1632`–`apps/VC3D/CVolumeViewer.cpp:1642`.
- `find_intersect_segments` itself walks the target grid, seeds path tracing with many randomized starts, advances in both directions along the intersection curve with predictive refinement, and splits into disjoint segments when point gaps exceed a threshold: `core/src/Surface.cpp:1348`–`core/src/Surface.cpp:1619`.

4) Appearance selection
- Default for non‑segmentation targets: pseudo‑random color derived from a stable hash of the key; width 2; z‑value 5: `apps/VC3D/CVolumeViewer.cpp:1661`–`apps/VC3D/CVolumeViewer.cpp:1671`.
- Special for key `"segmentation"`: color depends on which plane viewer we are in:
  - `seg yz` → `COLOR_SEG_YZ` (yellow)
  - `seg xz` → `COLOR_SEG_XZ` (red)
  - otherwise → `COLOR_SEG_XY` (orange)
  Width 3, z‑value 20 for priority rendering: `apps/VC3D/CVolumeViewer.cpp:1673`–`apps/VC3D/CVolumeViewer.cpp:1683`.
- All items use the current intersection opacity: `apps/VC3D/CVolumeViewer.cpp:1695`, `apps/VC3D/CVolumeViewer.hpp:197`.

5) Draw the path(s)
- Project each 3D point to plane coordinates, build a `QPainterPath`, and split the path when successive projected points are ≥ 8 pixels apart (discontinuity threshold). Each contiguous chunk becomes a `QGraphicsPathItem`: `apps/VC3D/CVolumeViewer.cpp:1690`–`apps/VC3D/CVolumeViewer.cpp:1714`.
- Store the created `QGraphicsItem*` list under `_intersect_items[key]` for later update/removal.

6) Publish intersections to the shared collection
- After drawing, create an `Intersection` and call `_surf_col->setIntersection(_surf_name, key, ...)`, which stores and emits `sendIntersectionChanged`. A guard pointer `_ignore_intersect_change` prevents the originating viewer from reacting to its own update: `apps/VC3D/CVolumeViewer.cpp:1716`–`apps/VC3D/CVolumeViewer.cpp:1719`, `apps/VC3D/CSurfaceCollection.cpp:56`.

## Full Flow: Segmentation Viewer (segmentation)

1) Read published intersections
- Ask the shared collection for all pairs that involve `"segmentation"`: `apps/VC3D/CVolumeViewer.cpp:1702`–`apps/VC3D/CVolumeViewer.cpp:1711`.

2) Respect the target set
- Skip any keys not in `_intersect_tgts` or already drawn: `apps/VC3D/CVolumeViewer.cpp:1715`–`apps/VC3D/CVolumeViewer.cpp:1723`.

3) Project onto the segmentation surface
- For every 3D point in each published segment, compute the surface‐local 2D location by `pointTo(...)` and `loc(...) * _scale`. Outliers (high residuals) are dropped: `apps/VC3D/CVolumeViewer.cpp:1730`–`apps/VC3D/CVolumeViewer.cpp:1752`.

4) Draw the path(s)
- Build paths with a gap‑split threshold of 8 pixels in segmentation scene space. Pen color reflects which slice the segment came from: `key == "seg yz" ? COLOR_SEG_YZ : COLOR_SEG_XZ`, width 2, and current opacity: `apps/VC3D/CVolumeViewer.cpp:1754`–`apps/VC3D/CVolumeViewer.cpp:1782`.

## What Controls Removal and Refresh

Intersections are removed or refreshed in these situations:

- Target set changes: `renderIntersections()` purges any `_intersect_items` whose keys are no longer in `_intersect_tgts`, deleting their `QGraphicsItem`s: `apps/VC3D/CVolumeViewer.cpp:1557`–`apps/VC3D/CVolumeViewer.cpp:1566`.
- Intersection change signal: `onIntersectionChanged(...)` invalidates and redraws the counterpart when a relevant pair updates, skipping self‑originated changes via `_ignore_intersect_change`: `apps/VC3D/CVolumeViewer.cpp:771`–`apps/VC3D/CVolumeViewer.cpp:785`.
- Overlay refresh: `updateAllOverlays()` explicitly calls `invalidateIntersect()` then `renderIntersections()` to fully rebuild: `apps/VC3D/CVolumeViewer.cpp:2255`–`apps/VC3D/CVolumeViewer.cpp:2268`.
- Viewer surface cleared or switched: clears items and scene as needed: `apps/VC3D/CVolumeViewer.cpp:972`–`apps/VC3D/CVolumeViewer.cpp:1008`.
- Zoom interaction: while the wheel zoom is processed, existing intersection items are temporarily hidden (`setVisible(false)`), then the next overlay update rebuilds them at the new scale: `apps/VC3D/CVolumeViewer.cpp:527`–`apps/VC3D/CVolumeViewer.cpp:548`, `apps/VC3D/CVolumeViewer.cpp:2255`.

## Why Some Intersections Look Different

Primary appearance factors:

- Surface identity special‑cases:
  - Intersections with key `"segmentation"` are emphasized in plane viewers: wider pen (3 px), higher z‑order (20), and deterministic plane color (XY/XY/XZ). Others are thinner (2 px) and lower z‑order (5): `apps/VC3D/CVolumeViewer.cpp:1669`–`apps/VC3D/CVolumeViewer.cpp:1683`.
  - In the segmentation viewer, slice contributors (`seg yz` vs `seg xz`) pick distinct pen colors: `apps/VC3D/CVolumeViewer.cpp:1777`–`apps/VC3D/CVolumeViewer.cpp:1782`.

- Color selection:
  - Non‑special targets use a pseudo‑random pastel‑ish color seeded by the target key’s hash to keep it stable between rerenders: `apps/VC3D/CVolumeViewer.cpp:1661`–`apps/VC3D/CVolumeViewer.cpp:1669`.

- Opacity:
  - Global intersection opacity is user‑controlled (0–100%) and applied to every drawn item: `apps/VC3D/CVolumeViewer.hpp:197`, `apps/VC3D/CVolumeViewer.cpp:795`, `apps/VC3D/ViewerManager.cpp:187`, `apps/VC3D/CWindow.cpp:1515`.

- Z‑ordering:
  - Special `segmentation` lines use `z=20` to draw above other overlays; others use `z=5`: `apps/VC3D/CVolumeViewer.cpp:1679`–`apps/VC3D/CVolumeViewer.cpp:1683`.

- Path segmentation:
  - Long intersections are split into multiple path items when projected point gaps exceed 8 pixels; the number and continuity of visible strokes depends on that threshold and the point spacing from `find_intersect_segments(...)`: `apps/VC3D/CVolumeViewer.cpp:1701`–`apps/VC3D/CVolumeViewer.cpp:1712`.

## UI and Settings That Control Behavior

- Intersection opacity
  - Backed by a `QSpinBox` and persisted to settings; changes are broadcast through `ViewerManager` to all viewers: `apps/VC3D/CWindow.cpp:1527`–`apps/VC3D/CWindow.cpp:1545`, `apps/VC3D/ViewerManager.cpp:177`–`apps/VC3D/ViewerManager.cpp:205`.

- Current Segment Only (Filters)
  - Located in the surface panel; when enabled, the per‑viewer target set is reduced to the active surface (plus `segmentation`), greatly reducing overdraw and speeding interaction: `apps/VC3D/SurfacePanelController.cpp:845`, `apps/VC3D/SurfacePanelController.cpp:1320`–`apps/VC3D/SurfacePanelController.cpp:1330`.
  - All viewers except the segmentation viewer receive `setIntersects(...)` based on filter results: `apps/VC3D/SurfacePanelController.cpp:1299`–`apps/VC3D/SurfacePanelController.cpp:1306`, `apps/VC3D/SurfacePanelController.cpp:1389`–`apps/VC3D/SurfacePanelController.cpp:1400`.

- Plane origin (focus) / slice index
  - Moving the plane’s POI origin (`focus`) or adjusting `_z_off` triggers overlay refresh and hence recomputation: `apps/VC3D/CVolumeViewer.cpp:1035`–`apps/VC3D/CVolumeViewer.cpp:1071`, `apps/VC3D/CVolumeViewer.cpp:544`–`apps/VC3D/CVolumeViewer.cpp:601`.

## Data Exchange Between Viewers

- Plane viewers compute and publish: `CSurfaceCollection::setIntersection(a, b, ...)` where `a` is the plane name (`seg xz`/`seg yz`) and `b` is the target (often `segmentation`): `apps/VC3D/CVolumeViewer.cpp:1716`–`apps/VC3D/CVolumeViewer.cpp:1719`, `apps/VC3D/CSurfaceCollection.cpp:56`.
- The segmentation viewer enumerates `CSurfaceCollection::intersections("segmentation")` and renders arrivals; this decouples computation (aligned to plane sampling) from drawing in segmentation coordinates: `apps/VC3D/CVolumeViewer.cpp:1702`–`apps/VC3D/CVolumeViewer.cpp:1713`.
- Change propagation uses `sendIntersectionChanged`; the originator ignores its own signal via guard pointer to avoid infinite loops: `apps/VC3D/CSurfaceCollection.hpp:33`, `apps/VC3D/CVolumeViewer.cpp:771`.

## Removal Rules Summary

- Removed immediately when the key exits `_intersect_tgts` or when the viewer’s surface is cleared.
- Invalidated and rebuilt on overlay refresh, plane movement, zoom (after interaction), intersection changes, and surface changes.
- Temporarily hidden during wheel zoom to avoid stale geometry during interaction.

## Notes and Edge Cases

- Naming: `_intersect_tgts` defaults to `{ "visible_segmentation" }` in code, but the active logic everywhere else uses `"segmentation"`. The change handlers include a compatibility check for this (`segmentation` vs `visible_segmentation`): `apps/VC3D/CVolumeViewer.hpp:219`, `apps/VC3D/CVolumeViewer.cpp:777`–`apps/VC3D/CVolumeViewer.cpp:784`.
- Performance: candidates are pruned by bbox intersection in 3D, and intersection tracing is parallelized with OpenMP in both candidate enumeration and per‑candidate computation: `apps/VC3D/CVolumeViewer.cpp:1593`, `apps/VC3D/CVolumeViewer.cpp:1629`.
- Density vs zoom: intersection tracing step uses `4 / _scale`, so higher zoom densifies samples and yields smoother paths.
- Path gaps: both plane and segmentation viewers split paths at projected gaps ≥ 8 pixels; noisy/fragmented source data or steep perspective can yield more segments.

## Quick Reference: Primary Entry Points

- Per‑viewer render entry:
  - `CVolumeViewer::renderIntersections()` — `apps/VC3D/CVolumeViewer.cpp:1551`
  - `CVolumeViewer::setIntersects(...)` — `apps/VC3D/CVolumeViewer.cpp:788`
  - `CVolumeViewer::onIntersectionChanged(...)` — `apps/VC3D/CVolumeViewer.cpp:771`

- Shared state:
  - `CSurfaceCollection::setIntersection(...)` — `apps/VC3D/CSurfaceCollection.cpp:56`
  - `CSurfaceCollection::intersections(...)` — `apps/VC3D/CSurfaceCollection.cpp:92`

- Geometry:
  - `find_intersect_segments(...)` — `core/src/Surface.cpp:1348`
  - `intersect(const Rect3D&, const Rect3D&)` — `core/src/Surface.cpp:2077`

- UI hooks:
  - Intersection opacity — `apps/VC3D/CWindow.cpp:1527`, `apps/VC3D/ViewerManager.cpp:177`
  - Filters/Current Segment Only — `apps/VC3D/SurfacePanelController.cpp:845`, `apps/VC3D/SurfacePanelController.cpp:1263`, `apps/VC3D/SurfacePanelController.cpp:1320`

## Tuning Knobs

- Intersection opacity: via main UI spinbox (0–100%), propagated through `ViewerManager`.
- Target set: surface panel filters, most notably “Current Segment Only.”
- Intersection density: indirectly via zoom (affects `step = 4 / _scale`).
- Visual precedence: adjust z‑values and widths in `renderIntersections()` if needed.

## Glossary

- Plane viewer: a `PlaneSurface`‑based view (`seg xz`, `seg yz`) showing volume slices.
- Segmentation viewer: a `QuadSurface` view named `"segmentation"` showing the segmentation surface directly.
- Target set (`_intersect_tgts`): set of surface IDs whose intersections are displayed in a viewer.
- Published intersections: cached `Intersection` objects stored in `CSurfaceCollection` for cross‑viewer reuse.

