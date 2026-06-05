# Atlas Implementation

Atlas generation stores V2 metadata, an unchanged copy of the selected base mesh, and V1 fiber mappings under `<volpkg-root>/atlases/<atlas-name>/`. Atlas creation reads the selected `.lasagna.json` manifest, resolves its `init_shell_dir` relative to that manifest, and uses only `shell_*.tifxyz` directories from that init shell directory as base candidates. Saved fiber `linePoints` are projected onto the selected base mesh using Lasagna normals and adaptive base-surface ray projection.

Atlas anchor U/V values are stored in base-mesh-relative grid coordinates. The metadata field `zero_winding_column` is not a coordinate transform; it only defines the origin for winding labels and display ranges:

`winding = floor((atlasU - zero_winding_column) / period_columns)`

## Atlas Storage Model

Atlas creation copies the selected `shell_*.tifxyz` base mesh as-is into the atlas directory. It does not rotate, reindex, or shift the saved base shell columns. The first and duplicate closing seam columns remain exactly as they were in the source shell, and ancillary surface channels are copied without column rotation.

Fiber anchors are also stored in that same base-mesh-relative coordinate system. Projection hits keep their raw base mesh U/V values. Continuation mapping may unwrap `atlasU` across the seam for line continuity, but the modulo column position still refers to the original base mesh columns.

`zero_winding_column` records the column that should be treated as winding zero for interpretation. It replaces the old `idx_rotation_columns` metadata from V1 atlases; V1 atlas metadata is intentionally unsupported.

## Atlas Overview

Atlas Overview intentionally exposes only the minimal object summary:

- `Fiber count`
- `Object covered atlas size`

`Object covered atlas size` is computed from mapped fiber `lineAnchors` only and is displayed as `W x H vx` in nominal volume voxel units using the saved atlas base mesh scale. `controlAnchors` are display metadata for source control points and are not used for footprint calculations.

## Atlas Viewer

The Atlas workspace uses the normal chunked slice viewer with `ViewerRole::Annotation`. The saved atlas base mesh is registered as a temporary internal `CState` surface, so Atlas display preserves the standard viewer behavior for pan, zoom, normal-offset scrolling, volume sampling, scalebar, and overlay refresh.

Atlas overlays are rendered through generic surface-coordinate overlay primitives. Atlas grid coordinates are converted into viewer surface coordinates using the `QuadSurface` center/scale convention:

`surface = displayed_grid - center * scale`

The atlas viewer constructs a display-only repeated surface from the saved base mesh, excluding the duplicate closing seam column from each repeat. The repeated surface starts at `zero_winding_column` so winding-zero can be shown first. This is the only place where the old "rotation placement" behavior still exists, and it is a viewer ordering choice only. Overlays convert stored anchor U/V values with the display range's atlas U offset; they do not rewrite or persist shifted anchor coordinates.

The Atlas viewer uses a live overlay controller. It draws each mapped fiber from `lineAnchors` as a line strip and draws source control points from `controlAnchors` as point markers during pan, zoom, normal-offset scrolling, and refresh.

## Atlas Object Search

Atlas Object Search uses atlas fiber mappings only for membership. The first
search mode treats mapped fibers as the source set and all saved fibers as the
target set, covering mapped-to-outside and mapped-to-mapped fiber pairs.
Intersection candidate search and refinement run in original saved fiber volume
coordinates, not atlas display coordinates. Search results are shown in the dock
table and are not persisted into atlas metadata yet.

The atlas viewer accepts Ctrl-click as a focus shortcut for the main volume
view. The shared focus point is updated from the clicked original-volume point,
then the UI switches back to the main workspace without changing the atlas
camera.
