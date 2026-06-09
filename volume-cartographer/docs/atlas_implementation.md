# Atlas Implementation

Atlas generation stores V4 metadata, an unchanged copy of the selected base mesh, typed links, and V3 fiber mappings under `<volpkg-root>/atlases/<atlas-name>/`. Atlas creation reads the selected `.lasagna.json` manifest, resolves its `init_shell_dir` relative to that manifest, and uses only `shell_*.tifxyz` directories from that init shell directory as base candidates. Saved fiber `linePoints` are projected onto the selected base mesh using Lasagna normals and adaptive base-surface ray projection.

Atlas anchor U/V values are stored in base-mesh-relative grid coordinates. The metadata field `zero_winding_column` is not a coordinate transform; it only defines the origin for winding labels and display ranges:

`winding = floor((atlasU - zero_winding_column) / period_columns)`

## Atlas Storage Model

Atlas creation copies the selected `shell_*.tifxyz` base mesh as-is into the atlas directory. It does not rotate, reindex, or shift the saved base shell columns. The first and duplicate closing seam columns remain exactly as they were in the source shell, and ancillary surface channels are copied without column rotation.

Fiber anchors store source geometry in `world` and mapped-shell coordinates in `atlasU`/`atlasV`. Projection hits keep their raw base mesh U/V values. Continuation mapping may unwrap `atlasU` across the seam for line continuity, but the modulo column position still refers to the original base mesh columns. Atlas layout never rewrites anchor coordinates or source fiber geometry.

Each mapped fiber stores an integer `winding_offset`, derived from atlas links at load/export time and defaulting to `0` before layout. Display and footprint calculations use:

`actual_atlasU = anchor.atlasU + winding_offset * period_columns`

Stored anchor U/V values remain immutable base-relative coordinates.

`zero_winding_column` records the column that should be treated as winding zero for interpretation. It replaces the old `idx_rotation_columns` metadata from V1 atlases; old atlas metadata and old fiber mapping versions are rejected by normal loads and require an explicit rebuild from source fiber JSON.

## Atlas Links And Layout

`links.json` stores typed atlas links. Each endpoint currently references a fiber object by `fiber_path` and records the inspected endpoint source index, arclength, and base atlas U/V coordinate. Old string links from early V2 development builds are ignored when loading.

The link field `desired_winding_delta` means:

`actual_winding(second) - actual_winding(first) = desired_winding_delta`

Atlas layout treats links as an undirected graph. `layoutAtlasObjects(atlas, periodColumns)` starts from the root mapped fiber, currently the first fiber in the atlas mapping list, gives it offset `0`, and flood-fills integer offsets through the link graph. Unreachable fibers keep offset `0`. Conflicting cycles are not resolved yet; the first flood-fill path assigns the offset.

New atlas objects are added either by creating a new atlas from a seed fiber or by accepting an intersection link from an atlas that already contains at least one of the inspected fibers. If neither inspected fiber is already mapped, the accept operation fails without mutating the atlas.

## Atlas Overview

Atlas Overview intentionally exposes only the minimal object summary:

- `Fiber count`
- `Object covered atlas size`

`Object covered atlas size` is computed from mapped fiber `lineAnchors` only and is displayed as `W x H vx` in nominal volume voxel units using the saved atlas base mesh scale. Fiber winding offsets are applied for width/range calculations when the base mesh period is known. `controlAnchors` are display metadata for source control points and are not used for footprint calculations.

## Atlas Viewer

The Atlas workspace uses the normal chunked slice viewer with `ViewerRole::Annotation`. The saved atlas base mesh is registered as a temporary internal `CState` surface, so Atlas display preserves the standard viewer behavior for pan, zoom, normal-offset scrolling, volume sampling, scalebar, and overlay refresh.

Atlas overlays are rendered through generic surface-coordinate overlay primitives. Atlas grid coordinates, after applying the owning fiber's integer winding offset, are converted into viewer surface coordinates using the `QuadSurface` center/scale convention:

`surface = displayed_grid - center * scale`

The atlas viewer constructs a display-only repeated surface from the saved base mesh, excluding the duplicate closing seam column from each repeat. The repeated surface starts at `zero_winding_column` so winding-zero can be shown first. This is the only place where the old "rotation placement" behavior still exists, and it is a viewer ordering choice only. Overlays convert actual anchor U/V values with the display range's atlas U offset; they do not rewrite or persist shifted anchor coordinates.

The Atlas viewer uses a live overlay controller. It draws each mapped fiber from `lineAnchors` as a line strip and draws source control points from `controlAnchors` as point markers during pan, zoom, normal-offset scrolling, and refresh. `controlAnchors[].sourceIndex` is the original `line_points[]` index, matching `lineAnchors[].sourceIndex`; control anchors are copied from already mapped line anchors and are not independently nearest-matched.

## Atlas Object Search

Atlas Object Search uses atlas fiber mappings only for membership. The first
search mode treats mapped fibers as the source set and all saved fibers as the
target set, covering mapped-to-outside and mapped-to-mapped fiber pairs.
Intersection candidate search and refinement run in original saved fiber volume
coordinates, not atlas display coordinates. Search results are shown in the dock
table.

Opening a result creates an Intersections inspection layout with a bottom-center
decision pane. The pane shows the decided H/V fiber names, manual H/V tags
before automatic scores, and three exclusive choices: `same winding (h inside
v)`, `different winding`, and `hard to say`. The default choice is `hard to
say`.

Accepting `same winding (h inside v)` loads the selected atlas, requires at
least one inspected fiber to already be mapped, maps the missing fiber to the
atlas base mesh when needed, appends a typed link with
`desired_winding_delta = 0`, runs atlas layout, saves, and redisplays the atlas.
For now, accepting `different winding` or `hard to say` only updates the pane
status and does not mutate the atlas.

The atlas viewer accepts Ctrl-click as a focus shortcut for the main volume
view. The shared focus point is updated from the clicked original-volume point,
then the UI switches back to the main workspace without changing the atlas
camera.
