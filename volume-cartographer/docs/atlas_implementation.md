# Atlas Implementation

Atlas generation stores metadata, the idx-rotated base mesh, and V1 fiber mappings under `<volpkg-root>/atlases/<atlas-name>/`. Atlas creation reads the selected `.lasagna.json` manifest, resolves its `init_shell_dir` relative to that manifest, and uses only `shell_*.tifxyz` directories from that init shell directory as base candidates. Saved fiber `linePoints` are projected onto the selected base mesh using Lasagna normals and adaptive base-surface ray projection.

## Atlas Overview

Atlas Overview intentionally exposes only the minimal object summary:

- `Fiber count`
- `Object covered atlas size`

`Object covered atlas size` is computed from mapped fiber `lineAnchors` only and is displayed as `W x H vx`. Legacy `controlAnchors` remain readable in saved mappings but are not used for footprint calculations or Atlas line display.

## Atlas Viewer

The Atlas workspace uses the normal chunked slice viewer with `ViewerRole::Annotation`. The saved atlas base mesh is registered as a temporary internal `CState` surface, so Atlas display preserves the standard viewer behavior for pan, zoom, normal-offset scrolling, volume sampling, scalebar, and overlay refresh.

Atlas overlays are rendered through generic surface-coordinate overlay primitives. Atlas grid coordinates are converted into viewer surface coordinates using the `QuadSurface` center/scale convention:

`surface = displayed_grid - center * scale`

When mapped objects live in a single unwrap, the saved base mesh is displayed directly. When mapped objects span multiple unwraps, the display surface repeats the saved base mesh columns enough times to cover the mapped range, and anchors subtract only the leftmost unwrap column offset.

The Atlas viewer uses a live overlay controller. It draws each mapped fiber from `lineAnchors` as a line strip and draws every line anchor as a point marker so projection density and gaps remain visible during pan, zoom, normal-offset scrolling, and refresh.
