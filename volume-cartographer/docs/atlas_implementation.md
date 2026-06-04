# PLAN

- Add an atlas core module with an in-memory model, JSON persistence, V1 fiber identity, and atlas directory layout under `<volpkg-root>/atlases/<atlas-name>/`.
- Create `New atlas from line` for saved fibers.
- Pick the base mesh from the fiber centermost control/line point by nearest shell.
- Rotate the base mesh by cyclic grid-column index only, preserving all point coordinates.
- Map the fiber onto the rotated base mesh using Lasagna normals and base-surface ray projection.
- Show the new atlas in the Atlas workspace and refresh Atlas Overview.
- Add focused regression tests for persistence, base selection, idx rotation, mapping, and mismatch stop behavior.

# Implemented Status

- Added `vc_atlas` with atlas metadata, links, fiber mappings, JSON read/write, unique atlas directories, nearest shell selection, idx rotation, base mesh copy, and V1 line mapping.
- Added `New atlas from line` to the fiber dock context menu when a fiber is selected.
- Wired atlas creation through `LineAnnotationController`, using the current volpkg, saved fiber JSON identity, selected Lasagna dataset, and available tifxyz shells.
- Added an Atlas canvas widget that renders the idx-rotated base mesh in atlas grid coordinates over seed winding `-1..+1` and draws the mapped fiber and control anchors.
- Added Atlas Overview refresh and activation for saved atlas directories.
- Added `test_atlas` coverage for core storage and mapping behavior.

# Current Limitations

- V1 creates one new single-fiber atlas per action.
- Links are represented and persisted but remain empty for V1.
- The base-shell selection currently uses deterministic nearest-point scans over candidate shells; cached full-shell spatial indexing can be added later for startup performance.
- The Atlas canvas is a lightweight 2D atlas-coordinate view, not a full editing surface.
