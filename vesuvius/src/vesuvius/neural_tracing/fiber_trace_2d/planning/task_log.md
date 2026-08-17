# Task Log: existing VC3D strips for replay

## Finding

- The replay implementation incorrectly called `render_surface_image()`, a
  fixed-level legacy helper that does not implement VC3D fine-to-coarse chunk
  fallback.
- Existing offline strip rendering already lives in
  `vc_lasagna_line_probe`: it resizes raw `QuadSurface` coordinates, requires a
  uint8 volume, blocks dependencies at the explicitly selected and coarser
  levels, then calls `ChunkedPlaneSampler::sampleCoordsFineToCoarse()`.
- Interactive `CChunkedVolumeViewer` uses the same fine-to-coarse sampler for
  non-plane surfaces.
- Replay also unnecessarily added a public line-ribbon builder, changed the
  existing line-view caller, fixed rendering to level zero and scale one,
  accepted uint16, generated a discarded mask, and introduced a PNG-specific
  artifact contract.

## Decision

- Restore the original line-view code and call `buildLineViewSurfaces()`.
- Mechanically extract and share the existing line-probe texture helper without
  changing its sampling behavior.
- Require `--volume` to name the concrete OME-Zarr dataset group. Use the `/2`
  group for the current Paris4 sparse local OME-Zarr because it is fully stored.
- Keep only necessary disconnected-layer atlas packaging after rendering.

## Plan Review

- Use `buildLineViewSurfaces(line)` with its exact default configuration and
  existing failure behavior; do not retain replay-specific strip geometry.
- Resolve the selected group against its parent OME `multiscales` metadata,
  transform only the sampling coordinates, and persist that mapping.
- Derive replay texture resolution from the selected Zarr group's voxel pitch;
  replay does not expose the line probe's fixed supersampling control.
- Extract existing `TexturedMesh`, OBJ/MTL, and TIFF conventions too. The only
  new packaging is an affine atlas transform of existing component UVs.
- Delete every alternate symbol/state/test rather than leaving compatibility
  paths, and add a regression around the no-behavior-change extraction.

## Validation

- Built `test_fiber_replay`, `test_lasagna_line_view_surfaces`, `vc_fiberlets`,
  and `vc_lasagna_line_probe` with `-j32`.
- `test_fiber_replay`: 7 cases passed.
- Focused CTest run: `test_fiber_replay` and
  `test_lasagna_line_view_surfaces` both passed.
- The existing `test_chunked_plane_sampler_fallback` regression passed after
  rebuilding its stale binary.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest
  vesuvius/tests/test_view_fiber_presence.py -q`: 67 passed.
- Ruff passed for the viewer and its test file.
- A 512-base-voxel Paris4 replay with `--volume <ct.ome.zarr>/2` produced 18 strict
  visualization manifests and 54 TIFF artifacts. Strict reload accepted all 18.
  All 36 nonempty reference/greedy atlases contained signal (maximum 157); the
  18 fiberlet atlases were the expected empty 1x1 textures because that replay
  had no local fiberlet surface components.
- Every Paris4 sampler report used the explicitly opened `/2` group, render
  scale 1, complete coverage, and zero error chunks.
- Normalized concrete OME-Zarr group directory identity for metadata matching;
  the replay renderer regression now passes the same group with a trailing `/`.

## Deviations

- None.

## Native-resolution correction

- The first implementation rendered every default 21-by-N surface at a fixed
  fourfold matrix scale, even after its coordinates were mapped into a coarser
  selected Zarr group. Napari then retained only values corresponding to the
  original OBJ vertices, discarding almost all stored texels.
- Each component now derives endpoint-inclusive width and height from maximum
  surface arc in selected-group voxel coordinates and calls the unchanged
  renderer at scale one. The napari adapter validates the standard OBJ/UV/TIFF
  atlas and tessellates it in memory to one surface vertex per stored texel.
- Focused validation passed: 7 C++ replay tests and 67 Python viewer tests. A
  512-base-voxel Paris4 run with `/2/`, `--fail 1`, and `--along 64` generated
  18 visualizations. Strict reload accepted all 18; all 36 nonempty CT textures
  contained signal and the 18 empty fiberlet textures remained valid.
