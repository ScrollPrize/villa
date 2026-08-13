# Task log

## Finding

- Queue logs showed line strips publishing level-3 demand for 31-38 distinct
  chunks while fallback selection claimed level 3 was coarse enough to span
  the viewport.
- `fallbackLevelCountForViewport()` divided framebuffer pixels by
  `CChunkedVolumeViewer::_scale`. That is valid for `PlaneSurface`, where scene
  units are base-volume voxels. For `QuadSurface`, `_scale` is pixels per
  surface parameter unit.
- Generated line ribbons use `QuadSurface` scale `{1,1}` and one horizontal
  parameter unit is one optimized line sample. Its physical displacement is
  not one volume voxel. Comparing that viewport extent with volume chunk edges
  therefore stopped fallback generation too early.

## Validation

- Changed the fallback helper to accept an optional scale explicitly named
  `pixelsPerLevel0VolumeVoxel`. Generated surfaces cannot pass their camera
  scale accidentally: the viewer passes `std::nullopt` and receives all
  available fallback levels up to the five-level cap.
- Affine `PlaneSurface` views continue to pass their volume-space scale and may
  stop early based on physical chunk coverage.
- Temporary textual queue diagnostics confirmed that generated strips now
  publish and fetch level-5 chunks. The logging option and logging hooks were
  then removed; the visual active-download overlay remains available.
- Added regression coverage for missing volume-space scale and pyramid-end
  bounding.
- Built `test_chunked_plane_sampler`, `test_chunk_cache`,
  `test_download_queue_debug`, and `VC3D` successfully.
- Focused CTest result: 3/3 passed (`test_chunked_plane_sampler`,
  `test_chunk_cache`, and `download_queue_debug`).
- `git diff --check` passed.
