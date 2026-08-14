# Task: make generated-view Zarr LOD use declared scale

Fix generated and parameterized VC3D views, especially line-annotation strips,
selecting source Zarr levels from the same explicit, view-wide scale contract as
plane views.

- VC3D has exactly one rendering LOD: the source volume/Zarr pyramid level.
- A render uses one constant scale from framebuffer pixels to level-0/base
  volume voxels. It must not estimate or vary that scale from generated volume
  coordinates.
- Surface parameterization resolution is not an LOD. It only maps surface
  parameter units to a `QuadSurface` point grid.
- `PlaneSurface` parameter units remain one level-0/base volume voxel.
- `QuadSurface` producers must declare their parameterization so one surface
  parameter unit also represents one level-0/base volume voxel. A producer
  without that information must fail rather than infer it for LOD selection.
- Line ribbons must not rely on input points having known or uniform spacing.
  They must arclength-resample to a uniform target spacing of 50 base voxels and
  declare that along-strip spacing plus the exact cross-strip spacing in
  `QuadSurface::scale()` instead of declaring `{1,1}` unconditionally.
- The selected Zarr level must consistently drive demand publication, volume
  sampling, coarse fallbacks, overlays, diagnostics, status, and derived
  surface-cache fills.
- Document the units and ownership of these concepts wherever they are used.
