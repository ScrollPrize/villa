# Status

- [x] Audit generated-view units, line-ribbon construction, source-level flow,
  coordinate consumers, and introduction history.
- [x] Correct the plan to use one declared view-wide scale and one Zarr LOD.
- [x] Replace input-spacing assumptions with uniform 50-base-voxel target
  resampling at the generated-view boundary.
- [x] Independently review the corrected plan against specification and code.
- [x] Formalize surface and camera scale units.
- [x] Add uniform arclength resampling and line-to-strip coordinate mapping.
- [x] Separate original cut-frame data from resampled ribbon-frame data.
- [x] Correct line-ribbon `QuadSurface` parameter scales.
- [x] Consolidate analytic source-Zarr-level selection across render paths.
- [x] Clarify SurfaceCache source-level versus parameter-step interfaces.
- [x] Audit all renderable QuadSurface producer declarations.
- [x] Add regression and mapping tests.
- [x] Update specification, documentation, changelog, and task log.
- [x] Build VC3D and run focused tests and rendering benchmark.
