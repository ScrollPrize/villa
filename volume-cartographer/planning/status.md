# Status

- [x] Audit generated-view units, line-ribbon construction, source-level flow,
  coordinate consumers, and introduction history.
- [x] Correct the plan to use one declared view-wide scale and one Zarr LOD.
- [x] Replace input-spacing assumptions with uniform 50-base-voxel target
  resampling at the generated-view boundary.
- [x] Independently review the corrected plan against specification and code.
- [ ] Formalize surface and camera scale units.
- [ ] Add uniform arclength resampling and line-to-strip coordinate mapping.
- [ ] Separate original cut-frame data from resampled ribbon-frame data.
- [ ] Correct line-ribbon `QuadSurface` parameter scales.
- [ ] Consolidate analytic source-Zarr-level selection across render paths.
- [ ] Clarify SurfaceCache source-level versus parameter-step interfaces.
- [ ] Audit all renderable QuadSurface producer declarations.
- [ ] Add regression and mapping tests.
- [ ] Update specification, documentation, changelog, and task log.
- [ ] Build VC3D and run focused tests and rendering benchmark.
