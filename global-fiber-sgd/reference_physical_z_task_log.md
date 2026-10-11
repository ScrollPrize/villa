# Matched physical-Z comparison (archived)

- reference_physical_z_plan.md independently reviewed and approved.
- Optional physical_z_range_base=[0,80000] maps physical XYZ-Z to rows0–999 in
  every map. Same physical horizontal/vertical scale; retain TIFF/SVG and colors.
- Reference background remaps valid adjacent row intervals to physical Z,
  including reversed intervals, without crossing missing vertices.
- Projected fibers use exact same coarse-cell triangle interpolation as cache;
  fitted fibers use original XYZ-Z, not optimized V. No reoptimization/projection.
- Approved runner run_reference_fullpass151_ccw_physical_z.json launched.
- New regression tests cover shared mapping endpoints, missing rows and exact
  triangle-based projected Z. Previous UV render mode remains available.
- Completed exit0; all output summaries confirm identical Z0–80000 and1000px.
  Results/paths: reference_physical_z_results.md.13tests passed including actual
  LZW TIFF blank padding. No optimization/inference rerun.
