# Fiber surface experiment specification

- Upstream winding/sign/HV solutions are immutable inputs.
- Use actual AFV polylines and their signed arc mapping, not visualized chords.
- Select H pieces with chosen-winding joint probability >0.5.
- Baseline mode keeps UV fixed and uses full positional XYZ error.
- Joint sliding mode adds per-attachment UV displacements at immutable source-arc
  samples. Geometry receives normal error; UV receives tangential error with
  appropriate detach routing. Both update together from the first optimizer step.
- Sliding mode uses the supplied deformed retained mask for both input layouts,
  rather than applying an additional confidence threshold. H-only remains fixed.
- Displacements are smooth along source fibers, penalized relative to input UV,
  with nominal limits 0.1 local winding width and 1000 vertical base voxels.
- Extrapolation creates geometry, not extra supervision; record its extent.
- Reuse Lasagna model, optimizer and regularizers; do not clone their math.
- Preserve all existing workflows and output tifxyz base-voxel coordinates.
- No claim that regularization guarantees a globally nonintersecting surface.
- CT rendering uses bilinear surface geometry and native blocking trilinear CT
  sampling. Overlay uses saved final H attachment UVs on the same pixel grid,
  separating cropped piece spans. TIFF outputs include validity and provenance.
- Opt-in repair freezes patch boundaries and saved UV while reconstructing XYZ;
  continuation restores joint updates relative to original UV, with fresh Adam
  state. Failed repairs leave geometry unchanged; save all comparison stages.
- Explicit untangling uses aligned fiber normals as fixed orientation evidence,
  minimizes signed-corner violations and backtracks subsequent updates to preserve
  attained positive Jacobians. Report any unresolved defects; no global guarantee.
