# Fiber surface experiment specification

- Upstream winding/sign/HV solutions are immutable inputs.
- Use actual AFV polylines and their signed arc mapping, not visualized chords.
- Select H pieces with chosen-winding joint probability >0.5.
- Baseline mode keeps UV fixed and uses full positional XYZ error.
- Joint sliding mode adds per-attachment UV displacements at immutable source-arc
  samples. Geometry receives normal error plus optional geometry_tangent_weight
  times tangential error (default0;1 gives full positional geometry loss).
  Geometry evaluation detaches UV and normals. UV receives tangential error with
  appropriate detach routing. Both update together from the first optimizer step.
- Sliding mode uses the supplied deformed retained mask for both input layouts,
  rather than applying an additional confidence threshold. H-only remains the default.
- Connected-HV mode uses the deformed retained H/V graph: all crossing UV positions
  in a connected source-fiber component share one translation; local corrections
  vanish exactly at crossings. Intermediate attachments may slide.
- Before initialization/fitting, connected-HV defaults to whole components with
  >=3 distinct source fibers and both H/V families. component_min_fibers controls
  the minimum; zero disables filtering for legacy replay. Never cut graph edges.
- V data uses weaker, separately normalized normal/tangent loss against immutable
  signed-normal offset targets measured from H at saved crossings; linear source-arc
  interpolation, constant ends, global signed crossing mean for unconnected V.
- Displacements are smooth along source fibers, penalized relative to input UV,
  with nominal limits 0.1 local winding width and 1000 vertical base voxels.
- Extrapolation creates geometry, not extra supervision; record its extent.
- Reuse Lasagna model, optimizer and regularizers; do not clone their math.
- Preserve all existing workflows and output tifxyz base-voxel coordinates.
- No claim that regularization guarantees a globally nonintersecting surface.
- Opt-in other-winding barriers reuse real unoffset H/V points and aligned inward
  normals. Initial per-point U determines phase/bucket (long curves can cross
  boundaries); phase maps using each destination width. Coordinates are frozen,
  geometry-only gradients. Inner/outer ordering uses a weak squared hinge.
  Normalize copy weights before suppressing near same-winding attachments in UV.
  Save/restore all arrays; no new movable fibers or assignments. Fixed phase
  matching is approximate under deformation, not a global collision guarantee.
- CT rendering uses bilinear surface geometry and native blocking trilinear CT
  sampling. Overlay uses saved final H/V attachment UVs on the same pixel grid,
  separating cropped piece spans. TIFF outputs include validity and provenance.
- Future render default:8baseUV voxels per pixel,4pixel fiber lines (half previous
  dimensions and linewidth). Existing images/frozen configs remain unchanged.
- Render existing opposing-triangle quad diagnostics on the exact CT pixel grid,
  with binary TIFF and red CT overlays. Cached fold-only refresh requires no GPU
  or CT re-fetch. Record crop-local and full-surface quad coordinates.
- Opt-in repair freezes patch boundaries and saved UV while reconstructing XYZ;
  continuation restores joint updates relative to original UV, with fresh Adam
  state. Failed repairs leave geometry unchanged; save all comparison stages.
- Explicit untangling uses aligned fiber normals as fixed orientation evidence,
  minimizes signed-corner violations and backtracks subsequent updates to preserve
  attained positive Jacobians. Report any unresolved defects; no global guarantee.
