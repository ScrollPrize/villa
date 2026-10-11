# Archived: H-anchored surface construction from20-sweep maps

- Updated both20-sweep visualizations first using run_uv_categorical_visualization20.json.
  Equal-scale1000pxLZW TIFF +SVG in data/uv_categorical_v1/visualization20.
- User confirmed100vx grid and1000vx maximum search per side. Strict joint>0.5.
  Independent plan/code review approved; added requested reversed-arc/CurveBank/
  source-hash regression tests and degenerate-input guards.
- Reused native orientFiberConstraintDatabase sweep as export-only normal cache.
  No new normal sampler/orientation implementation, no source DB rewrite.
  Full export verified all14,736,670 baked constraint endpoints;512,698fibers,
  45,649,609regular samples;5invalid original samples. Native sweep37.06s,
  full approved-runner wall106.04s; normal cache data/normal_curves_v1.
- Build via bash vesuvius/scripts/run_fiber3d_native_build (RelWithDebInfo).
  Referenced legacy scripts/build_dependencies.sh is absent; inspected existing
  CMake/build runner, used installed dependencies only, no setup/install.
- Surface interpolation uses original AFV curves, signed source arc and inward
  stage1 signs. H exact; V chord residuals projected along interpolated normals.
  V candidates must span H interval before selecting nearest bracket. Missing
  support/invalid or opposed H normals ->holes; no extrapolation/repair.
- Both full-grid exports complete via run_surfaces20.json:CCW44.83s/CW50.59s.
  CCW584950valid vertices,332497quads,1.6959%grid coverage,19807twisted/degenerate
  quads;CW576328vertices,325793quads,1.6626%,19338twisted/degenerate. Long3D
  jumps remain; this is not a clean or anatomically validated surface.
- Reused vesuvius.tifxyz writer:float32 LZW coordinate bands,0/255mask,
  (-1,-1,-1)holes,scale[.01,.01].~12MiB/output. No surface rendering.
- run_surfaces20_audit.json passed stored bands/mask/finite/sentinel/count/scale
  checks and DB/AFV/stage1 SHA256 against frozen tensor evidence. Native loader
  contract inspected; interactive vc3d loading not performed. Normal export
  creation-time DB digest wasn't captured; all endpoint gauges were verified,
  and current source digest is checked against frozen evidence in validation.json.
- Previous100-sweep runs left unchanged. Surface uses saved20-sweep solutions.
- Final focused regression suite31passed in9.91s (surface,vis,anchor,layout).
  git diff --check clean. No unsupported viewer-success claim: vc3d native load
  contract was read, TIFF geometry/scale/mask and source hashes audited.
