# Task log

## Audit findings

- `CChunkedVolumeViewer::recalcPyramidLevel()` treats camera scale as
  framebuffer pixels per level-0/base-volume voxel. That is the intended
  view-wide contract, not a value to replace with local geometry estimates.
- `PlaneSurface` already satisfies the contract: one surface parameter unit is
  one level-0/base-volume voxel.
- `QuadSurface` represents point-grid density with `scale`, in grid samples per
  surface parameter unit. Correct producers can satisfy the same contract
  without inspecting rendered volume coordinates.
- Line ribbons violate the contract in `LineViewBuilder::buildRibbon()` by
  always constructing `QuadSurface(points, {1,1})`. Existing tests use points
  and cross offsets ten base voxels apart, for which `{0.1,0.1}` is the truthful
  declaration.
- Input line points are not guaranteed to have known or uniform spacing,
  especially after mixed native/Lasagna reconstruction. Optimizer or tracer
  step settings therefore cannot be reused as the strip's declared scale.
- The derived strip can declare an exact scale by arclength-resampling the
  current line. The chosen policy is a 50-base-voxel target maximum, with
  `ceil(totalLength / 50)` uniform intervals and exact endpoint retention.
- Cross-strip spacing is exactly known from half extent and `crossSamples`.
  Auto-sized strips use the same 50-base-voxel target.
- Resampling breaks the old implicit equivalence between original line index
  and ribbon column. Generated-view controls, hover/focus, framing, linked cuts,
  branches, and intersection views need an explicit bidirectional mapping.
- Duplicate consecutive source points collapse to one arclength, so their
  inverse mapping is necessarily canonical rather than one-to-one. A fully
  zero-length line cannot define a ribbon scale and must fail.
- Original-point up/frame arrays are consumed by cut planes and must remain
  indexed like the model. Ribbon frames therefore form a separate resampled
  data set; only ribbon consumers use the strip-position map.
- Several UI paths directly interpret line positions as ribbon columns,
  including held pre-update overlays and intersection inspection. The mapping
  must be versioned with the generated surfaces and propagated through all of
  those consumers.
- Other transient `QuadSurface` producers also need an audit. Serialized
  surfaces already carry a `scale` declaration; transient producers must supply
  a valid declaration or fail at the render boundary.
- `SurfaceCache` currently uses the same integer to select a source volume level
  and derive its surface parameter-grid step. The source Zarr level is the only
  LOD; the latter is an implementation detail that must remain derived.
- The line-ribbon `{1,1}` declaration was introduced by `226fb35546` on
  2026-05-26. Viewport demand publication in `e9416cc21` on 2026-08-12 exposed
  the latent scale error by eagerly publishing the incorrectly fine working
  set. The later fallback-range fix addressed a separate unit mismatch.
- The generated-view scalebar shares the visible symptom only because the line
  ribbon violates the declared parameter-unit contract. Once producers are
  correct, the analytic camera/voxel calculation applies to every view.

## Constraints confirmed

- No finite differences over generated framebuffer coordinates.
- No local or per-pixel LOD selection on warped surfaces.
- No geometry-derived estimate for render LOD.
- No reliance on input line points being evenly spaced.
- No second surface or parameterization LOD.
- One selected source Zarr level for the complete render.

## Deviations

- None.

## Independent review

- Completed after the uniform-resampling correction. All actionable findings
  are reflected in `task_plan.md`.

## Implementation

- Pending plan approval.

## Validation

- Pending.
