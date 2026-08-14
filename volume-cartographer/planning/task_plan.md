# Task plan

## Implementation

1. Formalize the existing render-scale contract in the surface API:
   `PlaneSurface` and renderable `QuadSurface` parameter coordinates are in
   level-0/base-volume voxel units, camera scale is framebuffer pixels per
   base voxel, and the selected source Zarr level is the only render LOD.
   `QuadSurface::scale()` remains point-grid samples per surface parameter unit;
   it is parameterization metadata, not an LOD.
2. Add `targetSpacingBaseVoxels` to `LineViewConfig`, defaulting to `50.0`, and
   reject non-finite or non-positive values. This setting belongs only to the
   derived view and is independent of optimizer steps, tracer steps, and fiber
   persistence.
3. Arclength-resample every input line before constructing ribbon frames. Use
   `max(1, ceil(totalLength / targetSpacingBaseVoxels))` intervals and an exact
   uniform spacing of `totalLength / intervalCount`. This retains both endpoints,
   never exceeds the configured target (50 base voxels by default), and gives
   the whole ribbon one declared along-strip scale. `LineModel::points` is the
   authoritative centerline; `segmentSamples` do not define strip position.
   Input point spacing may be arbitrary.
4. Derive cross-strip spacing exactly from each configured half extent and
   `crossSamples`. Auto-sized strips use the same 50-base-voxel target in the
   cross direction. Construct both ribbons with
   `QuadSurface::scale() = {1 / alongSpacing, 1 / crossSpacing}`. Arclength is
   used to construct the new parameterization, never to infer render LOD.
5. Return a `LineStripPositionMap` with explicit
   `originalPositionToStripGridColumn()` and
   `stripGridColumnToOriginalPosition()` APIs. Convert grid columns to surface
   coordinates only through `QuadSurface::gridToSurface()`/`surfaceToGrid()`.
   Consecutive duplicate input points collapse to one arclength; define the
   inverse to return the first position in that zero-length run, require exact
   round trips only on positive-length segments, and reject a line with zero
   total arclength.
6. Preserve original-point frame/up data for cut planes and model operations.
   Build separate ribbon samples and frames on the resampled centerline, map the
   display anchor through arclength, and resolve/interpolate valid normals before
   the existing whole-line sign pin. Require odd `crossSamples >= 3`, so the
   ribbon has an exact zero-offset center row.
7. Propagate the position map with both current and held generated-view data.
   Replace direct line-index-as-column assumptions in control, seed, branch and
   current-position markers; span labels; hover/context-menu conversion;
   overview recentering; initial framing; pre-update overlays; linked cut
   updates; and intersection inspection. Associate asynchronous results with
   the generated-view/mapping generation that produced them. Optimization and
   serialized line points remain unchanged.
8. Keep `CChunkedVolumeViewer` source-level selection view-wide and analytic
   from camera scale. Consolidate base and overlay selection policy in one
   helper. For each source pyramid independently, preserve the documented
   quality threshold and select the coarsest permitted level whose largest
   base-voxel source-voxel extent is within that threshold; this is conservative
   for anisotropic transforms. Define tie/clamp behavior in the helper tests.
   Base and overlay numeric indices may differ, but each gets one constant
   source level for the complete render.
9. Once every rendered surface obeys the same declared-unit contract, use the
   analytic view footprint for generated-view fallback bounds too. Remove the
   special assumption that all generated surfaces lack a surface-to-volume
   scale; do not inspect generated coordinates or vary level within a frame.
10. Rename ambiguous render-path variables/comments to `zarrLevel` or
   `sourceLevel`. Use that selected level consistently for demand and fallback
   publication, direct and overlay sampling, download-debug maps, status/profile
   data, and render-result reuse.
11. Clarify `SurfaceCache` terminology and interfaces: its requested level is
   the source Zarr level, while its parameter-grid sampling step is a derived
   implementation detail, not a separately selected LOD. Keep the derivation
   analytic from the source level and declared surface units. Unsupported
   transforms must bypass/fail the cache path explicitly, never trigger local
   geometry measurement.
12. Audit all renderable `QuadSurface` producers. Existing serialized `scale`
    metadata is accepted as the producer's declaration; transient producers
    must either provide a valid base-voxel parameter scale or be rejected at the
    render boundary. Record every corrected or explicitly accepted producer.
13. Keep scalebars analytic: physical units per framebuffer pixel are volume
    voxel size divided by camera pixels per base voxel for every surface.

## Compatibility

- Fiber JSON and `LineModel` remain unchanged. Arbitrary and mixed input point
  spacing is normalized only in derived line-strip views.
- Existing serialized `QuadSurface` files retain the established meaning of
  `scale`; no surface-file migration is required.
- Existing line indices remain the model, control, and persistence coordinate.
  The new mapping translates them at the generated-view boundary.
- Rendering remains one source level per accepted render. Warped geometry,
  randomized demand probes, and cache residency cannot alter it.

## Spec update

- Replace the existing generated-surface clause in `spec.md` that says camera
  scale cannot be compared with volume extents; it is incompatible with the
  corrected declared-base-voxel surface contract.
- Define source Zarr level as VC3D's sole render LOD.
- Define surface coordinates as level-0/base-volume voxel units and
  `QuadSurface::scale()` as point-grid samples per surface unit.
- State that camera scale is framebuffer pixels per base voxel for both plane
  and generated views and selects one source level for the complete render.
- Forbid local geometry measurements, finite differences, generated-coordinate
  probes, or cache state from choosing render scale.
- Require generated-surface producers to supply explicit parameter scale and
  fail when no valid declaration exists.
- Define line strips as uniformly arclength-resampled derived views with a
  default target interval of 50 base voxels and an explicit original-line to
  strip-coordinate mapping.
- Clarify that SurfaceCache parameter-grid stride is derived from the selected
  source level and is not a second LOD.

## Documentation updates

- Update `docs/remote_file_cache.md` with the single-LOD model, declared
  view-wide scale, generated-view fallback bounds, and SurfaceCache terminology.
- Update `QuadSurface`, `LineViewBuilder`, `SurfaceCache`, and viewer API
  comments with exact units.
- Document the 50-base-voxel strip target, exact endpoint-preserving resampling,
  and bidirectional line-position mapping in line-annotation documentation.
- Replace the active task log/status and add a concise changelog entry after
  implementation.

## Testing

- Add `LineViewBuilder` tests proving uneven, mixed, and reversed input points
  are resampled uniformly with intervals no larger than 50 base voxels, retain
  both endpoints, and produce the expected reciprocal `QuadSurface::scale()`.
- Cover explicit and auto cross widths, short lines, degenerate segments,
  invalid target spacing, and lines whose length is not divisible by 50.
- Add mapping tests proving original integer and fractional line positions map
  to the correct strip grid column and round-trip on positive-length segments.
  Test canonical inversion and volume-position equivalence across duplicate
  runs, plus alignment of control/branch markers, hover/focus, framing, cuts,
  held overlays, and intersection views.
- Add viewer tests proving plane and equivalently declared QuadSurface views at
  the same camera zoom choose the same single Zarr level, independent of warp,
  coordinate contents, SurfaceCache use, demand probing, and overlays.
- Add SurfaceCache tests proving the selected source level and derived
  parameter step do not become independently selected LODs.
- Add a producer-audit test or validation table covering every transient
  renderable `QuadSurface` construction path and render-boundary rejection of
  invalid scale declarations.
- Add a line-strip demand regression showing the corrected declared scale
  publishes the expected coarse-to-fine Zarr range instead of thousands of
  erroneous level-0 chunks.
- Build `VC3D` and affected tests, run line-view, generated-view, chunked
  sampler, and SurfaceCache tests, then run `git diff --check`.
- Run the virtualized synthetic rendering benchmark before and after; record
  the exact command, build type, dataset, repetitions, and summary statistics.

## Independent plan review

- Independent review completed. Its findings about degenerate mapping,
  original/resampled frame separation, hidden line-index consumers, producer
  coverage, source-level policy, and conflicting existing spec text are
  incorporated above.
