# Task plan

## Scope and invariants

- Treat viewport and chunk extents as comparable only when both are in level-0
  volume-voxel units.
- Keep the existing maximum of five coarser fallback levels.
- Preserve affine plane behavior and all queue priority semantics.
- Keep the visual active-download overlay available.

## Implementation

1. Make the fallback helper accept an optional pixels-per-level-0-volume-voxel
   scale rather than an ambiguously named unconditional camera scale.
2. Pass the camera scale only for `PlaneSurface` rendering.
3. Pass no volume-space scale for generated/flattened `QuadSurface` views;
   without a valid conversion, select all available fallback levels up to the
   existing five-level bound.
4. Document the unit contract directly at the API and render call site.

## Testing

- Extend fallback-range unit tests:
  - affine volume-space scale may stop before five levels;
  - absent volume-space scale returns the bounded full range;
  - available-level bounds still apply.
- Build VC3D and focused sampler/cache/overlay tests.
- Run focused CTest cases and `git diff --check`.

## Spec update

- Explicitly distinguish screen pixels per volume voxel from screen pixels per
  surface parameter unit.
- State that generated/flattened surfaces use the bounded full fallback range
  unless an explicit volume-space conversion is available.

## Docs updates

- Update the render/fetch specification; no separate user guide is needed.

## Changelog update

- Add a dated entry describing the fallback unit correction.

## Independent plan review

- The conservative five-level range can add coarse demand for non-affine
  surfaces, which is the intended behavior and remains capped.
- Plane behavior remains numerically identical because its camera scale is in
  the required units.
- No requirement is deferred.
