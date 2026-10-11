# H-anchored fiber-map surfaces

First refresh both20-sweep visualizations; leave100-sweep optimizations running.
Construct both20-sweep conventions as full tifxyz grids, no rendered surface.

## Geometry and interpolation
- Use selected winding joint probability strictly>0.5; actual unmodified AFV
  curves mapped by shared hard_geometry and signed source arc, not vis endpoints.
- Grid100native base voxels, full supported UV bounds. Bracket exactU with H
  below/above and exactV with V left/right. Search maximum1000base voxels per
  side, confirmed by user. Missing support -> holes; no extrapolation.
- V curves must span the selected H interval. Transfer scalar normal displacement
  from each V's endpoint chord, blend left/right, apply along interpolated H
  normals to ruled H baseline. Zero endpoint residual preserves H exactly.
- No dense nearest-neighbor approximation: interval membership and sorted
  orthogonal coordinates select brackets; vectorized curve interpolation.
- Preserve winding offsets as solved; measure seam/jump/fold diagnostics, don't
  silently repair geometry or claim anatomical correctness.

## Reuse / normal export
Extend existing native orientation pass with export-only sampled-curve mode:
same16vx sweep merged with all constraint endpoints, shared sampler/sweep,
validate every baked endpoint against schema2 before publishing. Persist regular
arc samples/normals and ragged index once, then apply stage1 inward signs. No
copied sampler/sweep algorithm and no cloned12GiB constraint database.
Surface code interpolates these normals; invalid support creates holes.

## Format / resources
Reuse vesuvius.tifxyz writer/types: float32 x/y/z TIFFs, sentinel(-1,-1,-1),
mask0/255, scale[.01,.01], metadata UV origin, source hashes/settings. vc3d
QuadSurface loader supports tiled float TIFF and rejects z<=0. Full100vx grid
is about35million vertices (~420MiB coordinate arrays), manageable; vectorized
row/column work, no all-pairs fiber scans. CPU native normals128threads; no GPU
surface work. Existing approved build/evaluation commands only.

## Tests and docs
Synthetic curved-V/flat-H tests, exactH boundary interpolation, confidence,
distance/endpoint holes, reversed arc direction, normal validity and tifxyz
roundtrip/mask/scale tests. Record real timings/coverage/quad diagnostics. Update
task/status/log/results. Independent plan/code review. No installs.

## Spec update
New derived experimental surface only; no upstream optimization or evidence
changes. Interpolated16vx normals (not newly fitted), normal-only V bending.
