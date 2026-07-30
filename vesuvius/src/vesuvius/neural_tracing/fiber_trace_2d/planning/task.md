# VC3D Path-Based Lasagna Volumes And Base-Space Fiber Tracing

## User request

Remove the hexadecimal Lasagna identity scheme completely. Project state and
cache metadata must use actual source locations that a developer can read and
follow:

- `lasagna_datasets[].location` remains the actual local manifest path or
  remote manifest locator;
- each automatically attached Lasagna volume uses its actual resolved local
  Zarr path or remote Zarr locator as its project volume location;
- Lasagna-derived provenance tags contain the actual local or remote manifest
  location, not an encoding, hash, cache path, or synthetic source location;
- remote access continues to use transparent persistent caching without
  exposing cache identities in project JSON or the GUI.

The encoded representation was unshipped WIP. Do not implement backward
compatibility, decoding, migration, fallback lookup, or preservation for it.
Remove the encoder, the encoded cache layout, the encoded sidecar identity,
the synthetic derived-volume scheme, and their tests/documentation. Projects
created with that WIP representation must be recreated by reattaching their
manifests.

Correct native GUI fiber segment tracing so fiber lines remain stored in their
native base-resolution coordinate system. The default fiber tracer works on
the sd2 grid, or 0.25x linear resolution (`4` base voxels per trace voxel for
the current/default manifests). The GUI must:

1. convert the base-space reference segment into trace coordinates;
2. let the prediction and normal samplers map trace coordinates into their
   persisted channel grids;
3. trace and evaluate the segment in trace coordinates;
4. convert the accepted fused segment back into base coordinates;
5. splice and persist only base-coordinate points, keeping original control
   point coordinates exact.

The current requirement that line storage scale equal trace working scale is
incorrect and must be removed. Physical endpoint thresholds and reported
errors must retain their existing micrometer meaning after the coordinate
conversion.

## Scope

This task covers:

1. Human-readable local/remote manifest provenance and actual Zarr volume
   locations in VC project JSON.
2. Deduplication, reconciliation, detach cleanup, and independent-volume
   preservation using actual source locations.
3. Direct-remote, explicit `lasagna-remote.json`, absolute-remote-group, and
   plain-local manifest source-location resolution.
4. A readable, path-mirroring remote cache layout and source-path sidecar
   validation with no hexadecimal identity helper or `url_hex` directory.
5. Base-to-trace and trace-to-base conversion for the VC3D native fiber
   segment action.
6. Trace-scale prediction and normal sampling while retaining the ordinary
   base-scale normal sampler used to rebuild the stored line.
7. Focused regression tests, VC3D/core builds, documentation, specifications,
   task log, and changelog updates.

## Out of scope

- Do not serialize cache paths, cache keys, credentials, or signed-query
  diagnostics into project JSON.
- Do not add generic 4D volume support.
- Do not change Trace2CP search/scoring behavior, inference values, or model
  output encoding.
- Do not resample or rewrite existing fiber JSON files.
- Do not add a user-facing trace-scale control in this task; retain the current
  default inference scaledown power of 2.

## Correctness constraints

- Actual source location and transparent cached materialization are separate
  concepts. Project state records the source; runtime loaders choose and reuse
  the cache.
- Cache directories mirror normalized remote scheme, authority, and object
  path components. Authentication query parameters remain runtime-only and
  are not persisted.
- A remote relative group path resolves against its authoritative remote
  manifest/artifact origin, never against the local cached manifest path.
- A local relative group path resolves against the local manifest directory.
- One actual ZYX source referenced by multiple manifests appears as one
  project volume with multiple readable manifest-provenance tags.
- Detaching one manifest removes only that manifest's provenance. The volume
  remains if another manifest or an independent manual attachment owns it.
- Fiber line/control-point storage remains in base coordinates. Trace and
  prediction coordinates are runtime-only.
- Default sd2 tracing uses the scale derived from the inference manifest and
  default inference scaledown power. For the current factor-16 prediction
  fields this is 4 base voxels per trace voxel.
- The prediction sampler sees trace coordinates with prediction spacing
  `prediction_to_base / trace_to_base`; it must not be passed base points or
  points pre-divided directly into prediction voxels.
- A trace-space endpoint error is converted to micrometers using the physical
  size of a trace voxel, not the physical size of a base voxel.
- Original stored CP endpoints remain bit-exact after a successful splice.
