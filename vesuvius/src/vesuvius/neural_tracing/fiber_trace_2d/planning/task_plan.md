# Task Plan: Path-Based Lasagna Volumes And Base-Space Fiber Tracing

## 1. Goal And Accepted Coordinate Contract

Make VC project files describe data with real source locations and make the
native GUI tracer treat coordinate systems explicitly.

The canonical coordinate spaces are:

- **base/fiber space:** persisted line and control-point coordinates;
- **trace space:** runtime sd2 tracing grid; with the current/default fiber
  manifests, one trace voxel spans 4 base voxels;
- **prediction space:** persisted inference-channel voxels; for the current
  manifests, one prediction voxel spans 16 base voxels or 4 trace voxels;
- **normal-channel space:** each regular Lasagna group keeps its own manifest
  scale and is sampled through the shared runtime coordinate adapter.

For a base point `p_base` and derived `trace_to_base`:

```text
p_trace = p_base / trace_to_base
p_base  = p_trace * trace_to_base
prediction_index = p_trace / (prediction_to_base / trace_to_base)
```

The implementation must use the derived scales rather than embedding literal
`4` or `16` in the segment worker. The existing default
`inference_scaledown_power=2` remains the source of the default 0.25x
inference/trace relationship.

## 2. Canonical Project Representation

### 2.1 Manifest entries

Keep the existing authoritative fields:

```json
{
  "location": "/local/data/fiber.lasagna.json",
  "tags": ["vc-lasagna-fiber"]
}
```

or:

```json
{
  "location": "s3://bucket/run/fiber.lasagna.json",
  "tags": ["vc-lasagna-fiber"]
}
```

The manifest entry needs no duplicate identity tag: its `location` is already
the identity. Keep role and unrelated user/Open Data tags only.

### 2.2 Derived ordinary volumes

Replace `lasagna-derived://<encoded identity>` with the actual resolved group
source:

- local group: absolute normalized Zarr array path, including its OME-Zarr
  level suffix when present;
- direct remote relative group: remote manifest parent plus relative group
  key;
- explicit-sidecar relative group: sidecar `artifact_url` plus relative group
  key;
- absolute remote group: the group's own remote locator.

Keep only `vc-lasagna-derived:<actual local manifest path or remote locator>`
as the combined provenance, auto-ownership, and reconstruction marker. Group,
channel, spacing, dtype, and shape remain authoritative in the manifest/Zarr
descriptor and must not be duplicated in project tags.

The `volumes[].location` is a source locator, not a cache locator. A derived
remote entry remains reconstructed by the Lasagna adapter so its exact-byte
read-through cache behavior is preserved; generic project loading must not
silently route it through a different lossy cache path.

### 2.3 Source-location API

Add one shared Lasagna helper that returns a group's authoritative source
location after origin resolution. Do not reconstruct URLs separately in
project code.

The resolver must retain two distinct values:

- human/project source location in its local, HTTP, or S3 form;
- runtime fetch endpoint/cache metadata used by `PersistentHttpStore`.

Use the existing remote URL join/normalization APIs for query-safe path joins.
Do not derive a remote group source from the machine-local cached manifest.

### 2.4 Deduplication and ownership

Normalize actual locations only for comparison; preserve the canonical source
string for serialization.

- Merge repeated references to the same actual Zarr source into one volume
  entry.
- Add one `vc-lasagna-derived:<manifest location>` tag per referencing
  manifest.
- Do not persist descriptive metadata that can be reconstructed from the
  manifest/Zarr descriptor.
- If the volume existed as an independent manual project entry before a
  Lasagna attachment, reuse it without adding an auto-ownership tag.
- Detach removes one provenance tag at a time. Remove the volume entry only
  when its final `vc-lasagna-derived:<manifest location>` tag is removed.
- Reconciliation uses manifest paths and actual group locations only.

## 3. Remove The Unshipped Encoded Representation

Delete it rather than migrating it:

1. Delete `remoteFileIdentityHex()` and `remoteFileIdentityPath()` from the
   public and private implementation.
2. Delete the `remote_lasagna/url_hex` layout and `source_identity_hex`
   sidecar field.
3. Delete synthetic derived-volume location generation and all recognizers or
   tests for that scheme.
4. Do not add a decoder, fallback reader, migration branch, or old-cache
   lookup.
5. Replace affected tests and fixtures with the path-based representation.
6. Treat any locally created project containing the WIP schema as disposable;
   reattach its manifests to recreate it.

The resulting source must contain no implementation or documentation of the
old encoded identity convention.

## 3.1 Readable transparent cache layout

Map normalized remote object paths into readable filesystem components, for
example:

```text
<cache-root>/remote_sources/https/example.org/bucket/run/file.json
<cache-root>/remote_sources/s3/bucket/run/file.json
```

The generic file cache still accepts a caller-selected destination. Its
sidecar records a canonical source path string and size, not an encoded
identity. Split and validate scheme, authority, and path components; reject
empty, absolute-escape, `.`/`..`, or platform-invalid components instead of
encoding an entire locator. Strip query/fragment authentication material from
the persistent identity and layout while using the full supplied locator only
for the in-memory request.

## 4. Base-Space GUI Fiber Trace Adapter

### 4.1 Dataset preparation

Keep the regular line-optimization normal sampler configured for the line's
base storage coordinates. Prepare separate native-trace runtime objects:

- fiber prediction dataset/field configured with
  `workingToBaseScale=trace_to_base`;
- regular Lasagna normal dataset/sampler configured with the same
  `workingToBaseScale=trace_to_base`.

Do not reuse the base-space normal sampler for trace-space points. Cache these
runtime objects in the line session by selected manifest locations and derived
trace scale so repeated segment actions do not reopen descriptors.

### 4.2 Remove the incorrect equality requirement

Delete the condition requiring `trace_to_base == line_to_base`. Replace it
with validation that:

- trace scale is positive and finite;
- all prediction channels agree on persisted prediction scale;
- the default inference scaledown relation is valid;
- both prediction and normal samplers were opened in trace coordinates.

The line storage scale and trace runtime scale are expected to differ.

### 4.3 Segment request conversion

Before launching the background trace:

1. Copy the base-space line and original endpoints.
2. Divide all reference-line points by `trace_to_base`.
3. Keep isotropic direction/plane-normal vectors normalized; their orientation
   does not change under uniform scaling.
4. Run `traceFiberSegment()` entirely in trace voxels.
5. Multiply every accepted fused point by `trace_to_base`.
6. Restore the first and last replacement points from the original base-space
   line exactly.
7. Splice the base-space replacement into the base-space line.
8. Rebuild the final line model with the ordinary base-space normal sampler.

Put the conversion in a small testable core/helper API also usable by future
GUI/CLI callers. Do not bury independent arithmetic copies in Qt lambdas.

### 4.4 Base-voxel acceptance and optional physical reporting

The Python native CLI, C++ metric CLI, and VC3D segment action use one fixed
endpoint threshold of `20` base-resolution voxels. Convert working-grid error
to base voxels before acceptance:

```text
endpoint_error_base = endpoint_error_working * working_to_base
```

At the default sd2 trace scale, `working_to_base=4`, so the internal working
threshold is `5` trace voxels. Physical size never controls acceptance. When a
finite positive base-voxel size is available, report
`endpoint_error_um = endpoint_error_base * base_voxel_um`; otherwise omit the
physical report without rejecting the trace.

Trace step, pruning distance, budgets, and other voxel-valued Trace2CP
parameters remain expressed in trace voxels. They are not multiplied before
calling the tracer.

## 5. Implementation Sequence

### Phase A: Remove encoded identity and replace the cache layout

1. Delete `remoteFileIdentityHex()`, `remoteFileIdentityPath()`, and every call
   site.
2. Replace `source_identity_hex` with a canonical readable source-path field
   in generic cache sidecars.
3. Replace `remote_lasagna/url_hex` with the readable hierarchical remote
   source layout.
4. Remove old-cache lookup and compatibility tests; a cold cache after this
   WIP change is accepted.
5. Keep cache-first validation, atomic publication, refresh, invalidation,
   authentication, and exact bytes unchanged.

Expected files:

- `volume-cartographer/core/include/vc/core/util/RemoteFileCache.hpp`
- `volume-cartographer/core/src/RemoteFileCache.cpp`
- `volume-cartographer/core/src/lasagna/ProjectVolumes.cpp`

### Phase B: Expose authoritative group source locations

1. Extend Lasagna group resolution with one authoritative source-location
   value/helper.
2. Populate it for local, direct remote, explicit-sidecar, and absolute remote
   groups.
3. Keep runtime HTTP/S3 endpoint, auth, and cache-root fields unchanged.
4. Add focused resolver tests before changing project serialization.

Expected files:

- `volume-cartographer/core/include/vc/lasagna/Manifest.hpp`
- `volume-cartographer/core/include/vc/lasagna/Dataset.hpp`
- `volume-cartographer/core/src/lasagna/Dataset.cpp`
- `volume-cartographer/core/test/test_lasagna_manifest.cpp`

### Phase C: Canonical path-based project volumes

1. Prepare volume locations from the authoritative group source location.
2. Generate `vc-lasagna-derived:<manifest location>` tags from the
   authoritative `lasagna_datasets` entry location supplied by `VolumePkg`.
3. Update attachment transaction, deduplication, reconciliation, and detach
   ownership rules for shared actual locations.
4. Remove all synthetic location and encoded-tag handling without a migration
   path.
5. Confirm reloading remote derived entries reconstructs their Lasagna exact
   cache-backed runtime volume rather than treating the locator as an unrelated
   generic volume.

Expected files:

- `volume-cartographer/core/include/vc/lasagna/ProjectVolumes.hpp`
- `volume-cartographer/core/src/lasagna/ProjectVolumes.cpp`
- `volume-cartographer/core/include/vc/core/types/VolumePkg.hpp`
- `volume-cartographer/core/src/VolumePkg.cpp`
- `volume-cartographer/core/test/test_lasagna_project_volumes.cpp`
- `volume-cartographer/core/test/test_volume_pkg.cpp`

### Phase D: Add explicit trace coordinate conversion

1. Add a shared/testable base-to-trace conversion helper.
2. Change GUI inference preparation to retain the derived trace scale without
   comparing it to base line scale.
3. Prepare a trace-scale regular normal sampler separately from the base line
   sampler.
4. Convert request points into trace space and accepted results back into base
   space around `traceFiberSegment()`.
5. Convert endpoint error to base voxels for acceptance and use physical voxel
   size only for optional reporting.
6. Keep endpoint replacement exact and final line reconstruction in base
   space.

Expected files:

- `volume-cartographer/core/include/vc/fiber_tracer/FiberTrace.hpp`
- `volume-cartographer/core/src/fiber_tracer/FiberTrace.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.cpp`
- `volume-cartographer/apps/VC3D/LineAnnotationController.hpp`
- `volume-cartographer/core/test/test_fiber_trace3d.cpp`

### Phase E: Integration audit

1. Verify local and remote attachment menus persist readable paths.
2. Verify selected regular/fiber manifest fields remain actual locations.
3. Verify CLI remote-manifest caching and scale behavior remain unchanged.
4. Verify ordinary line optimization still uses its base-space normal sampler.
5. Verify detach/reload does not remove independent volume entries.

## 6. Testing And Validation

### 6.1 Project/path tests

- Local manifest with relative Zarr groups writes actual resolved local paths.
- Direct HTTP/S3 manifest with relative groups writes actual resolved remote
  locators while a second open hits the transparent cache.
- Explicit `lasagna-remote.json` resolves group paths from `artifact_url`.
- Absolute remote group paths remain unchanged.
- Project JSON contains only actual manifest/group source paths and readable
  `vc-lasagna-derived:<manifest location>` provenance.
- Source and documentation contain no encoded identity helper, `url_hex`,
  `source_identity_hex`, synthetic derived scheme, decoder, or migration path.
- Two manifests referencing one Zarr source deduplicate and retain two readable
  provenance tags.
- Detaching one manifest preserves the shared volume; detaching the last owner
  removes only a manifest-owned volume.
- An independently attached volume survives all manifest detaches.
- Remote credentials/query diagnostics are not copied into logs or cache
  metadata beyond the project's explicitly supplied source locator contract.

### 6.2 Scale tests

- Manifest `source_to_base=1`, prediction group level 4, default inference
  power 2 resolves `prediction_to_base=16`, `trace_to_base=4`, and prediction
  spacing 4 trace voxels.
- Nontrivial base points round-trip base -> trace -> base within floating-point
  tolerance; original endpoint objects are restored exactly.
- Prediction and normal sampler probes at a trace point address the same base
  location as a direct base-space probe.
- The GUI/core segment adapter passes a base span of 64 voxels to the tracer as
  16 trace voxels and returns a fused base-space span with exact endpoints.
- A one-trace-voxel endpoint error at sd2 is reported as four base voxels and
  is accepted against the fixed 20-base-voxel threshold.
- Missing physical voxel-size metadata still permits tracing and omits only
  micrometer output.
- Repeated segment actions reuse trace-scale datasets and do not rebuild on the
  expected base/trace scale difference.
- Existing whole-fiber metric scale tests continue to pass.

### 6.3 Commands

Use the existing configured build tree and focused targets first:

```bash
cmake --build volume-cartographer/build/ci-tests-clang-systemdeps \
  --target test_remote_file_cache test_lasagna_manifest \
  test_lasagna_project_volumes test_volume_pkg test_fiber_trace3d \
  test_open_data_manifest VC3D vc_fiber_trace_metric -j32
```

Run focused binaries/CTest entries with `--output-on-failure`, followed by the
broader applicable core/VC3D suite if the focused set passes. Run
`git diff --check`. No network-dependent test is allowed.

Perform a manual VC3D smoke test with the supplied project shape:

1. open the project and confirm volume entries show readable source paths;
2. save and inspect JSON for actual local/remote source locations;
3. select the base volume and optimize one generated fiber segment;
4. confirm tracing runs at derived scale 4 and the stored line/control points
   remain in base coordinates;
5. reopen the project and confirm the resulting fiber geometry is unchanged.

## 7. Spec Update

Update `planning/specs.md` to:

- require actual local/remote manifest and group source locations in project
  JSON;
- prohibit cache identities, reversible encodings, hashes, and synthetic
  locations as project source fields;
- state explicitly that the unshipped encoded representation has no backward
  compatibility;
- replace the incorrect trace/line scale equality requirement;
- define base-space persisted fibers, runtime trace space, prediction spacing,
  trace-scale normal sampling, result conversion, exact endpoints, and
  physical-unit conversion.

Replace the old cache-layout specification with the readable source-path
layout and explicitly remove backward compatibility for the WIP layout.

## 8. Documentation Updates

Update:

- `volume-cartographer/docs/vc3d_project_files.md` with readable local and
  remote examples plus ownership/deduplication behavior;
- `volume-cartographer/docs/remote_file_cache.md` with the readable mirrored
  source layout and actual source-path sidecar metadata;
- `fiber_trace_2d/docs/code_structure.md` with the base/trace/prediction
  coordinate adapter and separate normal samplers;
- planning task log/status throughout implementation.

## 9. Changelog Update

Add one completion entry covering:

- path-based Lasagna-derived project volumes and removal of encoded identities;
- base-space VC3D fiber persistence with sd2 runtime tracing and correct
  physical scale conversion.

## 10. Risks And Guardrails

- **Shared-source ownership:** actual-location deduplication can accidentally
  delete a manually attached volume. Preserve independent ownership explicitly
  and test it.
- **Remote origin loss:** materialized manifests must not turn cache paths into
  project sources. Carry authoritative origin separately and test every origin
  form.
- **Signed locators:** source locators are user-provided project data, but
  diagnostics and cache metadata must still follow existing redaction rules.
- **Sampler-space mix-up:** using the base normal sampler during trace-space
  tracing silently samples the wrong location. Maintain separate typed/session
  fields and integration tests.
- **Threshold-unit regression:** comparing a working-grid error directly to
  `20` would make acceptance scale-dependent. Test nonzero errors with
  `working_to_base=4`.
- **Endpoint drift:** multiplication back into base space can perturb CPs.
  Restore endpoints from the original stored points exactly.
- **Cache path safety:** readable URL mirroring must reject traversal and
  platform-invalid components without falling back to whole-locator encoding.
- **Cold cache:** removing the unshipped URL-hex layout intentionally discards
  compatibility with any WIP cache contents.

## 11. Plan Review

The plan was checked directly against `task.md`, `planning/specs.md`, the
current manifest/cache implementation, project attachment/reconciliation code,
and GUI trace call path. Independent agent review is not performed because the
active runtime policy prohibits delegation unless the user explicitly requests
it; this is recorded rather than silently skipped.
