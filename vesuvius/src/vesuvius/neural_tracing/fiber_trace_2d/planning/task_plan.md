# Task Plan: VC3D Lasagna Attachment And Generic Remote File Cache

## 1. Goal And Task Boundary

Implement reusable cache-first materialization of arbitrary remote files, use
it for Lasagna manifests in core/CLI/VC3D, and make tagged local or remote
Lasagna manifests attachable through VC3D projects together with all ordinary
project volumes derived from their channel groups.

This task stops after the data can be attached, selected, resolved, and opened
through the shared code paths. It does not alter native Trace2CP search,
Line Annotation interaction design, trace parameters, or trace quality. A
follow-up task will exercise the result interactively in VC3D and the Line
Annotation window.

## 2. Accepted Design Decisions

### 2.1 Canonical project representation

- Keep one `lasagna_datasets` array.
- Use the reserved boolean tag `vc-lasagna-fiber` for learned fiber inference
  data.
- Treat absence of that tag as ordinary Lasagna data.
- Keep independent ordinary/fiber selected-location fields.
- Read and migrate the current `fiber_inference_datasets` field, but stop
  writing it in newly saved project state.

The tag will be declared once in core and consumed through helper functions;
callers must not repeat string literals.

### 2.2 Generic file-cache contract

Add a core utility, tentatively:

- `core/include/vc/core/util/RemoteFileCache.hpp`
- `core/src/RemoteFileCache.cpp`

The utility will cache one exact remote object into a caller-selected path
under a configured cache root. Its persistence layer will not know about JSON,
Lasagna, Zarr, or any other file format.

The API will expose:

- a stable source identity;
- a destination relative to a configured cache root;
- cache policy (`CacheFirst` and `Refresh`);
- an optional caller-provided fetch-to-temporary-file function;
- a standard HTTP/S3 fetch adapter using `RemoteUrl` and `HttpAuth`;
- a result containing the local path, hit/download status, normalized remote
  endpoint, and a read lease when the payload is budget-managed;
- explicit invalidation.

The generic fetch callback writes a cache-owned temporary path. This keeps the
cache usable by future streaming transports without requiring all files to fit
in memory. The initial HTTP/S3 adapter may bridge the existing buffered
`HttpClient` response into that writer, without making buffering part of the
cache abstraction.

### 2.3 Cache identity and publication

- Reuse/extract the collision-free segmented URL-hex identity currently
  private to `vc::lasagna::Dataset.cpp`; do not introduce an independent hash
  convention.
- Record only identity hex, byte size, schema version, and accounting class in
  the cache sidecar. Do not store raw signed URLs or credentials.
- Verify sidecar identity, destination containment, regular-file state, and
  byte size on a hit.
- Download into a unique temporary sibling, fsync/close through the existing
  file-writing conventions where available, then atomically rename and publish
  the sidecar last.
- On failure, remove the temporary file and leave any previously valid cache
  entry intact.
- Coalesce concurrent in-process requests keyed by normalized destination and
  source identity. Cross-process first downloads may duplicate work, but
  atomic publication must prevent partial/corrupt results.
- Managed arbitrary payloads use the existing persistent cache budget
  reservation/read-pin mechanism. The budget scanner will recognize generic
  payload sidecars after restart. Lasagna/Open Data manifests are small control
  metadata and will be marked unmanaged so their cached path remains stable.

Renaming `PersistentZarrCacheBudget` is not required for this task. Its
implementation already accounts for remote volume and Lasagna data; only its
generic sidecar recognition is extended.

### 2.4 Lasagna cache layout compatibility

For a direct remote manifest, compute the same artifact directory currently
used for its remote Zarr groups:

```text
<remote-cache-root>/remote_lasagna/url_hex/<segmented-normalized-manifest-url>/
```

Materialize a reserved manifest filename and `lasagna-remote.json` inside that
directory. Relative Zarr keys therefore continue to use the same directory in
which existing warm metadata/chunks already reside.

The generic cache accepts a caller-selected destination specifically so this
layout and the existing Open Data layout can be retained.

### 2.5 Remote-origin compatibility

Support three manifest-origin cases through one resolver without collapsing
their persisted formats:

1. A plain local manifest resolves relative group Zarr paths from its local
   parent directory.
2. A local/materialized manifest with the existing `lasagna-remote.json`
   sidecar resolves relative groups from the sidecar's `artifact_url`; its
   `manifest_file` must still identify the opened manifest.
3. A manifest opened directly from HTTP/S3 derives its group origin from the
   remote parent URL. Once cached, a generated sidecar records that origin so
   reopening the materialized file has identical semantics.

An absolute remote `groups[*].zarr` locator remains its own origin. Resolution
precedence is therefore: absolute group locator, explicit valid sidecar,
direct manifest origin, then local manifest parent. Relative keys must be
normalized and prevented from escaping their origin.

### 2.6 Derived primary-volume representation

Manifest attachment automatically adds its Zarr data to `VolumePkg::volumes`:

- actual `(Z,Y,X)` Zarr volumes produce ordinary volume entries directly;
- a conceptual multi-channel Lasagna collection stored as multiple actual 3D
  Zarr volumes is flattened to those ordinary entries before it reaches VC3D;
- each group must name exactly one channel and reference an actual 3D array;
  older flat `(C,Z,Y,X)` Lasagna preprocessing/fit intermediates must be
  converted to per-channel 3D OME-Zarr before attachment.

The generic `Volume`, `RemoteVolumeSpec`, project-volume loader, and VC3D remain
strictly 3D and receive no general 4D or channel-grouping behavior. The
Lasagna project adapter and VC3D sampling paths reject actual 4D arrays. Persist a
Lasagna-derived identity that the Lasagna resolver can reconstruct; do not add
a generic `#vc-channel` selector or teach VC3D to parse Lasagna groups.

Every derived entry carries reserved provenance tags for manifest attachment
identity, group name, and channel name/index. Tags also carry the Lasagna
spacing/base-scale relationship needed for correct project volume metadata.
The manifest remains authoritative: reload reconciles derived entries from it,
and detach removes only entries no longer referenced by another manifest or an
independent project attachment.

## 3. Implementation Sequence

### Phase A: Mechanical extraction of shared remote primitives

1. Move the reusable URL normalization, redaction, collision-free path, and
   HTTP/S3 fetch diagnostics needed by arbitrary files out of the private
   Lasagna translation unit into core utility code.
2. Port `LasagnaDataset` to those shared primitives with no cache behavior
   change in this mechanical step.
3. Keep public URL parsing behavior, S3 region resolution, default credential
   loading, timeouts, and diagnostic text compatible.
4. Add focused tests before adding cache-first behavior.

Expected files:

- `volume-cartographer/core/include/vc/core/util/RemoteFileCache.hpp`
- `volume-cartographer/core/src/RemoteFileCache.cpp`
- `volume-cartographer/core/CMakeLists.txt`
- `volume-cartographer/core/src/lasagna/Dataset.cpp`

### Phase B: Implement arbitrary remote-file caching

1. Implement cache hit validation, unique temporary publication, sidecar
   publication, refresh, invalidation, and in-process single-flight behavior.
2. Reject absolute/escaping caller destinations before any file operation.
3. Add managed/unmanaged accounting and update
   `PersistentZarrCacheBudget` scanning so managed generic payloads survive
   process restart accounting and can be evicted without deleting sidecars.
4. Ensure a sidecar whose payload was evicted is a normal cache miss.
5. Ensure a failed refresh retains the previous valid file.
6. Add injectable fetcher tests using arbitrary binary data, including NUL
   bytes and non-JSON extensions. No test may use the public network.

Expected tests/files:

- new `volume-cartographer/core/test/test_remote_file_cache.cpp`
- `volume-cartographer/core/test/test_persistent_zarr_cache_budget.cpp`
- `volume-cartographer/core/test/CMakeLists.txt`

### Phase C: Materialize remote Lasagna manifests through the cache

1. Extend `LasagnaDatasetOpenOptions` with optional resolved remote auth,
   cache policy, and the test/custom fetch hook needed by the generic cache.
2. Add a shared Lasagna manifest-materialization function returning the local
   cached path plus hit/download information.
3. Change `LasagnaDataset::openLocation()` to:
   - bypass caching for local paths;
   - cache/materialize remote files;
   - write/validate the Lasagna remote marker atomically;
   - reopen the cached file through the ordinary local parser;
   - restore the remote manifest identity and parent URL in runtime metadata.
4. Preserve `LasagnaDataset::open()` support for existing
   `lasagna-remote.json` sidecars. Keep the sidecar `artifact_url` authoritative
   rather than inferring an origin from its local cached path.
5. Centralize group origin resolution using section 2.5 for existing sidecars,
   direct remote manifests, absolute remote groups, and plain local manifests.
6. On cached parse/marker failure, invalidate and refetch exactly once. Do not
   invalidate a valid pre-existing explicit sidecar.
7. Preserve absolute remote group handling and the existing exact-byte Zarr
   read-through store.
8. Keep `vc_fiber_trace_metric` CLI syntax unchanged. Both its fiber and normal
   `openLocation()` calls inherit the cache behavior.
9. Add cache hit/miss diagnostics that do not expose signed query strings.

Tests in `test_lasagna_manifest` will cover first fetch, second-open cache hit,
forced refresh, corrupt/truncated cache recovery, marker semantics, relative
remote group resolution, query redaction, and unchanged local opening. Test the
same relative group layout separately through an existing sidecar-backed
manifest and a direct remote manifest, plus absolute remote groups and path
traversal rejection.

### Phase D: Port existing Open Data Lasagna manifest caching

1. Keep Open Data's artifact listing/discovery, outer metadata validation,
   coordinate tags, and existing deterministic cache directory.
2. Replace its private manifest download/temporary publication with the new
   generic file cacher once the concrete manifest URL is known.
3. Preserve its existing marker and validated fast path so already cached Open
   Data manifests remain reusable without migration or download.
4. Keep Zarr descriptor/chunk validation behavior unchanged.

The Open Data catalog's top-level `metadata.json` live-refresh cache is not
ported in this task: it intentionally has network-first refresh/fallback UI
semantics rather than immutable cache-first object semantics. This distinction
will be documented rather than silently treated as generic-cache debt.

### Phase E: Canonicalize project Lasagna entries

1. Add core tag helpers:
   - reserved tag constant;
   - `isFiberLasagnaEntry`;
   - filtered ordinary/fiber entry access;
   - role reconciliation for an existing location.
2. Replace the in-memory independent fiber entry collection with tagged
   Lasagna entries.
3. During `fromJson()`:
   - load canonical `lasagna_datasets`;
   - load legacy `fiber_inference_datasets` when present;
   - add the reserved tag;
   - merge duplicates without discarding user/Open Data tags;
   - reconcile selected fields.
4. During `toJson()`, emit only canonical tagged entries while retaining both
   selection fields.
5. Make attachment identity comparison handle equivalent local paths and the
   existing normalized S3/HTTPS identities rather than raw-string duplicates.
6. Reclassification must be one persisted mutation. Move the role tag, update
   the selected field for the new role, and clear the old role selection when
   it names the reclassified entry.
7. Extend removal/detach behavior and regression tests.

Tests in `test_volume_pkg` will cover canonical round trips, legacy migration,
tag preservation, duplicate merge, both selections, reclassification, removal,
relative local paths, and remote locator persistence.

### Phase F: Prepare ordinary 3D project volumes from Lasagna data

1. Add a Lasagna-owned preparation API that resolves a manifest and returns a
   flat list of prepared 3D project volumes. Its public result contains no
   channel-group or 4D concepts needed by VC3D.
2. Open actual 3D group Zarr volumes through the existing local/remote 3D
   volume path and preserve its normal cache behavior.
3. Require every attachable group to name exactly one channel and reference an
   actual 3D ZYX array. Reject flat CZYX intermediates with a clear conversion
   error rather than projecting them inside VC3D.
4. Add only the minimum generic 3D factory needed to construct a `Volume` from
   an already prepared 3D chunked source, if the current API cannot accept the
   Lasagna source. That factory must enforce a 3D shape and contain no
   channel-axis or Lasagna logic.
5. Derive canonical primary-volume attachment records from the flat prepared
   results. Validate channel names/counts inside Lasagna before returning them.
6. Add reserved provenance/display tags containing a stable manifest identity,
   group, channel, and Lasagna scale metadata. Keep original user/Open Data
   tags when the same backing view is reconciled.
7. Add a `VolumePkg` batch attachment mutation accepting one prepared Lasagna
   entry plus all prepared `Volume` objects. Preflight duplicate identities and
   conflicts, insert everything, update selections/cache root, and persist
   once. Roll back the complete in-memory and autosave state on failure.
8. Add reference-aware detach/reconciliation: removing a manifest removes its
   derived entries only when no other manifest provenance or independent
   attachment remains; loading a project repairs missing/stale derived entries
   from the authoritative manifest where resolution is available.

Expected focused coverage includes Lasagna 3D-volume preparation and CZYX
rejection tests plus extensions to `test_volume_pkg` for batch atomicity,
shared references, reload reconciliation, and detach behavior. Generic volume
tests must continue to reject actual 4D Zarr arrays.

### Phase G: Add shared VC3D remote-data attachment plumbing

1. Extract project-wide remote cache-root selection and AWS credential
   resolution from `VolumeAttachmentController` into a shared VC3D helper.
2. Port volume attachment to the helper before using it for Lasagna, avoiding
   a copied credential/cache prompt implementation.
3. Add a small Lasagna project attachment service/controller that:
   - accepts a local or remote locator and role;
   - opens/materializes and validates it off the GUI thread for remote input;
   - resolves and prepares every ordinary project volume derived from the
     manifest using Phase F;
   - validates ordinary manifests as usable Lasagna normal data;
   - validates fiber manifests through the existing fiber prediction scale and
     `presence/nx/ny` contract helpers;
   - commits the entry, tag, selection, cache root, and all prepared primary
     volumes through the atomic batch mutation only after validation;
   - returns structured duplicate/reclassification/error outcomes.
4. Keep parsing/validation logic in core libraries; the controller owns only
   Qt scheduling and presentation.

Expected files include new focused VC3D attachment/helper files plus updates
to `VolumeAttachmentController`, `CMakeLists.txt`, and project refresh paths.

### Phase H: Add VC3D menu actions and project resolution

1. Add File menu actions:
   - `Attach Lasagna Manifest...`
   - `Attach Remote Lasagna Manifest...`
2. Local action uses a manifest file picker. Remote action accepts only a
   supported remote file locator through the existing browser/auth UI.
3. Prompt for `Regular Lasagna` or `Fiber inference`, defaulting to regular.
4. Disable the action while its background attachment is in flight and report
   status/errors on the GUI thread.
5. Persist the portable locator, selected role, tag, newly chosen project cache
   root, and automatically derived ordinary volume entries, then refresh
   project UI.
6. Add ordinary/fiber Lasagna rows to Detach and label their role. Detaching a
   manifest also reconciles its automatically attached project volumes.
7. Introduce one locator-aware project Lasagna resolver used by VC3D callers.
   It returns a materialized local manifest path/dataset for local, Open Data,
   or direct remote entries.
8. Replace direct local-only selection assumptions in VC3D, including the
   Line Annotation dataset selection plumbing, with this resolver. Do not
   change tracing behavior or add tracer controls.
9. Fiber selection must filter tagged Lasagna entries; ordinary selection must
   ignore them. Exactly one matching entry may still auto-select as today.

A focused noninteractive Qt test will exercise action presence, role mapping,
successful local attachment, structured remote completion with a fake fetcher,
failed-attachment rollback, and detach visibility. Heavy interactive dialogs
will not be driven in the unit test.

### Phase I: Compatibility audit

Search all consumers of:

- `lasagnaDatasetEntries()`;
- `fiberInferenceDatasetEntries()`;
- `selectedLasagnaDatasetPath()`;
- `selectedFiberInferenceDatasetPath()`;
- `LasagnaDataset::open()` where the source may now be a project locator.

Update VC3D and relevant native CLI call sites to use canonical role helpers or
the shared materializer. Remove obsolete fiber collection APIs only after all
callers and tests are migrated.

Project-based atlas CLIs that have no remote-cache option will continue to
support local selected manifests. Direct remote selected-manifest support for
those separate atlas CLIs is outside this task and must be recorded in
`task_log.md` if the audit confirms they cannot safely inherit a cache root.

## 4. Testing And Validation

### 4.1 Automated tests

Build focused targets without installing dependencies:

```bash
cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target \
  test_remote_file_cache test_persistent_zarr_cache_budget \
  test_lasagna_project_volumes \
  test_lasagna_manifest test_volume_pkg test_open_data_manifest \
  test_vc3d_lasagna_attachment vc_fiber_trace_metric VC3D -j 8
```

Run focused tests:

```bash
ctest --test-dir volume-cartographer/build/ci-tests-clang-systemdeps \
  --output-on-failure \
  -R '^(test_remote_file_cache|test_persistent_zarr_cache_budget|test_lasagna_project_volumes|test_lasagna_manifest|test_volume_pkg|test_open_data_manifest|test_vc3d_lasagna_attachment)$'
```

If the named build directory is unavailable or does not include Qt/VC3D, use
an already configured test-enabled build directory discovered from its
`CMakeCache.txt`; do not bootstrap dependencies or create a new environment
without user approval. Record the exact replacement commands in
`task_log.md`.

### 4.2 Required cache assertions

- A fake remote fetch counter is `1` after two cache-first opens.
- Restart-style reconstruction of the cache object still hits the file.
- Refresh increments the counter and atomically replaces the bytes.
- Invalid size/sidecar/payload causes one refetch.
- Concurrent same-file requests publish one complete result.
- Failed initial fetch leaves no hit; failed refresh preserves the old hit.
- Managed file accounting and eviction behave correctly; unmanaged Lasagna
  manifest control files remain present.
- S3 and HTTP locator normalization select stable identities without storing
  secrets.

### 4.3 Required project assertions

- Untagged entry is regular; tagged entry is fiber.
- Legacy fiber arrays load as tagged canonical entries.
- Saved JSON contains canonical `lasagna_datasets` and no independent legacy
  array.
- Duplicate/reclassification behavior preserves unrelated tags.
- Both selected fields survive save/load and are cleared correctly on role
  changes/removal.
- Remote project JSON retains the remote locator, never the cache path.
- Each actual 3D Zarr volume referenced by Lasagna automatically creates one
  ordinary project volume entry.
- Conceptual Lasagna grouping is flattened before VC3D receives the prepared
  3D volumes.
- Actual 4D Lasagna backing arrays are rejected by project attachment and the
  VC3D sampler; older flat intermediates require conversion to per-channel 3D
  OME-Zarr.
- Generic project-volume loading continues to reject actual 4D Zarr inputs and
  contains no Lasagna grouping/channel selector.
- Automatic volume entries retain manifest/group/channel provenance and
  canonical local or remote view locators across save/load.
- Batch attachment is all-or-nothing when any group/channel fails preparation,
  conflicts by volume ID, or project persistence throws.
- Detach removes unshared derived volumes and preserves shared or independently
  attached volumes.

### 4.4 Required VC3D/CLI assertions

- VC3D builds with both actions present.
- A local manifest can be attached for either role.
- A fake remote manifest attachment performs one fetch and succeeds from cache
  on repeat.
- Invalid manifests do not mutate the project.
- Local, sidecar-backed remote, and direct-remote manifests all automatically
  attach their resolved ordinary project volumes.
- `vc_fiber_trace_metric --help`/usage remains compatible.
- A fixture-level `openLocation()` sequence representing both CLI manifests
  proves their second opens do not fetch again.

No real S3 workload or interactive native fiber trace is required in this
task. Those are follow-up usage tests.

### 4.5 Broader regression

After focused tests pass, run the full available Volume Cartographer core test
suite in the selected build directory if its expected runtime is reasonable.
At minimum run all tests whose names contain `lasagna`, `volume_pkg`,
`open_data`, `remote`, or `cache`. Record failures and whether they pre-existed.

## 5. Spec Update

Update `planning/specs.md` with a new VC3D/remote-data section specifying:

1. Exact-byte arbitrary single-file cache behavior, supported built-in
   transports, custom fetcher extensibility, cache-first/refresh semantics,
   atomicity, identity validation, authentication secrecy, and disk accounting.
2. Remote Lasagna manifests are persistently materialized and reused by both
   `LasagnaDataset::openLocation()` callers and `vc_fiber_trace_metric`.
3. Both explicit `lasagna-remote.json` artifact origins and implicit direct
   remote-manifest parent origins remain supported, with deterministic
   relative/absolute group resolution and the existing exact-byte Zarr cache
   layout.
4. Canonical `lasagna_datasets` project entries and the
   `vc-lasagna-fiber` role tag, including default untagged regular semantics
   and legacy migration.
5. VC3D local/remote attachment actions and validation-before-commit behavior.
6. Automatic ordinary 3D project-volume creation for Lasagna data. Each group
   names one channel in one actual ZYX array; CZYX intermediates are rejected.
   Generic `Volume`, remote volume loading, and VC3D remain 3D-only. Include
   provenance, atomic rollback, and detach rules.
7. This task does not change native tracing numerics or interaction behavior.

Do not remove or weaken the existing remote manifest, scale derivation,
determinism, or exact-value requirements.

## 6. Documentation Updates

1. Add `volume-cartographer/docs/remote_file_cache.md` documenting public API,
   built-in transports, cache layout/sidecars, cache policies, authentication,
   accounting classes, and single-file rather than directory semantics.
2. Add `volume-cartographer/docs/vc3d_project_files.md` documenting project
   Lasagna entries, the reserved fiber tag, selected fields, legacy migration,
   local/remote examples, cache-root behavior, derived ordinary volume entries,
   provenance tags, Lasagna-derived 3D identities, and detach/reconciliation
   rules.
3. Update
   `vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md`
   so native fiber inference documents the tagged VC3D project integration and
   shared cached manifest path.
4. Update any existing Open Data Lasagna comments/docs whose ownership moves
   to the generic cache.

## 7. Changelog And Task Records

On completion:

- add a dated entry to `planning/changelog.md` covering generic cached remote
  files and VC3D tagged Lasagna attachment;
- keep `planning/status.md` current as phases complete;
- replace/update `planning/task_log.md` with implementation findings,
  deviations, exact validation commands, and results.

## 8. Risks, Limitations, And Review Checks

- **Stale mutable URLs:** cache-first intentionally treats a locator as stable.
  Refresh/invalidation exists, but automatic TTL/conditional GET is not part of
  this task.
- **Single files only:** prefix/directory recursion is not part of the generic
  cache. Zarr continues to use its object store.
- **Cross-process duplication:** atomic publication prevents corruption, but
  two processes may both perform the first fetch. Cross-process locking is not
  required unless implementation evidence shows it is necessary.
- **Old VC3D writers:** newly saved canonical tagged projects may not expose
  fiber inference entries correctly in older builds that only understand the
  legacy array. Current code must read old projects; reverse compatibility is
  documented rather than achieved by writing duplicate authorities.
- **Remote atlas shell data:** caching a manifest does not make arbitrary
  manifest-relative directories such as `init_shell_dir` remotely listable.
  This task supports file manifests and their existing remote Zarr groups, not
  remote atlas shell directory materialization.
- **GUI responsiveness:** all remote fetch/parse/descriptor checks must run off
  the GUI thread.
- **3D/storage boundary:** project primary volumes, generic remote volume
  loading, and VC3D remain strictly 3D. Older CZYX Lasagna preprocessing/fit
  intermediates require explicit conversion to per-channel 3D OME-Zarr. No
  general 4D `Volume` support or generic channel selector is added.
- **Derived-entry drift:** project reload and detach must reconcile derived
  volume entries from the manifest/provenance identity. The manifest is the
  authority; duplicated raw origin URLs must not become a second authority.
- **Numerics:** cached bytes and manifest interpretation must remain exact; no
  volume-cache compression/quantization path may be used.

Independent review must verify this plan against `task.md`, `specs.md`, and
`plan.md`, with special attention to cache-layout compatibility, migration
semantics, failure atomicity, and the explicit follow-up boundary for
interactive Line Annotation testing.
