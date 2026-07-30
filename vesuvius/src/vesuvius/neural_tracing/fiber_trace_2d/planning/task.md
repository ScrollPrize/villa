# VC3D Lasagna Manifest Attachment And Generic Remote File Cache

## User request

VC3D projects must be able to attach local and remote Lasagna manifests from
the GUI. Attaching a manifest must automatically attach the Zarr volumes it
references as ordinary primary project volumes. Remote Lasagna manifests must
be cached persistently so VC3D and the native CLI do not download the manifest
again on every open.

The cache used for this must not be Lasagna-manifest-specific. Implement a
reusable arbitrary remote-file cache which can cache any single file/object
from the supported remote transports. Lasagna manifest handling is its first
consumer.

Project Lasagna entries must distinguish ordinary Lasagna data from learned
fiber inference data through a tag:

- an untagged Lasagna entry is ordinary Lasagna data;
- a Lasagna entry carrying the reserved fiber tag is fiber inference data.

The current separate `fiber_inference_datasets` project representation should
be migrated to the tagged `lasagna_datasets` representation without making
existing project files unloadable.

Add GUI actions for:

- attaching a local Lasagna manifest;
- attaching a remote Lasagna manifest.

Both attachment paths must let the user choose ordinary or fiber data, with
ordinary Lasagna as the default. A successful attachment becomes the selected
dataset for its role.

## Scope for this task

This task covers:

1. A reusable exact-byte arbitrary remote-file cache in VC core.
2. Cached remote Lasagna manifest materialization shared by VC3D, the native
   CLI, and existing Open Data Lasagna preparation where applicable.
3. Canonical tagged Lasagna entries and legacy project migration.
4. Local and remote Lasagna attachment actions in VC3D.
5. Updating existing VC3D Lasagna/fiber dataset resolution to understand the
   tagged entries and materialized remote manifests.
6. Automatically attaching manifest-referenced Zarr data as ordinary project
   volumes while keeping the generic VC volume/runtime contract strictly 3D.
7. Automated unit/integration coverage, VC3D compilation, and native CLI
   compilation/smoke validation.

This task does not change the native fiber tracing algorithm, expose new trace
controls, or evaluate tracing quality. Interactive use of the attached data in
VC3D and the Line Annotation native fiber action is a follow-up task after this
manifest/project plumbing is complete.

## Required behavior

### Generic remote-file cache

- Cache arbitrary individual files/objects, independent of file extension or
  content type.
- Standard transport support must cover `http://`, `https://`, `s3://`, and
  `s3+REGION://`. The cache abstraction must permit a caller-provided fetcher
  so other transports can reuse the same persistence logic.
- Preserve source bytes exactly. Do not decode, recompress, quantize, or
  otherwise transform cached files.
- Default to cache-first behavior: a valid cached file is returned without a
  network request.
- Provide an explicit refresh/invalidation mechanism for mutable remote
  objects.
- Publish downloads atomically and never treat partial temporary files as
  cache hits.
- Coalesce duplicate in-process fetches for the same destination.
- Validate cache identity and recorded size before declaring a hit.
- Use the configured persistent remote-cache root and cooperate with its disk
  accounting for managed payloads. Small control files such as manifests may
  be marked non-evictable metadata.
- Never persist credentials or unredacted signed-query diagnostics in cache
  metadata or logs.

### Lasagna manifest caching

- `LasagnaDataset::openLocation()` must materialize a remote manifest through
  the generic cache and reuse it on subsequent opens.
- Preserve the existing local/materialized manifest plus
  `lasagna-remote.json` format. Its `artifact_url` remains the explicit origin
  for relative group paths and `manifest_file` must still identify the opened
  manifest.
- Also support the simpler direct remote-manifest form used by
  `vc_fiber_trace_metric`: when the manifest origin is HTTP/S3, resolve its
  relative group paths against the remote manifest's parent URL and treat the
  referenced Zarr data as remote.
- Absolute HTTP/S3 group paths remain independent remote origins. Plain local
  manifests without a remote sidecar keep local manifest-relative semantics.
- Preserve the existing `remote_lasagna/url_hex/...` artifact directory
  identity so existing cached Lasagna Zarr metadata/chunks remain warm.
- Place/update the Lasagna remote marker beside the materialized manifest so
  reopening the cached path preserves remote-relative Zarr group semantics.
- A missing, truncated, or invalid cached manifest must be fetched again once;
  persistent invalid remote content must still fail clearly.
- Preserve current S3 region, SigV4/default-credential, public HTTP/S3, and
  detailed remote failure behavior.
- VC3D must be able to pass its resolved AWS credentials into the core loader;
  the CLI must retain its current default credential behavior.
- The existing `vc_fiber_trace_metric` command line remains compatible. Its
  fiber and normal remote manifests both gain cache-first reuse automatically.

### Automatic project-volume attachment

- Treat the Lasagna entry and all project volume entries derived from its
  `groups` object as one attachment transaction.
- Resolve each distinct group Zarr through the local, explicit-sidecar,
  direct-remote-origin, or absolute-remote rules above.
- A Lasagna channel collection is conceptually a multi-channel volume. When it
  is stored as multiple actual 3D `(Z,Y,X)` Zarr volumes, attach those volumes
  through the ordinary project-volume path; VC3D does not need to understand
  their conceptual grouping.
- VC3D-compatible manifests must reference one actual 3D `(Z,Y,X)` array per
  named channel. Flat channel-first `(C,Z,Y,X)` arrays belong to an older
  Lasagna preprocessing/fitting intermediate format and are not attached or
  sampled by VC3D; convert them to per-channel 3D OME-Zarr first. Do not add
  general 4D support or channel selectors to `Volume`, `RemoteVolumeSpec`,
  VC3D, or the generic remote-volume stack.
- Derived volume entries must carry shared provenance tags identifying the
  Lasagna manifest, group, and channel where applicable. Use those tags for
  deduplication, reload reconciliation, display naming, and detach cleanup.
- Prepare and validate every derived volume before mutating the project. Any
  missing/incompatible group, channel-count mismatch, volume-ID conflict, or
  persistence failure rolls back the entire manifest attachment.
- Use the existing prepared-volume attachment path for the final commit so the
  new entries behave like manually attached primary volumes in selectors,
  rendering, caching, and project reload.
- Do not eagerly download all remote chunks. Attachment opens descriptors and
  constructs read-through volume views; chunks remain demand-loaded.
- Detaching a manifest removes volume entries derived only from that manifest.
  Preserve a derived volume still referenced by another attached manifest or
  an independently attached entry.

### Project schema

- `lasagna_datasets` is the canonical project collection for both roles.
- Use one reserved tag constant, initially `vc-lasagna-fiber`, for fiber
  inference entries. No role tag means ordinary Lasagna.
- Preserve `selected_lasagna_dataset` and
  `selected_fiber_inference_dataset` so each role can be selected
  independently.
- On load, merge legacy `fiber_inference_datasets` entries into
  `lasagna_datasets` with the reserved tag. Deduplicate by normalized
  attachment identity and preserve non-role tags.
- Save the canonical tagged representation rather than continuing to write a
  second independent fiber dataset collection.
- Reclassifying an entry must update its role tag and clear/fix stale selected
  role fields atomically.
- Detaching a Lasagna entry must clear either selected role when applicable.

### VC3D attachment

- Add `Attach Lasagna Manifest...` and
  `Attach Remote Lasagna Manifest...` to the File menu.
- Local attachment accepts a manifest file; remote attachment accepts a remote
  file/object locator rather than a directory.
- Ask for the role through an explicit regular/fiber choice, defaulting to
  regular.
- Validate and materialize before mutating the project. Failed attachment must
  leave project JSON and selections unchanged.
- A successful attachment includes all ordinary project volumes derived from
  the manifest; attaching only its JSON entry is incomplete.
- Remote I/O and parsing must not block the GUI thread.
- Persist the original portable remote locator in project JSON, not the
  machine-local cache path.
- Store the chosen remote-cache root on a project that did not already have
  one.
- Show Lasagna entries in the existing Detach UI with their role.

## Correctness constraints

- No numerical behavior or cached Lasagna channel bytes may change.
- Remote-relative manifest paths must resolve against the remote manifest's
  parent, never the local cache directory as an origin.
- Existing `lasagna-remote.json` specifications remain authoritative for their
  local/materialized manifests and must remain loadable.
- Local relative paths remain relative to the project/manifest as currently
  defined.
- No installation/bootstrap commands are part of this task.
- Tests must not require network access or real AWS credentials.
