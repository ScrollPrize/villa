# Object-store inputs for the Spiral service and VC3D workspace

## Scope and dependencies

Companion to [OBJECT_STORE_CLI_PLAN.md](OBJECT_STORE_CLI_PLAN.md). CLI steps
11–12 are complete as of 2026-10-26: cleanup landed in `84757fd9d`, acceptance
and documentation in `cb0e796d8`, and review fixes in `ebc43f070`. Normal
sources are then addressed by array path rather than multiscales index
(`89abe9606`, `b793e0b88`, superseding `743d6d6d2`). Stage 0 commits 1–3 landed
on 2026-10-27 (`0ed53d4ce`, `6221efeba`, `0061bc9b8`); step 4's re-ingest and
CPU startup smoke are recorded in the CLI acceptance notes. Stage 1 is now
implemented; its steps 13–14 record below includes the real-data 1,000-step
service fit, checkpoint and preview acceptance. Stage 2 is implemented except
step 4 (hub crash recovery, owned by `vc_object_store` separately); see its
record at the end.

The CLI acceptance notes record a real-data, single-GPU 1,000-step fit with no
patch/fiber/PCL files, finite losses, complete provenance and matching no-refresh
counts/digest. They also record 19 unrelated pre-existing suite failures; real-data
two-rank fitting and changed-input resume remain unrun (CPU startup/resume tests
cover those contracts). These limits do not replace the service acceptance below.
That CLI run predates the array-path normal addressing and stage 0's store
changes. The later stage 1 service acceptance uses the re-ingested stage 0
store; see its record below for the commands, results and remaining limits.

Port both `spiral-fitting`'s persistent HTTP service and the Spiral workspace in
`volume-cartographer/apps/VC3D`. Deliver in two stages, after a prerequisite
stage 0 that merges `origin/tifxyz-to-obj-store` into `spiral-object-store`:

1. A working service/workspace using frozen object-store inputs, without live
   input updates.
2. Store-backed editing in VC3D, and updating a resident fit at a safe
   boundary to a fixed, already-linked store position. Client and service
   share one authoritative store through the object-store hub. This absorbs
   the earlier separate "shared database over SSH" stage.

These stages port fit **inputs** only. Spiral outputs — checkpoints, previews,
diagnostics, flattened/exported surfaces — stay service-owned files; nothing
writes fit results into the object DB. The only DB writes are detector link
refreshes by the service (at stage 1 startup, and on each stage 2 input update)
and stage 2's input edits from VC3D. Storing fit results as store objects is later work. The proposed launch flag is unrelated to
`lasagna/fit_service.py --object-store-dir` (a content-addressed artifact
directory).

For stage 1, assume the same logical store is available on client and service.
Local paths can differ. Provisioning that store is outside stage 1; do not
invent a transfer protocol. Independent copies must be brought into agreement
externally before an input generation can be accepted, and agreement is judged
by the replica-independent digest. Matching paths or object names do not prove
agreement. Stage 2 replaces the external provisioning with hub replicas.

CT, scroll specification/umbilicus, shell, lasagna normals, external supervision,
packed tracks, outputs and caches retain the CLI's external-input contracts.
The object database does not become a CT location registry. Retain service-owned
paths, authenticated HTTP, SSH service tunneling, immutable preview/checkpoint
artifacts, and local/remote connection behavior.

The final service/workspace supports only store-backed patch, fiber and PCL
inputs. Keep the old non-store paths temporarily as regression references, then
remove them after the corresponding store-backed behavior passes comparison.
They are not a permanent fallback or compatibility mode. Capture reference
fixtures/results before deletion so regression coverage survives the cleanup.

## Current seams to replace or reuse

- `fit_session.py`: `FIT_INPUT_CATALOG`, `SpiralInputPaths`, dataset resolution,
  request validation and autosave metadata still describe filesystem inputs.
- `spiral_service.py`: dataset advertisement, initialize/rebuild, PCL display
  artifacts and editing routes depend on those contracts. Bump `API_VERSION`
  together with `SpiralServiceManager::kApiVersion`; reject incompatible clients.
- `spiral_runtime.py`: `_resolve_session_config()` and `_fit_main()` construct
  `FitContext`; input batches use an iteration boundary and distributed command
  barriers. Reuse lifecycle/barrier infrastructure, not removed reload code.
  Step 11 removed `FitContext.prepare_input_changes`, workspace constructor
  bindings and Run-boundary reload/relink paths; runtime callers still reference
  them. Stage 1 must reject those requests before dispatch; stage 2 needs a new
  candidate-generation prepare/install interface, not a call to the deleted hook.
- `service_editing.py`, `input_workspace.py`, `input_snapshot.py`,
  `workspace_storage.py` and input portions of `service_uploads.py` implement
  file snapshots, uploads, Apply/Commit and conflicts. Audit their callers;
  retain independently useful checkpoint/artifact machinery.
- `object_store_inputs.py`: reuse `load_fit_inputs()`, `InputRequest`,
  `read_snapshot()`, materialization and `input_manifest()` from the CLI port.
  Share validation, selection, linking, conversion and provenance rather than
  building a service-specific adapter.
- VC3D: `SpiralServiceProfile`, `SpiralServiceManager{,Inputs}.cpp`,
  `SpiralPanel`, `SpiralWorkspace`, input drafts/rows, point collection and brush
  editors, tracked fibers, session comparison and preview provenance all need
  auditing. The workspace already shares `CState`'s `ObjectStoreSession` and
  some fiber creation is store-backed; extend that integration.
- Reuse the store's coordinate helpers (`GeometryCoordinates.hpp`:
  `geometryScaleToVolume`, `volumeCoordinates`), revision/annotation APIs
  (`currentRevisions(selection)`, typed snapshot accessors at a geometry
  revision) and session events (`StoreSessionEvent::touches`,
  `ObjectStoreSession::holdForEditing`). UUIDs, stable point/control IDs and
  revision pins replace file aliases and document-array positions.

## Stage 0 — merge `tifxyz-to-obj-store`

Merge `origin/tifxyz-to-obj-store` directly into `spiral-object-store` now,
rather than waiting for it to land on main, and port the CLI adapter onto it.
Stages 1 and 2 then build on the store and VC3D APIs it introduces. If the
branch later lands on main as a squash, the next main merge meets the same
conflicts again; resolve them the same way.

Why it is needed: the branch forked from `39a8cf3a1`, before the CLI port's
store changes, and changes the store underneath the adapter.

The deleted Python bindings have been restored by `25e0c0c02` on
`tifxyz-to-obj-store` (fetched and inspected 2026-10-27). Merge that restoration
rather than implementing another one. The inventory below reflects this tip;
the earlier merge trial used `3f3d2e62a`. Any further reinstatement must
initially match the original implementation, including binding location,
signatures and build dependencies, with only required `surface` API rename
adaptations. Preserve this branch's CLI extensions separately as described
below. Defer binding moves, refactoring and replacement with equivalent APIs
until after the branches converge, to avoid conflicting restorations.

- `8a6c1397b` trims the Python bindings to what main's pre-port Spiral code
  used. It deletes the relationship-detector bindings `refresh_links()`
  calls and drops `vc_object_links` from the module's
  `target_link_libraries`; the C++ detectors remain in
  `object_links/ObjectLinks.hpp`, `src/linking/` and the `vc_object_links`
  app. It also deletes `revision_read_view`, `read_geometry_array`,
  `geometry_point_annotations`, `fork_replica`, `relationships_at` and
  `revision_read_view_at`, although their C++ API (`revisionSelectionAt`,
  `relationshipsAt`, `forkReplica`) remains. `25e0c0c02` restores all of these
  bindings, the renamed surface detectors in `_native`, and the
  `vc_object_links` link dependency. No separate links module is needed.
- `4e9ed190c` renames object kind `patch` to `surface`, with the detectors
  (`linkPointCollectionsToSurfaces`, `linkSurfaceOverlaps`,
  `linkFibersToSurfaces`, `linkTracksToSurfaces`), edge kind `point-patch`
  (now `point-surface`) and link metadata (`fiber_surface`, `track_surface`,
  `surface_grid`). It bumps the store schema; older stores are rejected and
  must be re-ingested. This branch is at schema 16, the tifxyz branch at 17.
- Bindings and C++ helpers this branch added are absent there:
  `geometry_coordinates`, `resolve_geometry_coordinates`,
  `geometry_scale_to_frame`, `fiber_hv_classification`,
  `fiber_branch_control_index`, `overlap_correspondences`,
  `geometryScaleToFrame`, `resolveGeometryCoordinates`.
- Restored detectors still expose the pre-CLI-port contract: their keyword
  parsers do not accept `volume_uuid` or lasagna `normal_x_volume` /
  `normal_y_volume`, and their reports omit structured `issues`. Preserve
  this branch's detector implementations and binding extensions during the
  merge, including normal-source handling and failure reporting. Merely
  renaming adapter calls to the restored surface detectors is insufficient:
  `refresh_links()` always supplies `volume_uuid` and may supply lasagna
  normals. The helper and overlap bindings above must also survive the merge.
- It drops the two-store `synchronize` (`995e42a08`) and VC3D's
  `ObjectStoreCoordinates.hpp` and `ObjectStoreEvents.hpp`, replaced by the
  store's `GeometryCoordinates.hpp` (`geometryScaleToVolume`,
  `volumeCoordinates`), `StoreSessionEvent::touches` and
  `ObjectStoreSession::holdForEditing`.
- A trial `git merge-tree` of `b793e0b88` with `3f3d2e62a` conflicts in 30
  files (28 content, 2 modify/delete: `PatchInput.cpp`,
  `test_object_links.py`), mainly `python/objects.cpp`, `src/linking/*`,
  ingest and VC3D's `FiberCollectionController`/`LineAnnotationController`.
  This is historical, not a conflict count for `25e0c0c02`; repeat the trial
  against the execution tip. The deleted Python link test file is restored.

Commits (push them together; the adapter is broken between 1 and 3):

1. **Merge.** Merge `origin/tifxyz-to-obj-store`, including `25e0c0c02`'s binding
   restoration. Take its API (the `surface` rename, VC3D session
   and coordinate changes) and port
   this branch's store changes onto it: the linker and overlap-correspondence
   work, ingest repairs and frame declarations, fiber H/V and branch-control
   helpers, coordinate helpers. Reconcile the schema version history. Store
   C++ tests and the VC3D build pass.
2. **Store: verify restored bindings and preserve CLI extensions.** Use the
   incoming restoration and retain its `vc_object_links` link dependency.
   Preserve this branch's detector keyword extensions (`volume_uuid`, lasagna
   normals) and structured `issues` reports under the surface names. Verify the
   revision read view, geometry-array and point-annotation reads,
   `relationships_at`/`revision_read_view_at` (stage 2 needs them), and retain
   this branch's coordinate/fiber helpers and `overlap_correspondences`.
   Do not replace restored bindings with `read_geometry_arrays` or
   `geometry_snapshot` equivalents in this stage; that can be a later cleanup.
   Run Python tests in `vc_object_store`, including restored detector tests
   and this branch's frame, normal-source and failure-reporting regressions.
3. **Adapter: the `surface` contract.** Object kind `surface`, edge kind
   `point-surface` (the edge-family table in `object_store_inputs.py` keys
   on `patch`/`point-patch`), link metadata `fiber_surface`, `track_surface`
   and `surface_grid` (read today as `fiber_patch` and `patch_grid`), and the
   renamed detectors and their `surface_uuids` keyword (formerly
   `patch_uuids`). Spiral's own role names (`pcl_patch`,
   `patch_overlap`) may stay. Run the CLI adapter tests with the built
   module, so they run rather than skip.
4. **Re-ingest and CLI acceptance.** Re-ingest the acceptance dataset with
   `vc_object_ingest`, using the README's corrected frames (patches/PCLs at
   scale 1, version-3 fibers at scale 4) and the documented point-ID and
   dangling-branch repairs. Rerun a short CLI fit with link refresh and
   confirm finite losses and consistent link counts and digest across a
   no-refresh rerun. Record commands and results in the CLI acceptance notes.
   This is the gate for stage 1.

## Stage 1 — frozen-input service and workspace

### 1.1 Bind the store and define the protocol

- Add an explicit service launch binding for the input DB (proposed
  `--object-store`), alongside the existing dataset/external-input root and
  output/cache bindings. Resolve the service path once; arbitrary client paths
  must not override it through advanced config or checkpoint configuration.
  `FitContext.load_host_inputs()` reads `config['object_store_path']`, and
  `_resolve_session_config()` merges the checkpoint's `cfg` and then request
  overrides, either of which can carry that key. Set it from the launch binding
  after that merge, and reject it in request overrides.
- Advertise the binding, supported capabilities, selectable objects and frame
  metadata through dataset discovery without loading Torch or refreshing links.
  The client supplies its local attachment separately. A local auto-launch
  profile can use its attached DB; a remote profile binds to the advertised
  store without assuming that a client path is valid on the service host.
- Use the actual CLI config names: `object_store_{patch,pcl,fiber}_{uuids,tags}`,
  refresh/approved-only options, `fit_volume_uuid`, normal-source settings and
  external frame settings. Preserve omitted versus explicitly empty selectors.
  Display the resolved UUID selection and counts, including excluded links.
  Reuse the adapter's normal-source resolution (`fiber_normal_source()`): the
  lasagna `normal_zarr_group` and `object_store_normal_ct_level` both name zarr
  groups, passed to the store's CT reader as `<store>/<group>` at level `0`,
  without consulting multiscales metadata. Do not duplicate that resolver in
  the service.
- Represent catalog entries by object UUID, kind, revision pins, display label,
  frame and metadata-derived PCL role. Do not collapse all collections of a role
  into one editable file. Relationship revisions/approval and materialization
  policy belong in the input provenance too.
- Advertise the active input manifest/digest in status and artifact provenance.
  Use the CLI manifest format and path-independent digest; compare the selected
  inputs, not just filenames or a nominal store identity. Object revision
  UUIDs are per replica, so the digest and the identities VC3D compares must
  be replica-independent (implementation step 2). VC3D displays its attached
  store as it is now (the store is edited live) and says which fitted inputs
  differ from what the fit read; it compares identities, never by
  re-serializing and hashing on the client. A different store (UUID
  mismatch) is refused outright. Preview viewing may remain available with an
  explicit mismatch indication.

### 1.2 Initialize, run, rebuild and resume through the CLI adapter

- Thread the bound DB and resolved input config through the service runtime into
  `FitContext`. Remove filesystem patch/fiber/PCL discovery and input bindings
  from this path. Keep CUDA startup checks and existing device selection.
- Run the shared startup lifecycle once per new input generation: validate,
  resolve selection, refresh on rank zero by default, aggregate failures,
  snapshot, materialize and compare rank manifests before training.
  No-refresh and approved-only have exactly the CLI semantics. Show structured
  refresh failures and any already-committed detector writes in service events.
  When the service's store is a separate copy, a service-side refresh makes the
  copies diverge. Such profiles run with no refresh, against links prepared
  before the copy, until stage 2 pushes service links through the hub.
- Freeze inputs for the resident fit. Run/stop, status, metrics, explicit and
  scheduled previews, checkpoint save/download and reconnection still work.
  “No live updates” means no changed fit inputs, not disabling these features.
- Reject input-changing requests and config changes while resident; expose a
  clear full-input rebuild operation. A model-only rebuild must retain the
  frozen inputs and must not rerun refresh. Do not map removed file-reload verbs
  onto hidden database reads. Classify `config.OBJECT_STORE_PCL_KEYS` and
  `input_use_pcl_relative`/`input_use_pcl_same_winding` as full-input rebuild
  changes: `FitContext.apply_config()` refuses them, including when called by
  a model-only rebuild. Preserve this distinction in profile comparison and UI
  controls; these settings cannot be applied at a Run boundary.
- Keep the CLI's no-concurrent-input-edit contract during startup and fitting in
  this stage. Disable Spiral input mutations while a session is resident and
  document that other clients/editors must also refrain from writing. A local
  UI guard is not a distributed lock. Unexpected changes make local input
  displays stale; they never silently replace resident constraints.
- Apply shared manifest/frame checks to checkpoint load, autosave recovery and
  rebuild. A mismatching or legacy checkpoint must produce a precise refusal
  or the CLI's explicit changed-input policy, never implicit acceptance.
  Preserve distributed all-rank preflight and refusal behavior.

### 1.3 Port the workspace's input display and controls

- Replace file-based rows and overlay loading with store reads of the fitted
  objects at their current heads. Use the shared coordinate helpers: stored XYZ divided by
  the object's scale gives reference-volume XYZ; convert to the preview/viewer
  frame explicitly. Preserve invalid patch vertices and stable annotation IDs.
- Show service connection, local store binding, selection, active manifest,
  resolved volume and input freshness. Require matching volume declarations;
  do not use dataset-root path substitution as an object identity mechanism.
- Retain read-only patch/fiber/PCL overlays and fit previews. The overlays
  follow the store: they show the fitted objects as the store holds them now
  and update the inputs a commit touches, not the whole set. The resident fit does
  not follow; the freshness notes say what differs and a rebuild reads it.
  Showing old revisions of an evolving store is not a goal: the main VC3D
  views show the store's current state, and the Spiral overlays must agree
  with them.
- Disable or remove the old file draft upload, Apply/Commit, brush-to-file,
  PCL replacement and tracked-fiber upload actions. Explain that input changes
  require ending/rebuilding the session until stage 2 is available. Preserve
  unfinished local drafts during profile migration; do not auto-import them.
- Update saved settings, config profile editing, initialize/rebuild comparison,
  stale preview handling and reconnect synchronization. Old settings should
  request an explicit store binding rather than infer one from filenames.
- Delete obsolete service file-input code and tests once callers have moved
  and the stage 1 regression comparisons below pass. Retain captured reference
  fixtures and store-backed regression tests, not the old implementations.
  Do not restore anything removed by CLI step 11 just to keep old APIs alive.

### Stage 1 acceptance

- Before deleting the old non-store paths, compare frozen-input service and
  workspace behavior against a recorded pre-port baseline. The current service
  cannot start a fit, so the baseline comes from a checkout before `b795e458f`
  with file inputs equivalent to the store ingest. Limit it to what is
  comparable across the CLI linking changes: workspace display
  geometry/coordinates and input-agnostic service lifecycle/protocol behavior
  (initialization, run/stop, previews, checkpoints, reconnect and resume). Fit
  results are not compared against file inputs; fit equivalence is the
  CLI-versus-service digest check below. Record commands, fixtures and results.
  Remove the replaced paths, obsolete routes, options and tests after these
  comparisons pass, then rerun store-backed acceptance with no legacy inputs
  available.
- Service discovery without the fit runtime; clear missing-store/API mismatch
  errors; local and remote profiles with different paths to equivalent inputs,
  including a replica rather than a byte copy (the digests match).
- Same selection/config produces the same materialized constraints and input
  digest through CLI and service. Test empty selectors, metadata winding roles,
  non-unit object/external scales, rejected/stale links and approved-only mode.
- Initialize → run → stop → preview → save → reconnect → resume works without
  patch directories or fiber/PCL JSONs. Input mutation routes are rejected and
  controls disabled; model-only rebuild retains its input digest.
- Two-rank initialization refreshes once, propagates failures, checks manifests
  and cannot start partially. Resume rejects changed revisions/frames.
- Run a short fit on an ingested real dataset and inspect it in VC3D. Record
  exact commands, config, revisions, counts, finite losses and checkpoint/preview
  provenance. This is a gate before starting stage 2, not merely adapter tests.

### Stage 1 implementation sequence

Planned as one commit each on `spiral-object-store`, after stage 0. Step 2
changes only the shared adapter and can land first. Steps 3–11 change the service/VC3D protocol and land together (one
`API_VERSION` / `kApiVersion` bump, in step 4 and step 9); step 13 deletes old
code only after step 12's comparisons pass. Keep service and VC3D commits
separate and include focused regression tests with each functional commit.

Findings from the current tree that the steps build on (HEAD `b793e0b88`;
recheck them after stage 0, which changes the store and VC3D code they cite):

- Every service build fails: `spiral_runtime._fit_main` passes
  `input_bindings=` to `FitContext`, which now takes `object_store_inputs=`.
  The bindings come from `EditingWorkspace.build_bindings()` via
  `ServiceState._begin_build` and thread through `create_session`,
  `InteractiveFitSession`, `DistributedInteractiveFitSession` and
  `_distributed_session_worker`. No test drives the real `_fit_main` path
  without a GPU; `test_interactive_e2e.py` (env-gated) would fail.
- `object_store_path` is an ordinary config key, advertised by
  `/configuration` and settable through `run.config` or `--config`.
- `validate_session_request` still requires the `verified_patches` directory
  and PCL files from `FIT_INPUT_CATALOG`/`PCL_ROLE_CONVENTIONS`, which the
  fitter no longer reads.
- The input-editing workspace is also the service's global mutation gate
  (`_dispatch_active`: every non-GET needs a claimed workspace token) and its
  teardown owner. Claiming snapshots dataset patches, PCLs and fibers. Those
  two roles need a non-input replacement before the workspace goes.
- Torch is imported at service start: `spiral_service.py` imports
  `checkpoint_migrations`, which imports torch at module level.
- Object revision UUIDs are per replica; geometry revision UUIDs and the store
  UUID are shared (`vc_object_store/docs/README.md`, "Hosted hub and
  replicas"). The manifest digest includes every object's `object_revision`
  (`StoredObject.provenance`), so identical inputs on two replicas digest
  differently. Only a byte copy of a store reproduces them.
- Only checkpoints record the object-store manifest/digest
  (`_checkpoint_input_manifest`). Status `input_manifest` lists file paths;
  autosave metadata, preview manifests, `PublishedPreviewIndex` and PCL
  artifact descriptors record none; `_write_object_store_manifest` runs only
  on the headless path.
- `_checkpoint_refusal` derives the offered rebuild stage from the cfg diff
  alone, so a manifest mismatch with an unchanged cfg is offered a "model"
  rebuild that cannot fix it.
- Service PCL display artifacts copy dataset PCL files
  (`_publish_pcl_artifact_locked`). VC3D loads input patches as local tifxyz
  through `mapServicePath` (dataset-root substitution) and keys them, and
  input identities in `logicalInputId`/`addPatchToCurrentFit`, by directory
  name. Fibers are exported from the store at current heads
  (`LineAnnotationController::fiberTransportPaths`) and uploaded as JSON.
  VC3D has no pinned patch/PCL read helper; the store library provides
  `RevisionSelection`, `readGeometryData`, `exportGeometry(..., revisionUuid)`,
  aligned annotations per geometry revision and `fiberNetworkJson(selection)`.
- Reusable, not input-specific: `service_uploads.UploadManager`'s checkpoint
  path, `input_workspace.MutationCoordinator` and `WorkspaceLease`,
  `workspace_storage`, `input_snapshot.snapshot_input` and
  `input_publication.PublicationTransaction`. Input-specific: `Catalog`,
  `Change`, `Content`, `InputIdentity`, `Revision` and
  `service_editing.EditingWorkspace`.

1. **Baselines: capture pre-port fixtures.** At a checkout before `b795e458f`
   (CLI step 10 made the store mandatory; confirm the service still fits there),
   with file inputs matching the acceptance ingest, record:
   - display geometry: for a fixed set of patches (including invalid
     vertices), PCL collections (with point IDs) and fibers, the viewer-frame
     coordinates the workspace displays. Use a small dump entrypoint that step
     10 reuses against the store path. Stage 0 changes VC3D's coordinate
     code substantially, so make it a standalone script over the workspace's
     inputs rather than a test of VC3D internals, runnable at both checkouts;
   - service lifecycle transcripts for initialize → run → stop → preview →
     save → reconnect → load checkpoint: state/event sequences and the
     preview manifest and checkpoint provenance keys, from one real short
     session and the mocked-runtime `test_spiral_service_v2.py` harness.
   Commit the fixtures under `tests/data/` with a note naming the commit,
   commands and dataset. No production change.
2. **Adapter: replica-independent input identity.** Identify objects in the
   digest by geometry revision plus a hash of the metadata the adapter reads,
   and relationships by content (family, approval, generation, anchors and
   endpoint geometry revisions) rather than by revision UUID. The manifest
   also records the metadata values themselves (`metadata`), so a client
   compares values; hashes are the digest's business, not something a C++
   client recomputes. Verify which
   relationship identifiers the store shares across replicas before
   choosing. Keep the per-replica revision UUIDs in the manifest for local
   resolution, outside `_compared()`. A checkpoint written before this change
   lacks the metadata hashes: compare it on geometry revisions with a warning
   naming the identity change, rather than refusing it. This is shared with
   the CLI. Test that two replicas of a fixture store give one digest, and
   that a metadata-only edit changes it. Make the replicas with `hub_init`
   and two `hub_clone`s: stage 0's store has no two-store `synchronize`
   (`fork_replica` is bound again), and a hub clone is the deployed case.
3. **Service: bind the object store at launch.** Add required `--object-store`
   next to `--cache` (`spiral_service.py` argparse): resolve it absolutely,
   open it read-only once and record its store UUID. Fail startup with a clear
   error if the store or the `vc_object_store` module is unavailable. Pass it
   to `ServiceState`; inject `run.config['object_store_path']` in
   `_prepare_session_request` and `startup_session_request`. Refuse a
   client-supplied `object_store_path` in `run.config`/`--config` as
   `SCROLL_SPEC_OWNED_RUN_KEYS` are refused, and mark it service-owned in
   `/configuration`. Request overrides follow the checkpoint `cfg` in
   `_resolve_session_config`, so the binding wins; test that with a checkpoint
   holding another path, and keep `_reject_overrides_the_checkpoint_contradicts`
   from treating it as a contradiction.
4. **Service: replace the editing workspace with a session lease.** Build a
   lease from `WorkspaceLease`, `MutationCoordinator` and `workspace_storage`
   that keeps the token gate, `editing_lifecycle` command idempotency and
   teardown, but holds no input catalog and snapshots nothing at claim.
   `editing_lifecycle` no longer calls `refresh_clean`/`replay_resident`.
   Input-mutation routes (`input-changes`, `apply-inputs`, `commit-inputs`,
   `resolve-input`, `discard-inputs`, `input-content`, `input-catalog`,
   `input-commands`) answer 409 with a stable `inputs_frozen` code naming the
   full-input rebuild. Uploads accept only `checkpoint`. Bump `API_VERSION`.
   `_begin_build` stops calling `EditingWorkspace.build_bindings()` and
   passes no bindings, so the commit still imports and its tests pass; step 6
   removes the parameter. The old modules stay in the tree, unreferenced,
   until step 13.
5. **Service: torch-free discovery of store inputs.** Drop patches, fibers
   and PCLs from the service's `FIT_INPUT_CATALOG` use, `PCL_ROLE_CONVENTIONS`
   probing, `pcl_inputs` and `paths.pcls`. `validate_session_request` stops
   requiring input files. `/dataset` advertises the binding (store UUID, its
   service-host path as information only) and the startup config's selection.
   Build that from `request_from_config()`, `resolve_selection()` and
   `read_snapshot()` read-only and without refresh, as a structured form of
   `snapshot_report()` shared with it: per object the UUID, kind, revision
   pins, label, frame/scale and metadata PCL role; plus counts, exclusions and
   the fit volume. Make the `checkpoint_migrations` import lazy and test in a
   subprocess that starting the service and `GET /dataset` do not import
   torch.
6. **Runtime: build `FitContext` from the bound store.** Remove
   `input_bindings` from `create_session` through `_fit_main`; `FitContext`
   loads inputs in `load_host_inputs` through `load_fit_inputs()` (rank-zero
   refresh, failure aggregation, manifest agreement). Route the adapter's log
   and `LinkRefreshError`/`ObjectStoreInputError` reasons (including committed
   detector counts from `refresh_report()`) into session progress, events and
   the error state, rather than stdout only. Write `object_store_inputs.json`
   on the interactive path too: `_write_object_store_manifest` and
   `_checkpoint_input_manifest` are `FitContext` methods in `fit_spiral.py`
   that only the headless path calls, so call them from shared lifecycle
   code rather than duplicating them in the runtime. `apply_input_changes`
   (single and distributed) refuses before reserving a boundary. Test on CPU with the
   device stages stubbed against `tests/object_store_fixture.py`: the context
   receives the bound path, a model-only rebuild keeps `object_store_inputs`
   and does not refresh, and a two-rank startup refreshes once, propagates a
   failure and never starts partially (reuse the startup tests' distributed
   harness).
7. **Service: input provenance in status and artifacts.** Replace the status
   `input_manifest` file listing with the context's object-store manifest and
   digest, keeping external paths (shell, lasagna, tracks, checkpoint). Add the
   digest to autosave metadata (`write_autosave_metadata`), preview manifests,
   `PublishedPreviewIndex.record`, and the preview-restore matching in
   `_note_checkpoint_state`/`_restore_published_preview`. Remove the dataset-file
   PCL display artifacts (`_publish_pcl_artifact_locked`, the
   `*_winding_artifact` status keys): VC3D reads PCLs from its attached store
   (step 10), so no service copy remains.
8. **Service: checkpoints, autosave and rebuild against frozen inputs.**
   `_checkpoint_refusal` offers a full-input rebuild, or names
   `object_store_allow_changed_inputs`, for a manifest mismatch, and refuses
   a legacy checkpoint without `input_manifest.object_store` precisely.
   `_rebuild_stage_locked` no longer compares input paths. Add an explicit
   full-input rebuild (`POST /session/rebuild` with `inputs: "reload"`) that
   forces stage "all" to pick up new store revisions under an unchanged
   config. Preserve all-rank preflight. Test the refusal, the forced reload
   and that the `OBJECT_STORE_PCL_KEYS`/role toggles stay rebuild-only.
9. **VC3D: protocol, profiles and store binding.** Bump `kApiVersion`.
   Replace the editing claim/release calls with the session lease, and drop
   `pcl_inputs`, `paths.pcls` and input path rows from `SpiralPanel`'s request
   and `prepareSessionRequest`. Local auto-launch passes `--object-store` with
   the store `CState::objectStoreAttachment()` names (the session is
   multi-attachment after stage 0) and restarts on an attachment change. A
   remote profile binds to the advertised store; both compare the local attachment's store UUID with the advertised one and
   refuse to present local data as the fitted inputs on a mismatch.
   `SpiralReloadComparison` compares the selection config instead of input
   paths and knows the input-reload rebuild. Old saved `form/pcls` and input
   path settings are ignored with a note, not mapped to store objects; brush
   working copies under `spiral-working/` are left on disk. Stage 0 already
   removes the one-time migration of the original global settings.
10. **VC3D: store reads for input overlays.** Add a helper
    (`SpiralStoreInputs`) that reads the manifest's patches, PCL collections
    and fibers from the attached store at their current heads and says, per
    input, whether it is what the fit read: geometry revision UUID (shared
    by replicas) and the adapter's metadata values from step 2, compared as
    JSON values. Relationships the fit consumed are compared with the ones
    the current heads give by record UUID, generation, approval state and
    endpoint geometry revisions, all replica-independent; changed, dropped
    and newly consumable links are reported. Never use the service's object
    revision UUIDs. Convert with the store's `geometryScaleToVolume` from the
    object's current frame, keeping invalid patch vertices and stable point
    IDs. Replace `loadInputSurfaces`' tifxyz reads and the PCL artifact
    overlay, and draw the fibers with a read-only overlay. Key surfaces and
    colours by UUID, not directory name. `mapServicePath` remains only for
    external inputs (shell, lasagna). An object the store no longer holds is
    reported missing and not drawn. Register a `spiral_store_inputs` ctest
    on a fixture store and compare against step 1's coordinates with the
    same dump entrypoint.
11. **VC3D: disable input editing; show bindings and freshness.** Disable
    brush-to-file staging (`finalizeBrushPaint`), PCL replacement and new-PCL
    staging, row Add/Commit/Remove/Restore/conflict actions,
    `addPatchToCurrentFit`, `addFibersToCurrentFit`/`stageFiberInputs` and
    managed patch copies, each explaining that input changes need a
    full-input rebuild until stage 2. Keep the in-memory brush and point
    editors for stage 2. Show service connection, store binding, selection
    counts, manifest digest, resolved volume, and input freshness: store
    events touching the fitted objects or links (`StoreSessionEvent::touches`)
    re-read only the records they name (and a changed object's links) and
    replace only those overlays, coalesced; a full read happens only for a
    new manifest or store binding. Other inputs a rebuild's selection may
    include are counted. Previews show their
    digest; a preview from another digest is labelled as such. Stage 0 moves
    VC3D point collections into the store (`PointCollectionSync`); check
    whether the Spiral point editors already write through it, and if so
    disable them for selected inputs here rather than port them in stage 2.
12. **Acceptance before deletion.** Run step 1's comparisons against the
    store-backed service and workspace and explain expected differences.
    Check that the CLI and the service produce the same digest and
    constraint counts for one config. Cover empty selectors, metadata winding
    roles, non-unit scales, rejected/stale links and approved-only mode in
    service tests. Record commands, fixtures and results here.
13. **Remove obsolete file-input code.** Python: `EditingWorkspace`, the
    input-specific `input_workspace` classes, `service_uploads` input
    validators, the input entries of `FIT_INPUT_CATALOG` and `PCL_ROLE_FILES`,
    the runtime input-batch path (`_input_batches`, reserve/prepare/finish,
    `_run_input_batch`, `_apply_input_changes_at_boundary`), the rejecting
    routes, and their tests (`test_service_editing*.py`,
    `revisioned_client_service.py`, input parts of `test_input_catalog.py`,
    `test_workspace_cleanup.py` and `test_revisioned_runtime.py`). Stage 2's
    candidate generations build on the shared command barriers, not on this
    path. VC3D: the staging/upload half of `SpiralServiceManagerInputs.cpp`,
    `SpiralInputCopy`, input-draft upload state, `inputUploadFinished` and
    `spiral_input_workflow`. Stage 0 already deletes the `check_spiral_*`
    scripts and the unemitted `fiberRevisionUploadFinished`,
    `pclReplacementUploadFinished` and `pclCommitConflict` signals
    (`7a8db3260`). Keep step 1's fixtures,
    `input_snapshot`, `PublicationTransaction` if still used, and the stage 2
    editors. Audit remaining references, then rerun the store-backed tests
    with no legacy inputs present.
14. **Docs and real-data acceptance.** Document service launch with
    `--object-store`, local and remote profiles, selection display, frozen
    inputs and the input-reload rebuild in README.md. Run a short fit through
    the service on the ingested real dataset, inspect it in VC3D, and record
    exact commands, config, revisions, counts, finite losses and
    checkpoint/preview provenance. This is the stage 2 gate.

### Stage 1 acceptance record (step 12, 2026-10-27)

Steps 1-11 landed as `99643ee9c` (baselines), `cc15e5480` (replica digest),
`93a7c1539`, `d2c4958ad`, `4810151fd`, `579b1cac2`, `a2ed6078e`, `684de4926`
(service), `fd407d9ba`, `d3efcb54f`, `efd29f25c` (VC3D). Environment: the
`villa-latest` checkout's `spiral-fitting/.venv` Python with this tree first
on `PYTHONPATH` and a fresh Release `vc_object_store` build of this tree's
store; VC3D RelWithDebInfo configured with `VC_TESTING=ON` (neither existing
VC3D build directory had tests enabled).

- Display geometry (`tests/data/spiral_service_baseline/`). VC3D's pinned
  reader (`SpiralStoreInputs`, the code that draws the overlays) read the
  baseline's nine objects from the stage 0 acceptance store at the pins of a
  manifest the Python adapter wrote:
  `SPIRAL_STORE_INPUTS_DUMP='STORE|MANIFEST|OUT' test_spiral_store_inputs`, then
  `display_geometry_dump.py compare display_geometry_files.json OUT`. Every
  coordinate, invalid and masked vertex count matches the file-backed
  workspace's. The one difference is expected: `same_windings:210` has point
  IDs 7542-7544 in the store and 0-2 in the file, the ingest's
  `--renumber-point-ids` repair.
- Service lifecycle. `service_lifecycle_transcript.py` against the store-backed
  service (`lifecycle_transcript_store.json`) makes the same 14 requests with
  the same HTTP statuses, lifecycle states and event kinds as the pre-port
  transcript. Expected differences: the claim is the session lease (`lease_id`
  added); status loses `commit_available`, `commit_unavailable_reason`,
  `same_winding_artifact` and `relative_winding_artifact` and gains
  `input_manifest_artifact` and `lease_id`; `/dataset` loses `pcl_inputs` and
  gains `object_store`, `inputs` and `capabilities`; the stranger's refusal
  names the session lease.
- CLI versus service (`test_runtime_object_store.py`): one store and
  configuration give the same digest, constraint counts and views through a
  CLI `FitContext` and a service session, with absolute, relative and
  same-winding PCLs by metadata and annotations, a patch and a fiber at scale
  2, approved, pending, rejected and stale links, in approved-only mode, and
  with explicitly empty selectors. The same file runs the service, runtime and
  fitter on the CPU (device stages stubbed): the session reads the bound
  store, reports the adapter's lines and refusals, keeps frozen inputs across
  a model rebuild, rereads them on `"inputs": "reload"`, refuses Run changes
  to input-shaping settings, and two gloo ranks refresh once and both start or
  both fail before Idle.
- Replicas (`test_object_store_startup.py`): two hub clones of one store, after
  a push and a pull that gave them different object and relationship revision
  UUIDs, give one digest.
- VC3D against the real service: `spiral_service_protocol`
  (`SPIRAL_TEST_PYTHON` set) connects `SpiralServiceManager` to
  `spiral_service.py` (runtime faked by `tests/protocol_client_service.py`,
  inputs from a real store): lease, matching store binding, a saved profile's
  `object_store_path` stripped, the input manifest artifact fetched and its
  pins resolved in the attached store, input changes refused as frozen, the
  reload rebuild accepted, and the lease released.
- Suites: the regression command below plus `test_service_object_store.py`,
  `test_runtime_object_store.py`, `test_preview_restore.py` and
  `test_spiral_headless.py`: 426 passed, 1 skipped (the opt-in real-data
  smoke). `ctest -R spiral`: 10 passed, including `spiral_store_inputs` and
  `spiral_service_protocol`.

Not run at step 12, because the GPU was held by another job: stage 0's
1,000-step GPU fit and the real short session of step 1 (preview manifest
and checkpoint provenance keys). Step 14's service fit below covers both
on the same data. The VC3D workspace was built and its logic tested, but not
driven interactively.

### Stage 1 steps 13-14 record (2026-10-27)

Step 13 landed as `58b5aab71` (service: editing workspace, publication,
live input batches and input uploads removed; `MutationCoordinator` and
`ClientLease` moved to `service_mutations.py`) and `7d9eeea69` (VC3D:
staging, working copies, `SpiralInputDraft`/`SpiralInputCopy` and their
test, the workspace's upload/commit bookkeeping, the "pending patches only"
display and the panel's per-input actions removed; the brush and point
editors kept, listed as local drafts). `spiral_service_protocol` no longer
checks a frozen-input refusal through the client, which has no input-change
call left; the service's 404 for the removed routes is tested in
`test_service_object_store.py`. `VC3D --test-object-store` imports a
transported fiber component directly instead of through the workspace.
Suites after step 13: spiral-fitting 935 passed, 77 skipped, 11 failed (the
known CpuTrackStorage/track-crossing tests and `test_grow_track_store`'s
`area_cm2` KeyError, all failing before stage 1); `ctest -R spiral` 9
passed; `VC3D --test-object-store` passed.

Step 14: README "Object store and frozen inputs" and "Connecting from VC3D",
launch commands with `--object-store`, and `volume-cartographer/docs`.
Real-data service fit (one RTX 3090, shared with an inference endpoint),
from `spiral-fitting/`: `spiral_service.py --dataset DIR --object-store
STORE --object-store-author stage1-acceptance --output OUT --api-key-file
KEY`, where DIR holds only `spiral-scroll.json`, `umbilicus.json`,
`lasagna_inputs/`, `outer_shell/` and `fiber_directions.npz` and STORE is a
copy of the stage 0 acceptance store; an HTTP client claimed the lease,
initialized with `{"run": {"z_begin": 8900, "z_end": 9300, "config":
{"optimizer_num_training_steps": 1000, "dense_spacing_mode": "grad_mag",
"loss_weight_dense_spacing_density": 0, "loss_weight_dense_normals": 0,
"loss_weight_dense_spacing": 0}}}`, ran 1,000 iterations, saved a
checkpoint, exported a preview and released the lease.

- Startup refresh: 553 PCL-patch, 6 fiber-patch, 444 overlap links
  created. Selection 181 surfaces, 662 point collections, 111 fibers;
  relationships fiber_branch 100, fiber_patch 6, patch_overlap 444,
  pcl_patch 553; 100 fiber links in 11 components.
- Loss 1747 -> 474 -> 125 -> 38 -> 28 at steps 0/200/400/600/800, 89 at
  1,000, all finite. Build, 1,000 steps, save and published preview: 121 s.
- Digest `1835418770f6...` in status, the `spiral-input-manifest` artifact,
  the checkpoint's `input_manifest.object_store_digest`, the autosave
  metadata and the preview index entry of the published preview (model
  state `c4a9d8aae29f...`, source iteration 1,000, pinned by both the
  checkpoint and the autosave).
- VC3D's pinned reader (`SPIRAL_STORE_INPUTS_DUMP`) resolved all 954
  objects of the published input manifest in the store at their pins.
- The same fit on z 9000-9100 trained (finite losses, 1,000 steps) but its
  preview did not publish: "Lasagna correspondence mapped no preview
  windings". The 21-row surface keeps 12 vertices after
  `_prepare_cleaned_lasagna_surface`'s 3-cell erosion and largest
  component; 400 slices keep 40.5%. Stage 1 did not change preview export
  or Lasagna publication: narrow windows need a fix there, outside this
  plan.
- Review follow-up (2026-10-28): the workspace no longer displays pinned
  revisions. Steps 10-11 were revised: overlays show the fitted objects as
  the attached store holds them now and reload on store commits touching
  them; freshness compares geometry revisions, the manifest's new adapter
  metadata values and relationship generation/approval/endpoint geometry,
  with no client-side hashing. A commit re-reads only the objects and links
  it names (`readFitInputChanges`): on the acceptance store a full read
  takes 0.58 s and one surface plus one link 0.018 s. Fibers are drawn by a read-only overlay
  (`SpiralFiberOverlayController`). The display comparisons above ran on an
  unchanged store, where current and pinned reads coincide; rerun on the
  acceptance store, the reader finds all 954 objects and 1,103 links as
  fitted and its dump is byte-identical to the pinned reader's.

Stage 2 has its own sequence (below), after the stage 1 gate. It depends on
step 2's replica-independent identities, step 4's lease, step 10's store
reads and step 11's freshness tracking. It also depends on the hub
prerequisites (2.1), which can be settled during stage 1.

### Pre-stage-2 integration of `tifxyz-to-obj-store` (2026-10-28)

Merged the incoming branch through `4ea767e44` before starting the store
prerequisites. Its surface-query changes replace VC3D's patch index with
store-backed intersections/proximity queries, add shared decoded-tile caching,
and provide `ObjectStoreSession::setDraft`, `updateDraft`, `clearDraft` and
`draftRevision` plus `StoreSessionEvent::Kind::Draft`. These drafts are local
to the session, never committed or synced; they are not new fit generations.
The incoming commits do not implement historical hub-position retrieval,
conditional publication, push holds or the hub crash-recovery prerequisite.

The Spiral integration keeps display registration keys separate from object
identity: fitted surfaces retain their store UUIDs and are queried directly,
without replacing a main editor's draft. Service previews, external shells
and local brush surfaces use session-only drafts for queries; Spiral removes
its own drafts on teardown. Preserve the stage 1 service protocol, input
freshness tracking, incremental committed-input reads and fiber overlay.
The removed intersection-layer and file-input-draft tests are not restored.

Merge validation on Linux:

- Rebuilt the native store with `cmake --build /tmp/vc-object-coordinate-build -j 6`
  and VC3D plus the relevant test targets in the existing RelWithDebInfo build.
- `ctest --test-dir /tmp/vc-object-coordinate-build --output-on-failure`:
  all 15 tests passed, including session surface drafts and hub tests.
- The 13 selected VC3D/core tests passed: store inputs, service protocol,
  brush patches, offscreen object-store controller, config profile editor,
  reload comparison, artifact pins, point placement, point collection edits,
  preview provenance, surface cache region edits, remote metadata cache and
  surface catalog. The input-reader regression verifies that displayed fit
  surfaces can be queried by their store UUID without creating a draft.
- The object-store Python suite and 16 Spiral service/runtime/input/config
  test modules passed: 475 tests and 60 subtests, with one skipped test.
  The process exited successfully but nanobind reported leaked references
  at shutdown; this validation does not establish that the bindings are leak-free.
- The rebuilt input reader loaded the nine-object real-data baseline with
  no missing or changed inputs; `display_geometry_dump.py compare` found its
  display geometry identical to the baseline.
- No interactive visual inspection or GPU fitting run was performed for
  this merge. Stage 2's prerequisites and acceptance tests remain outstanding.

## Stage 2 — store-backed edits and pinned input updates

This replaces the earlier "live apply" and "shared database over SSH" sketches
with one model. VC3D pushes its edits and names the position it reached; the
service links on top of that, pushes the links, and fits the resulting fixed
hub position.

### 2.1 One authoritative store and input positions

- Client and service share one authoritative
  [hub](../vc_object_store/docs/README.md#hosted-hub-and-replicas), with a
  separate working replica on each side. Local profiles use a local hub file
  through filesystem access; remote profiles can use a mount path or an
  `sftp://`/`ssh://` address. Local hub/replica setup is managed by the
  application. Both modes use the same protocol; stage 2 does not add a
  shared-working-file update path or queue local database writes. The
  HTTP tunnel still carries fit commands and artifacts; store data moves only
  through hub sync. Do not sync live SQLite files or add a store server. An
  existing hub is never opened as an ordinary working store.
- An **input position** names an immutable store state by hub identity and
  journal sequence, in local and remote mode alike. Retain historical-position
  semantics, using the store's existing revision history rather than a separate
  fit-snapshot system. First investigate on-demand retrieval/import of a
  historical hub state into a replica-local read view, including metadata,
  tags, deletions, relationship membership/approval and referenced geometry.
  Geometry revisions alone are insufficient. The exact retrieval API remains
  store implementation work; do not pass a journal sequence to the existing
  `published_at_us` API. Pins and digests use stage 1's replica-independent
  identities. A fresh clone copies the hub's existing history; incremental
  pulls install current record values and can skip intermediate object and
  relationship revisions, so those replicas need the historical retrieval path.
- The service's replica writes only detector relationships, under the
  service's own author (for example `spiral-service@<host>`). It never edits
  objects and so never holds leases; the detectors write only edges and
  replace only their own still-pending ones, so a user's approval or rejection
  is never overwritten. The service pulls and pushes only on command, never on
  a background schedule, so nothing changes its replica while detectors run.
  The service is the only detector writer for its selection; a CLI refresh
  against the same hub is a second writer and must be avoided or will conflict.
- The hub journal is linear and sync is per-record optimistic concurrency: a
  push is one hub transaction whose records must still have the hub version
  they were edited from, and records changed on both sides become conflicts.
  So divergence exists only as unpushed replica changes, and every fitted
  position is one hub state.
- The service syncs through its own `ObjectStore` connection's `hub_sync`
  (`pull`/`push` flags), not an `ObjectStoreSession` attachment: the store's
  documented Python-service pattern (lasagna's) attaches a session, whose
  worker pulls and pushes every `HubSessionOptions::syncInterval` (15 s) and
  after each action. `hub_sync` requires an idle connection, so the update
  loop holds no other reference to that connection while it syncs.
- Settle the remaining hub prerequisites before the service port. Checked
  against this branch on 2026-10-28: Python already exposes `hub_sync`
  (separate pull/push flags), `hub_status`, `revision_selection_at`,
  `revision_read_view_at` and `relationships_at`. The historical selection
  APIs take a local publication timestamp, not a hub position.
  `HubSyncReport` still returns only pulled/pushed counts and conflicts:
  no journal cursor, push sequence or installed-record versions. The session
  has no push hold, and the crash-recovery prerequisite below remains open.
  - Extend the existing Python/C++ sync APIs with the replica's pulled
    journal position, the resulting position of the service's own push, and
    an optional expected-head precondition checked atomically with publication;
  - reconstructing a hub position as a replica-local read view. The current
    pull installs the latest values of changed records in one local
    transaction; it does not reproduce every intermediate hub state. Reuse
    hub history, initially investigating on-demand historical retrieval as
    described above, rather than requiring all history to sync eagerly;
  - replica lifecycle (clone, reattach, loss);
  - hub placement. Put the hub on the service host: the service opens it by
    local path and needs no credentials, and VC3D reaches it over `sftp://`
    with its existing ssh prompts. A hub elsewhere needs the service account
    to have an ssh key (batch mode: keys or agent only) with write access to
    the hub file and its directory, and a `known_hosts` entry; check this at
    startup and fail clearly. Push needs no more access than pull, since every
    hub operation, a pull included, creates `<hub>.lock`;
  - account for the existing push semantics: version checks are per changed
    record, not against the whole hub head, so a push can land on top of
    unrelated foreign commits. Preserve that behavior for ordinary sync;
    the service's conditional push refuses if the hub head differs from H;
  - behaviour when the hub is unreachable: the resident fit continues and an
    update fails;
  - hub crash recovery (being fixed separately in `vc_object_store`). After a
    client dies mid-push and `vc_object_hub unlock --force`, the next open
    must roll back the interrupted transaction from its journal. Today the
    `unix-dotfile` and SFTP lock checks see the opener's own lock directory
    as another writer's reserved lock, so the journal is never treated as hot
    and the half-applied push is read. Stage 2 makes the service a regular
    pusher, so it depends on this fix and its crash test. The existing
    `test_object_store_hub` simulated-crash case covers lost local bookkeeping
    after a successful push, not rollback of an interrupted hub transaction;
  - relationship conflicts: relationships are not leased, and two replicas
    changing one relationship (a VC3D edit revalidating its links, or a user
    approval, against a service refresh) conflict.

### 2.2 Edit, then update

- Restore patch brush, PCL and fiber editing in VC3D as checked writes into its
  replica (`holdLeases`, `expected_revisions`), synced by `ObjectStoreSession`'s
  background worker. Extract shared editor behavior where necessary rather than
  copy private implementations. Preserve metadata roles, stable IDs,
  coordinate declarations and branch links. A saved edit is a durable revision.
  Discarding a local draft does not roll back history; undo is a new checked
  edit. Do not recreate the old filesystem Apply/Commit model.
- Reuse the merged session surface-draft APIs and store-backed viewer queries.
  `Draft` events describe unsaved geometry; `Committed` events describe saved
  store revisions. Keep that distinction in the four UI states below. A draft
  may override geometry in local viewer queries, but linking and fit snapshots
  read committed history only. Saving still needs checked revisions and leases;
  `setDraft`/`updateDraft` do not save or acquire them.
- The UI distinguishes four states: an unsaved draft; a saved edit (with hub
  sync pending, pushed or in conflict); inputs linked at a prepared position;
  and the position active in the fit. Store events mark the fit's inputs stale
  and coalesce notifications. Nothing changes the resident fit automatically,
  and no store change reinterprets a pinned preview underneath an edit.
- **Update fit inputs** is an explicit user action in VC3D, not part of every
  save:
  1. Push pending edits and wait for sync. Stop on hub conflicts and show them.
     Read the hub position M they reached.
  2. Hold further pushes from VC3D's sync worker until the update finishes
     (a new session push hold; `HubSessionOptions` has none today).
     Edits still save locally and push afterwards. With one VC3D client this
     keeps the update free of interleaved edits; 2.3's loop covers the rest.
  3. Send an apply request with M, a command ID and the active generation.
     The service answers with the position N it fitted and the manifest/digest.
     VC3D then pulls to N, which also brings it the service's links.
  VC3D runs no detectors, and needs neither the normal arrays nor the C++
  detector calls. Linking stays in `object_store_inputs.refresh_links()`, shared
  with the CLI.
- Initialize and full-input rebuild accept a position in the same way, and use
  the same link-then-pin path; stage 1's startup refresh becomes that path.

### 2.3 Prepare and install a pinned generation

- Link, then pin. Each update runs this loop on rank zero:
  1. Pull to the hub head H (at least M).
  2. Run `refresh_links()` over the whole selection at H, so every selected
     object at H is linked, new objects included.
  3. Conditionally push the links with expected hub head H. In the same hub
     write transaction, check the head and either publish all changes at N or
     publish nothing and return a head-mismatch result. Keep the existing
     per-record conflict checks. No hub lock is held while detectors run.
     Even a no-op refresh must validate H atomically; success then returns N=H.
     Reconcile an uncertain/lost response before retrying: a previously
     successful publication must recover its original N, not be mistaken for
     a fresh head mismatch and executed again.
  4. On success, resolve and pin the exact historical state N. Later hub
     commits do not change this position; pulling past N must not substitute
     newer inputs. There is no post-push foreign-record scan.
  5. On a head mismatch, discard/reconcile unpushed detector output against
     the hub, pull the new head and restart at step 2. Never retry old results
     with a new expected head without relinking. An unrelated hub edit also
     causes a retry; accept this conservative behavior initially.
  If a push or pull reports conflicts, resolve them to `"hub"`, since this
  service replica writes only recomputable detector output and the hub side
  may hold a user's edit or approval. Restart from the newly pulled head.
  Bound the loop; when the bound is hit, fail the update
  ("inputs kept changing; retry") and keep the active fit. A refresh failure
  likewise leaves the fit untouched and reports the detector writes already
  pushed, plus any local detector writes awaiting reconciliation. A successful
  conditional push establishes N as H plus the service's links. Existing
  object leases and per-record checks alone do not establish this: they do
  not freeze selector membership or prevent unrelated hub commits. VC3D's
  push hold (2.2) makes repeats rare without blocking local saves.
- N may include edits made after M. That is correct, because the loop linked
  them; report N, and VC3D shows which edits it includes beyond M. Records
  after N are later edits, excluded and shown as stale.
- Then pin N's read view and build the candidate snapshot through the shared
  adapter with the CLI's no-refresh semantics: selection, approvals,
  deletions, relationships, selector membership and materialization. Because
  N is immutable, no write exclusion is needed while preparing. Edits after N
  are not included, and show as stale.
- Full rematerialization is the correctness path. Diffing the active and
  candidate positions (compare their revision selections, or use the records
  the pull installed) can later limit rematerialization to changed objects
  and the edges touching them. Adopt that only as a measured optimization
  whose materialized inputs are identical.
- Links depend on the refresh parameters (the `pcl_link_*` values and the
  normal source from `fiber_normal_source()`), and the normal arrays live
  outside the store. Record the parameters, including the normal source's
  identity, in the manifest, so that a later parameter or normal change shows
  the fit's links as stale. Every apply refreshes, so the loop needs no store
  record of which detectors ran over which objects. `relationships_at` remains
  a cheap assertion that no pinned link is inapplicable.
- Input updates change supervision, not trainable parameters. Keep the existing
  `SpiralAndTransform`, optimizer (including moments), scheduler and iteration
  count. Adding/removing/editing patches, fibers or PCLs does not introduce
  parameters. Rebuild their atlases, sampling/index structures and other
  input-dependent state, and invalidate derived caches. UUID mappings identify
  inputs and derived indices, not optimizer parameters; never reuse an old
  array index as an identity. An atlas layout change is not a model rebuild.
  Model configuration/domain changes remain separate operations; an input
  update must not implicitly resize or reinterpret the model or its frame.
- At an acknowledged iteration boundary, pause every rank. Reuse the shared
  command barriers with a validation/replacement interface on `FitContext`.
  Validate the pinned host inputs and agree on their digest across ranks
  before releasing device input state. Optimize for successful updates and
  low peak device memory: release the old input device stores before building
  their replacements, rather than retaining two generations. Keep the model
  and training state described above; do not duplicate them for rollback.
- Failure semantics have an explicit destructive boundary. Before any rank
  releases old device input state, a refusal leaves the old fit usable.
  After release begins, any allocation, materialization or installation failure
  stops all ranks and marks the session recovery-required, even if only one
  rank failed. Do not claim that the old generation remains runnable, resume
  partially installed state, or automatically retry the destructive command.
  Require an explicit full rebuild or checkpoint recovery. Checkpoint recovery
  uses the last successfully saved checkpoint; do not promise an automatic
  pre-update checkpoint. Report the candidate position, failure phase and any
  already-published links. Publish the new active generation only after every
  rank has installed it successfully.
- Version previews, diagnostics, checkpoints, caches and input overlays with
  the session generation, position and digest. An old preview stays viewable as
  old; it cannot overwrite the active generation's display or authorize an edit
  against newer geometry. VC3D displays its replica as it is now and compares
  it with the active position's manifest, as in stage 1; a replica behind
  the service's position pulls forward first. On reconnect, recover status before replaying commands;
  event cursors alone are not durable applied-state evidence.

### Stage 2 acceptance

- Before removing stage 1's frozen-input restrictions, regression-test their
  replacement against stage 1 behavior for reconnect, preview freshness and
  checkpoint provenance. Once the comparisons pass, delete the replaced code,
  routes, options and tests.
- A local hub file with two working replicas, and one remote hub with a VC3D
  replica on a mount path and a service replica at an
  `sftp://` address: both give identical digests for one position. Test an
  interrupted push or pull, an unreachable hub, and a lease or conflict that
  blocks Update fit inputs.
- Concurrent edits during an update: a second client edits a linked patch,
  creates an object, and approves an edge the service is re-writing. The loop
  links them, conflicts resolve to the hub, the approval survives, and the
  bound fails the update cleanly under continuous edits.
- Create/edit/delete patches, PCLs and fibers; annotation and approval changes;
  ordinary versus adjacent branches; empty selections and changed tags. Edits
  after N are excluded and shown as stale.
- The conditional push publishes nothing when the hub advanced after H,
  including unrelated commits. Test no-op linking, retry limits, lost push
  responses and a commit immediately after successful publication. A later
  pull must still recover exactly N. Test historical reads on a fresh clone
  and a replica that skipped N, including metadata-only and approval changes.
  A normal-source change marks the fit's links stale. A detector failure
  leaves the fit untouched and reports pushed detector writes.
- Applying one position twice is idempotent. Test disconnecting after apply,
  and delayed previews.
- Two ranks: matching installation, validation failure before device-state
  release, and allocation/materialization failure after release on either
  rank. The first leaves the old fit usable; the latter stops every rank and
  requires recovery. On successful updates, verify model parameters, optimizer
  moments, scheduler and iteration count are unchanged at the installation
  boundary, including when input counts and atlas layouts change. Resume from
  a checkpoint of the applied generation. Measure peak device memory to verify
  old and replacement input stores are not retained together.
- Real-data VC3D smoke over a real SSH hub: edit → save → Update fit inputs →
  continue fitting → view the new preview. Old previews remain correctly
  identified. Pre-release validation failure leaves the previous fit usable;
  post-release failure is visibly recovery-required and can be recovered by
  an explicit rebuild/checkpoint load.

### Stage 2 implementation sequence

Split into 2a, store-backed editing with initialize/full-input rebuild at a hub
position, and 2b, updating a resident fit. 2a gives a usable edit-and-refit
loop before replacing input device stores inside a resident fit. Stage 2b
keeps model and optimizer state in place; its additional risk is incomplete
replacement of input-dependent state. Store prerequisites (steps 1–5) land in `vc_object_store` first,
agreed with its owners; the adapter steps (6–7) are shared with the CLI and
land before the service. Each of 2a and 2b lands its service and VC3D steps
together with one `API_VERSION` / `kApiVersion` bump; keep service and VC3D
commits separate.

1. **Store: hub positions.** `HubSyncReport` carries the journal cursor after
   a pull and the resulting journal position of this replica's push. Extend the
   existing Python report/status bindings. C++ hub tests and Python tests.
2. **Store: position to read view.** Map a hub position to the replica's
   local read view, including positions skipped by a pull. Reuse the existing
   hub history, investigating on-demand historical retrieval/import first,
   and the historical read bindings where appropriate; their local timestamp
   arguments must not be mistaken for hub positions. Test that two replicas
   with different pull histories give the same selection at one position,
   by replica-independent identities.
3. **Store: session push hold.** An RAII hold on an attachment's pushes
   (like `LeaseHold`); `syncHubAndWait` reports the position it reached.
4. **Store: hub crash recovery** and its crash test (in flight separately).
5. **Store: conditional publication.** Add an expected-head precondition to
   push, checked within the publication transaction. Mismatch publishes
   nothing; success returns N, including N=H for a no-op. Preserve ordinary
   per-record sync behavior. Test concurrent commits, no-op validation and
   reconciliation after a lost response; keep publication idempotent.
6. **Adapter: input positions.** The request and manifest carry the position;
   the snapshot reads at its pinned view. The manifest records the refresh
   parameters (`pcl_link_*`) and the normal source's identity, so a parameter
   or normal change marks the links stale.
7. **Adapter: link-then-pin.** `prepare_position()` runs 2.3's loop on rank
   zero: pull H, `refresh_links()`, conditional push at H, pin N. Head
   mismatch reconciles local detector output, pulls and relinks; conflicts
   resolve to `"hub"`. Bound retries and report writes already pushed and
   local writes awaiting reconciliation. Test a foreign push, a conflicting
   approval, no-op linking, and recovering exact N after a later pull.

Stage 2a:

8. **Service: hub binding.** Launch binds a replica and its hub address under
   the service's author. Startup clones or reattaches the replica and checks
   hub reachability and, for remote hub access, ssh batch mode and
   `known_hosts`, failing clearly. Local access uses filesystem paths.
   Discovery advertises the hub identity and the fitted position. Bump
   `API_VERSION`.
9. **Service: initialize and full-input rebuild at a position.** Both take M
   and run `prepare_position()`, replacing stage 1's startup refresh; all
   ranks agree on the digest before training. Report N and the
   manifest/digest; command IDs make a repeated request idempotent.
10. **VC3D: hub profiles.** Profiles bind a hub (mount path or `sftp://`) with
    a local replica, compare store UUID and position with the service, and
    replace stage 1's externally provisioned store. Manage local hub and
    separate client/service replicas automatically; preserve existing data
    and pending edits when provisioning or reattaching. Bump `kApiVersion`.
11. **VC3D: store-backed editing.** Patch brush through `holdForEditing`/
    `editCurrent`, using the merged session draft APIs for unsaved surface
    display; PCLs through `PointCollectionSync`, fibers through the
    existing store edits; preserve metadata roles, stable IDs, coordinate
    declarations and branch links. Show 2.2's four states.
12. **VC3D: Update inputs (rebuild form).** Push and wait, stop on and show
    conflicts, hold pushes, send M with a full-input rebuild, pull to N, show
    edits included beyond M and stale ones after N.
13. **Service and VC3D: generation provenance.** Version previews, autosave,
    `PublishedPreviewIndex`, checkpoints, caches and overlays with generation,
    position and digest; resume from a checkpoint at a position.
14. **2a acceptance.** Second-replica concurrent edits, an unreachable hub,
    interrupted push/pull, a conflict blocking the update, and a real-data
    VC3D smoke over a real SSH hub (edit → save → rebuild → fit → preview).

Stage 2b:

15. **Fitter: input replacement.** A validate/replace interface on
    `FitContext`. Inventory input-dependent atlases, pools, indices and
    caches; rebuild/invalidate them while keeping model, optimizer moments,
    scheduler and iteration count unchanged. Release old input device stores
    before allocating replacements. No optimizer remapping or model rebuild
    for changed input counts/layouts. Test the destructive boundary and state
    preservation; measure device-memory use on representative inputs.
16. **Runtime: apply at an iteration boundary.** Reuse the shared command
    barriers; all ranks validate and agree on the candidate digest before
    releasing old device inputs. Pre-release refusal keeps the old generation;
    any post-release failure stops all ranks and requires explicit recovery.
    Publish the generation only after collective success. Two-rank tests.
17. **Service: apply route.** `POST` with M, command ID and active generation
    returns N, manifest/digest and the new generation; idempotent by command
    ID; reconnect recovers status before replaying. Lift stage 1's
    `inputs_frozen` refusal for this route only. Bump `API_VERSION`.
18. **VC3D: Update fit inputs against the resident fit.** Reuse step 12's flow
    against the apply route; the display follows the active generation,
    pulling forward if behind and marking a mismatch otherwise. Bump
    `kApiVersion`.
19. **Acceptance and cleanup.** Run the stage 2 acceptance above, compare
    reconnect, preview freshness and checkpoint provenance against stage 1,
    then delete the frozen-input restrictions, routes, options and tests;
    document hub setup and updates in README.md.

Later work: diff-based incremental materialization; automatic Update fit
inputs; storing Spiral results as store objects.

### Stage 2 implementation record (2026-10-28)

Commits on `spiral-object-store` (not pushed), in order:

- Store (steps 1–3, 5): `b5ed834b2` hub positions in sync reports and
  statuses, `hub_publications` (position → publication time) written by
  every journaling hub write, `readHubPosition`, `PushHold`, conditional
  `hubPush(expectedHead)` with lost-response recovery; `32f8b0c42` read-only
  replicas read positions, a no-op retry recovers its own publication;
  `6504d0f9e` session status names its hub; `f379073e9` `hubJournal`;
  `43d67591e` fixes a use-after-free in the session worker's conflict
  resolution (found by the conflict acceptance test); `c5c3f0fb5` docs.
- Step 2 investigation outcome: a position is read through the hub's own
  history (`revisionSelectionAt` at the publication's time, then
  `relationshipsAt`/`objectRevision` on the hub), not by importing history
  into the replica. Records are small and the reader has just contacted the
  hub; geometry revisions are shared and read locally, and a read refuses a
  position whose geometry the replica has not pulled. A replica-local import
  would duplicate history with conflicting revision identities. Resuming at a
  position therefore needs the hub to be reachable.
- Adapter (steps 6–7): `6b17a8984` `InputPosition` in requests and manifests
  (outside the digest), link parameters and normal source recorded,
  `stale_link_reasons`, `prepare_position`.
- 2a service (8–9): `c04bb94c3` shared config merge and lasagna inputs,
  `cd8526db9` `--object-store-hub`, builds at linked, pinned positions before
  teardown, checkpoint resume at its position (API 37).
- 2a VC3D (10–12): `d3fea3432` hub profiles, separate local service replica,
  Host Object Store Locally, Clone the Service's Hub, Update Fit Inputs;
  `d10969011` store-backed brush/PCL saves (`SpiralStoreEdits`) and the four
  states; `f83f7ec27` conflict-blocking protocol test.
- 2a provenance (13): `d3b31f39b`, `633f1fd8f`.
- 2b (15–18): `7a64c111f` mechanical split of `load_host_inputs`;
  `50b2cb952` `FitContext.prepare_input_replacement`/`replace_inputs`;
  `5603117da` prepare/install/publish runtime commands and coordinators;
  `d004b35ba` `POST /session/update-inputs` (API 38); `cd03c8615` VC3D in
  place (kApiVersion 38).
- 19: `2fe7681db` upload refusal no longer cites frozen inputs (code
  `inputs_in_object_store`), README; `ac3be576f` VC3D message and docs.
  `inputs_frozen` remains a capability, true for a service without a hub.

Deviations: track exclusion masks with `track_exclusion_radius > 0` depend on
the inputs' trusted geometry and are refused for in-place updates (rebuild
instead). A failed session after an update is recovered through the existing
rebuild/checkpoint paths. A plain (non-hub) store keeps stage 1's frozen
semantics rather than being removed. Step 11 does not use `holdForEditing`
or `PointCollectionSync`: `SpiralStoreEdits` holds leases and pins each
replaced or retired object at the revision its edit started from (a brush
patch at the revision its last save wrote, a fitted collection at the
revision the workspace read, `local_object_revision` in its document), via
`act`/`editKeepingRelationships` and `writeVolumeCollections(..., expected)`;
a save is refused when the object has moved on.

Review fixes (2026-10-28, after the record above):

- `5b5e91993` store: `hubRevert` (discard unpushed relationship changes and
  conflicts), `hubAddress`, `writeVolumeCollections` takes expected pins.
- `b5ce7cb22` `prepare_position` discards the replica's unpublished links
  before every attempt and on failure (a head mismatch had relinked over
  them, so a link to an object deselected meanwhile could be published).
- `a4100ced4` the service compares hub addresses as the store records them
  (an `ssh://` hub could not restart).
- `d3f62a227` an update of a busy session is refused before admission (the
  single-rank refusal had queued a discard that broke a running step's
  barrier); Run, rebuild and checkpoint loads wait for an update in flight.
- `56ff8b695` VC3D ends an update whose connection changed (its pushes were
  held until restart), disables Run during updates, refuses to rebuild a
  busy fit for an update.
- `755dd9003` the update candidate drops the old theta topology, which kept
  the old atlas and strips alive through the install. Re-measured with the
  2b driver: peak +35.9 MB (was +48.9 MB), everything else as below.
- `d1ce39606` VC3D saves pin the revision their edit started from (they
  pinned the current revision after the lease's pull).

Second review round (2026-10-28):

- Provenance: `064d55cdd` a preview fitted to replaced inputs stays indexed
  but is not shown as current (status `preview_input_position`), and the
  update records its position in the live session request (a model-only
  rebuild reported the old one); `f5a9b8acd`/`d6dfd3fa5` overlap pairs are
  materialized from the snapshot's edge, frames and geometry revisions
  (`overlap_correspondences_at`), not the replica's current state;
  `3223ce27e` the input manifest is keyed by digest and input generation;
  `ae41f76b7` discovery reports why the replica's position is unreadable
  (the advertised position stays the replica's last pull, as documented).
- Robustness: `e704a11cf` a push whose retry fails is a PositionError;
  `5feede971` coordinator prepare timeouts and publish failures are
  InputUpdateFailed; `0087f237f` refusal codes, theta exclusions judged
  afresh per update, startup drops unpublished links and refuses other
  edits on the service replica; `313341723` a position needs only the
  geometry a reader reads (`requireHubGeometry`); `48f140e6d` an indexed hub
  (and its replicas) has schema version 18, which pre-index binaries refuse;
  `5e44a783a`/`70eb6cf15` PushHold limits documented, and VC3D ends an update
  whose attachment changes; `9f403f732` a new collection saved twice
  replaces its object; `2691f02e4` VC3D saves drawings off the GUI thread;
  `4101fb6b8` the relink note after a lost publication response.
- Not changed: lost-push recovery still matches pushed records by value (a
  push token would need a hub_publications column; the relink is correct and
  only its note was wrong); Host Object Store Locally relies on the
  synchronous service stop on detach plus its -wal/-shm check.
- The 2b driver after this round: all steps as above, peak +35.9 MB.

Third review round (2026-10-28): `1f2d17b0f` no second VC3D update while
one runs (and a dropped request no longer reports an outcome);
`aef428a28` hubRevert is a pull that also replaces the discarded
relationships (the hub's current edge could be bound to endpoint geometry
the replica lacked: FOREIGN KEY failure), and `hub_writers`;
`4d79aaccd` startup refuses a replica others write before reverting
anything; `060ee801b` an update retires only drawings unchanged since the
save it included, a save whose revision cannot be read back stands, and
a refusal because another editor saved first names their revision so
saving again replaces it.

Acceptance (Linux, RTX 3090 shared with another job; stage 0 acceptance store
copied as a hub; dataset dir with spiral-scroll.json, umbilicus.json,
lasagna_inputs, outer_shell, fiber_directions.npz; z 8900–9300):

- 2a over a real SSH hub (`sftp://wari/...`, scratch driver `real_2a.py`):
  VC3D-role replica and service replica both cloned over SFTP; initialize at
  head 2881 linked 553 PCL-patch, 6 fiber-patch and 444 overlap links and
  pinned 3884; 300 iterations and a preview; a surface's erosion and a
  collection's sampling group edited, pushed to M = 3886; an update request
  for a position the hub never reached was refused with the fit Idle and
  unchanged; rebuild at M pinned N = 3886 (no new links), the fitted manifest
  carried both edits, the client pulled to 3886; 300 more iterations and a
  preview. VC3D's `SpiralStoreInputs` (SPIRAL_STORE_INPUTS_DUMP) resolved all
  954 objects and 1,103 links as fitted in the client replica.
- 2b on GPU (`tests/input_update_driver.py`, in-process ServiceState): after
  300 iterations an editor retired a fitted surface (keeping its links as
  recorded) and edited a collection, pushed to 3886; the in-place update
  (9.2 s) relinked and pinned 4534; patches 181 → 180; model parameters,
  optimiser state, scheduler state and iteration bitwise unchanged, same
  objects; device memory 737.4 MB before, peak +48.9 MB (+35.9 MB after
  `755dd9003`: the new theta topology and staging; atlases 11.5 MB each),
  737.4 MB after. 300 more iterations, loss
  49.8, finite; previews indexed at positions 3884 and 4534; checkpoint
  records 4534; a refused update (linking) left the fit Idle, digest unchanged,
  training on; a simulated allocation failure after release put the session
  in Error with `recovery_required`, and a rebuild from the checkpoint resumed
  at 4534 with the same digest and iteration 600.
- Suites: store ctest 16/16 (with `object_store_hub_positions`), store Python
  136; spiral-fitting position/update/service/runtime suites; VC3D `ctest -R
  spiral` 10/10 (with `spiral_store_edits` and the hub protocol test),
  `VC3D --test-object-store`.

Not run: two-rank GPU updates (one GPU; two gloo ranks and the coordinator's
decisions are tested on the CPU); step 4's crash recovery and an interrupted
hub transaction; a hub on a different host from the service; interactive VC3D
GUI inspection (the VC3D client code ran in the protocol tests, against the
fixture service).

## Implementation and validation workflow

Keep service and VC3D changes in reviewable patches, with their protocol changes
landed together: bumping `API_VERSION` and `SpiralServiceManager::kApiVersion`
couples the stage 1 service and workspace ports into one landing. Deleting the
old file-input code can follow as a separate patch once the baseline
comparisons pass. No new fitter/linker numerics are intended beyond the accepted
CLI port; successful stage 2 input updates preserve the existing model and
optimizer state. Explicit model rebuilds and checkpoint recovery remain
separate operations. Preserve Ubuntu and macOS support and existing
CUDA/MPS/CPU behavior.

Use existing environments; no installation/bootstrap is part of this plan.
From the repository root, use the regression entrypoint below with an existing
Python environment (`spiral-fitting/.venv` is the documented default, not a
prerequisite to create here). Include the built `vc_object_store` Python module
so adapter tests run rather than skip; `VC_OBJECT_STORE_BUILD` names an existing
store build. These tests cover the stage 1 service and shared adapter;
stage 2 adds the position, editing and update acceptance cases above:

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting:$VC_OBJECT_STORE_BUILD/python spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_spiral_service_v2.py \
  spiral-fitting/tests/test_revisioned_runtime.py \
  spiral-fitting/tests/test_checkpoint_load.py \
  spiral-fitting/tests/test_service_events.py \
  spiral-fitting/tests/test_service_object_store.py \
  spiral-fitting/tests/test_runtime_object_store.py \
  spiral-fitting/tests/test_preview_restore.py \
  spiral-fitting/tests/test_object_store_startup.py \
  spiral-fitting/tests/test_object_store_inputs.py \
  spiral-fitting/tests/test_object_store_materialize.py \
  spiral-fitting/tests/test_object_store_fibers.py \
  spiral-fitting/tests/test_object_store_overlaps.py \
  spiral-fitting/tests/test_config.py \
  spiral-fitting/tests/test_input_toggles.py \
  spiral-fitting/tests/test_run_boundary_settings.py
```

Stage 1 removed the live input-batch paths and their old runtime tests.
The old file-input editing tests are historical baseline sources at the
pre-port checkout, not tests to restore or run against the current service.
Retain the captured fixtures under `tests/data/spiral_service_baseline/` and
the store-backed regression tests as stage 2's comparison baseline.

Capture old-path regression baselines before porting file-input tests to the new
contract, and add focused service/workspace regressions for the acceptance cases
above. Gate deletion of each old path on its replacement passing those
comparisons; audit remaining references after deletion. Keep CLI adapter tests
passing. Use a pre-port checkout or captured fixtures for implementations already
removed by the CLI port; do not restore them to production code.
For VC3D, use an existing configured build with tests enabled; discover its build
directory and registered targets before running `cmake --build <build-dir>
--target VC3D` and `ctest --test-dir <build-dir> -R spiral --output-on-failure`.
Build the relevant registered test targets as well; building VC3D alone does not
guarantee those executables exist. Use offscreen Qt where supported for UI tests.
Record exact resolved commands and real-data inputs in each implementation's
acceptance notes. This documentation-only change does not claim those runs pass.
