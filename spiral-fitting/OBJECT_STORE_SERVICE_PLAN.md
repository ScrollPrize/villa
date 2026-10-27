# Object-store inputs for the Spiral service and VC3D workspace

## Scope and dependencies

Companion to [OBJECT_STORE_CLI_PLAN.md](OBJECT_STORE_CLI_PLAN.md). CLI steps
11–12 are complete as of 2026-10-26: cleanup landed in `84757fd9d`, acceptance
and documentation in `cb0e796d8`, and review fixes in `ebc43f070`. Normal
sources are then addressed by array path rather than multiscales index
(`89abe9606`, `b793e0b88`, superseding `743d6d6d2`). Stage 0 commits 1–3 landed
on 2026-10-27 (`0ed53d4ce`, `6221efeba`, `0061bc9b8`); step 4's re-ingest and
CPU startup smoke are recorded in the CLI acceptance notes, and its GPU fit is
still to run.
This document is a plan, not a claim that the service already supports that port:
the current service cannot start a fit against the store-only `FitContext`
(it still passes the removed `input_bindings` constructor argument).

The CLI acceptance notes record a real-data, single-GPU 1,000-step fit with no
patch/fiber/PCL files, finite losses, complete provenance and matching no-refresh
counts/digest. They also record 19 unrelated pre-existing suite failures; real-data
two-rank fitting and changed-input resume remain unrun (CPU startup/resume tests
cover those contracts). These limits do not replace the service acceptance below.
That run predates the array-path normal addressing and stage 0's store
changes; stage 0 reruns it before stage 1 relies on it (pending: the store was
re-ingested and its startup refresh checked, but the GPU fit has not rerun).

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
  UUIDs are per replica, so the digest and the pins VC3D resolves must use
  replica-independent identities (implementation step 2). Verify local display
  geometry/annotations against the service's pins. On mismatch, explain which
  inputs differ and refuse to present current local data as the fitted inputs.
  Preview viewing may remain available with an explicit mismatch indication.

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

- Replace file-based rows and overlay loading with store reads pinned to the
  active selection. Use the shared coordinate helpers: stored XYZ divided by
  the object's scale gives reference-volume XYZ; convert to the preview/viewer
  frame explicitly. Preserve invalid patch vertices and stable annotation IDs.
- Show service connection, local store binding, selection, active manifest,
  resolved volume and input freshness. Require matching volume declarations;
  do not use dataset-root path substitution as an object identity mechanism.
- Retain read-only patch/fiber/PCL overlays and fit previews. Adapt service PCL
  display artifacts to the frozen snapshot if still needed; never regenerate
  fitted-input overlays from unpinned current DB contents.
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
   endpoint geometry revisions) rather than by revision UUID. Verify which
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
   `*_winding_artifact` status keys): VC3D reads PCLs pinned to the manifest
   (step 10), so no unpinned copy remains.
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
10. **VC3D: pinned store reads for input overlays.** Add a shared helper
    beside `ObjectStoreArtifacts.hpp` that reads patches, PCL collections and
    fibers at the revisions the status manifest pins (`RevisionSelection`,
    typed snapshot accessors at a geometry revision, aligned annotations,
    `fiberNetworkJson(selection)`; `currentRevisions(selection)` for the
    local heads compared against them). Resolve pins by the shared geometry
    revision UUIDs and check the metadata hashes from step 2, never by the
    service's object revision UUIDs, so a replica of the service's store
    resolves them. Convert them with the store's `geometryScaleToVolume`/
    `volumeCoordinates`, keeping invalid patch vertices and stable point IDs. `fiberTransportPaths` moves onto it rather
    than keeping a second current-heads export. Replace `loadInputSurfaces`'
    tifxyz reads and the PCL artifact overlay. Key surfaces and colours by
    UUID, not directory name. `mapServicePath` remains only for external
    inputs (shell, lasagna). A revision missing locally marks the overlay
    mismatched rather than falling back to current heads. Register a
    `spiral_store_inputs` ctest on a fixture store and compare against step
    1's coordinates with the same dump entrypoint.
11. **VC3D: disable input editing; show bindings and freshness.** Disable
    brush-to-file staging (`finalizeBrushPaint`), PCL replacement and new-PCL
    staging, row Add/Commit/Remove/Restore/conflict actions,
    `addPatchToCurrentFit`, `addFibersToCurrentFit`/`stageFiberInputs` and
    managed patch copies, each explaining that input changes need a
    full-input rebuild until stage 2. Keep the in-memory brush and point
    editors for stage 2. Show service connection, store binding, selection
    counts, manifest digest, resolved volume, and input freshness: store
    events touching selected objects (`StoreSessionEvent::touches`) mark the
    overlays and fit stale without reloading them. Previews show their
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

Not run, because the GPU was held by another job throughout: stage 0's
1,000-step GPU fit, the real short session of step 1 (preview manifest and
checkpoint provenance keys), and step 14's real-data service fit inspected in
VC3D. The VC3D workspace was built and its logic tested, but not driven
interactively.

Stage 2 has its own sequence (below), after the stage 1 gate. It depends on
step 2's replica-independent identities, step 4's lease, step 10's pinned
reads and step 11's freshness tracking. It also depends on the hub
prerequisites (2.1), which can be settled during stage 1.

## Stage 2 — store-backed edits and pinned input updates

This replaces the earlier "live apply" and "shared database over SSH" sketches
with one model. VC3D pushes its edits and names the position it reached; the
service links on top of that, pushes the links, and fits the resulting fixed
hub position.

### 2.1 One authoritative store and input positions

- Client and service share one authoritative store. Either there is one
  [hub](../vc_object_store/docs/README.md#hosted-hub-and-replicas) with a
  working replica on each side (a mount path or an `sftp://`/`ssh://`
  address), or a local profile has both processes open one store file. The
  HTTP tunnel still carries fit commands and artifacts; store data moves only
  through hub sync. Do not sync live SQLite files or add a store server. An
  existing hub is never opened as an ordinary working store.
- An **input position** names an immutable store state. It is the hub journal
  sequence for a hub-backed store, or a `published_at_us` time for a single
  shared store. The service pins `revision_read_view_at()` on its replica once
  it has reached that position. Pins and digests use the replica-independent
  identities from stage 1 step 2.
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
- Settle the hub prerequisites before implementation. As of stage 0's store,
  `hub_sync` returns a `HubSyncReport` with only pulled/pushed counts and
  conflicts: no journal cursor, push sequence or per-record hub versions; the
  Python `hub_status` and the revision-at bindings are not bound; the session
  has no push hold; and no commit addresses the crash-recovery lock check.
  - Python access to pull and push, to a replica's pulled journal position,
    to the journal sequence of the service's own push, and to the hub versions
    of the records a pull installs (2.3's update loop needs all of these);
  - mapping a hub position to the replica's local read view;
  - replica lifecycle (clone, reattach, loss);
  - hub placement. Put the hub on the service host: the service opens it by
    local path and needs no credentials, and VC3D reaches it over `sftp://`
    with its existing ssh prompts. A hub elsewhere needs the service account
    to have an ssh key (batch mode: keys or agent only) with write access to
    the hub file and its directory, and a `known_hosts` entry; check this at
    startup and fail clearly. Push needs no more access than pull, since every
    hub operation, a pull included, creates `<hub>.lock`;
  - whether a push needs the replica to have pulled the hub head first, or can
    land on top of foreign commits (2.3's update loop is written either way);
  - behaviour when the hub is unreachable: the resident fit continues and an
    update fails;
  - hub crash recovery (being fixed separately in `vc_object_store`). After a
    client dies mid-push and `vc_object_hub unlock --force`, the next open
    must roll back the interrupted transaction from its journal. Today the
    `unix-dotfile` and SFTP lock checks see the opener's own lock directory
    as another writer's reserved lock, so the journal is never treated as hot
    and the half-applied push is read. Stage 2 makes the service a regular
    pusher, so it depends on this fix and its crash test;
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
  3. Push the links; their transaction is at hub position N.
  4. Pull again. If it installs no foreign records with hub versions between H
     and N, then N is exactly H plus the service's own links and is linked by
     construction. Pin N.
  5. Otherwise return to step 2 from the new head. Re-linking is cheap: the
     detectors cache unchanged inputs and net no-op writes create no revision.
  If a push conflicts, resolve every conflict to `"hub"`, since detector output
  is recomputable and the hub side may hold a user's edit or approval, and
  return to step 2. Bound the loop; when the bound is hit, fail the update
  ("inputs kept changing; retry") and keep the active fit. A refresh failure
  likewise leaves the fit untouched and reports the detector writes already
  pushed. The service never fits its local replica state: it is not a hub
  position when other commits land before its push. VC3D's push hold (2.2)
  makes repeats rare.
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
- At an acknowledged iteration boundary, pause every rank. Reuse the shared
  command barriers with a new candidate-generation prepare/install interface
  on `FitContext`. All ranks validate and agree on the candidate digest before
  installation. If preparation fails, discard the candidate runtime state and
  keep the active generation. If installation fails after ranks diverge, stop
  the session and require recovery rather than resume a mixture of
  generations.
- Specify which model and optimizer state survives each update. Preserve it
  only through validated UUID-based mappings. Changes affecting atlas topology,
  model domain, parameter shapes or frame require an explicit rebuild. Never
  reuse runtime indices based on array position. Report any reset before
  accepting the update.
- Version previews, diagnostics, checkpoints, caches and input overlays with
  the session generation, position and digest. An old preview stays viewable as
  old; it cannot overwrite the active generation's display or authorize an edit
  against newer geometry. VC3D displays the fitted inputs at the service's
  pinned position, pulling its own replica forward if it is behind and marking
  a mismatch otherwise. On reconnect, recover status before replaying commands;
  event cursors alone are not durable applied-state evidence.

### Stage 2 acceptance

- Before removing stage 1's frozen-input restrictions, regression-test their
  replacement against stage 1 behavior for reconnect, preview freshness and
  checkpoint provenance. Once the comparisons pass, delete the replaced code,
  routes, options and tests.
- One hub, with a VC3D replica on a mount path and a service replica at an
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
- The loop refuses to pin a position with foreign records between its pull
  and push, and a normal-source change marks the fit's links stale. A detector failure on the service leaves the fit
  untouched and reports pushed detector writes.
- Applying one position twice is idempotent. Test disconnecting after apply,
  and delayed previews.
- Two ranks: matching installation, a preparation failure on one rank, and a
  fatal partial installation. Check that model/optimizer state is preserved or
  the explicit rebuild happens, then resume from a checkpoint of the applied
  generation.
- Real-data VC3D smoke over a real SSH hub: edit → save → Update fit inputs →
  continue fitting → view the new preview. Old previews remain correctly
  identified, and a failed preparation leaves the previous resident fit usable.

### Stage 2 implementation sequence

Split into 2a, store-backed editing with initialize/full-input rebuild at a hub
position, and 2b, updating a resident fit. 2a gives a usable edit-and-refit
loop without carrying model and optimizer state across generations, the
riskiest part. Store prerequisites (steps 1–5) land in `vc_object_store` first,
agreed with its owners; the adapter steps (6–7) are shared with the CLI and
land before the service. Each of 2a and 2b lands its service and VC3D steps
together with one `API_VERSION` / `kApiVersion` bump; keep service and VC3D
commits separate.

1. **Store: hub positions.** `HubSyncReport` carries the journal cursor after
   a pull, the journal sequence of this replica's push, and the records a
   pull installed with their hub versions; bind `hub_status`. C++ hub tests
   and Python tests.
2. **Store: position to read view.** Map a hub position to the replica's
   local read view, and bind `revision_read_view_at`/`relationships_at` for
   it. Test that two replicas give the same selection at one position, by
   replica-independent identities.
3. **Store: session push hold.** An RAII hold on an attachment's pushes
   (like `LeaseHold`); `syncHubAndWait` reports the position it reached.
4. **Store: hub crash recovery** and its crash test (in flight separately).
5. **Store: document** whether a push lands on top of foreign commits or
   requires pulling the head first.
6. **Adapter: input positions.** The request and manifest carry the position;
   the snapshot reads at its pinned view. The manifest records the refresh
   parameters (`pcl_link_*`) and the normal source's identity, so a parameter
   or normal change marks the links stale.
7. **Adapter: link-then-pin.** `prepare_position()` runs 2.3's loop on rank
   zero: pull, `refresh_links()`, push, pull, foreign-record check, pin.
   Conflicts resolve to `"hub"`; the loop is bounded and reports detector
   writes already pushed. Tests on a fixture hub inject a foreign push
   between loop steps and a conflicting approval.

Stage 2a:

8. **Service: hub binding.** Launch binds a replica and its hub address under
   the service's author. Startup clones or reattaches the replica and checks
   ssh batch mode, `known_hosts` and hub reachability, failing clearly.
   Discovery advertises the hub identity and the fitted position. Bump
   `API_VERSION`.
9. **Service: initialize and full-input rebuild at a position.** Both take M
   and run `prepare_position()`, replacing stage 1's startup refresh; all
   ranks agree on the digest before training. Report N and the
   manifest/digest; command IDs make a repeated request idempotent.
10. **VC3D: hub profiles.** Profiles bind a hub (mount path or `sftp://`) with
    a local replica, compare store UUID and position with the service, and
    replace stage 1's externally provisioned store. Bump `kApiVersion`.
11. **VC3D: store-backed editing.** Patch brush through `holdForEditing`/
    `editCurrent`, PCLs through `PointCollectionSync`, fibers through the
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

15. **Fitter: candidate generations.** A prepare/install interface on
    `FitContext`, and the state-preservation policy: UUID-keyed model and
    optimizer mappings; atlas topology, model domain, parameter shape or
    frame changes refuse and require a rebuild; resets are reported before
    the update is accepted. CPU unit tests.
16. **Runtime: apply at an iteration boundary.** Reuse the shared command
    barriers; all ranks validate and agree on the candidate digest. A failed
    preparation discards the candidate and keeps the active generation; a
    partial installation stops the session. Two-rank tests.
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

## Implementation and validation workflow

Keep service and VC3D changes in reviewable patches, with their protocol changes
landed together: bumping `API_VERSION` and `SpiralServiceManager::kApiVersion`
couples the stage 1 service and workspace ports into one landing. Deleting the
old file-input code can follow as a separate patch once the baseline
comparisons pass. No new fitter/linker numerics are intended beyond the accepted
CLI port; stage 2 must explicitly report any optimizer resets. Preserve Ubuntu
and macOS support and existing CUDA/MPS/CPU behavior.

Use existing environments; no installation/bootstrap is part of this plan.
From the repository root, use the regression entrypoint below with an existing
Python environment (`spiral-fitting/.venv` is the documented default, not a
prerequisite to create here). Include the built `vc_object_store` Python module
so adapter tests run rather than skip; `VC_OBJECT_STORE_BUILD` names an existing
store build. Service tests are migration coverage, not evidence that the
currently unported service works:

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting:$VC_OBJECT_STORE_BUILD/python spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_spiral_service_v2.py \
  spiral-fitting/tests/test_revisioned_runtime.py \
  spiral-fitting/tests/test_checkpoint_load.py \
  spiral-fitting/tests/test_service_events.py \
  spiral-fitting/tests/test_object_store_startup.py \
  spiral-fitting/tests/test_object_store_inputs.py \
  spiral-fitting/tests/test_object_store_materialize.py \
  spiral-fitting/tests/test_object_store_fibers.py \
  spiral-fitting/tests/test_object_store_overlaps.py \
  spiral-fitting/tests/test_config.py \
  spiral-fitting/tests/test_input_toggles.py \
  spiral-fitting/tests/test_run_boundary_settings.py
```

`test_revisioned_runtime.py` mocks the removed `prepare_input_changes`; stage 1
rewrites or deletes it with the input-batch paths. The old file-input service
behavior is covered by `test_service_editing.py`, `test_service_editing_http.py`,
`test_input_catalog.py`, `test_input_snapshot.py` and `test_workspace_cleanup.py`
(not in the command above); use them, at the pre-port checkout, as baseline
sources.

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
