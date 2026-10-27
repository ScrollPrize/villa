# Object-store inputs for the Spiral service and VC3D workspace

## Scope and dependencies

Companion to [OBJECT_STORE_CLI_PLAN.md](OBJECT_STORE_CLI_PLAN.md). CLI steps
11–12 are complete as of 2026-10-26: cleanup landed in `84757fd9d`, acceptance
and documentation in `cb0e796d8`, and review fixes in `ebc43f070`. Normal
sources are then addressed by array path rather than multiscales index
(`89abe9606`, `b793e0b88`, superseding `743d6d6d2`). Stage 1 can now begin.
This document is a plan, not a claim that the service already supports that port:
the current service cannot start a fit against the store-only `FitContext`
(it still passes the removed `input_bindings` constructor argument).

The CLI acceptance notes record a real-data, single-GPU 1,000-step fit with no
patch/fiber/PCL files, finite losses, complete provenance and matching no-refresh
counts/digest. They also record 19 unrelated pre-existing suite failures; real-data
two-rank fitting and changed-input resume remain unrun (CPU startup/resume tests
cover those contracts). These limits do not replace the service acceptance below.
That run predates the array-path normal addressing. The input digest excludes
link-refresh provenance and the same arrays should be read, but before stage 1
relies on it, rerun a short CLI fit with link refresh on the acceptance dataset
and confirm unchanged link counts and digest.
Use the README's corrected ingest frames for that dataset: patches/PCLs at scale
1, version-3 fibers at scale 4, with the documented point-ID and dangling-branch
repairs. The earlier step-10 fiber ingest used the wrong scale.

Port both `spiral-fitting`'s persistent HTTP service and the Spiral workspace in
`volume-cartographer/apps/VC3D`. Deliver in three stages:

1. A working service/workspace using frozen object-store inputs, without live
   input updates.
2. Store-backed editing and live application of changed inputs at safe fit
   boundaries.
3. Optional use of one authoritative database over SSH, using the existing
   object-store hub infrastructure. This stage is a sketch requiring further design.

These stages port fit **inputs** only. Spiral outputs — checkpoints, previews,
diagnostics, flattened/exported surfaces — stay service-owned files; nothing
writes fit results into the object DB. The only DB writes are detector link
refreshes (and, in stage 2, input edits). Storing fit results as store objects
is later work. The proposed launch flag is unrelated to
`lasagna/fit_service.py --object-store-dir` (a content-addressed artifact
directory).

For stages 1–2, assume the same logical store is available on client and service,
with matching UUIDs and revisions. Local paths can differ. Provisioning or
replicating that store is outside these stages; do not invent a transfer protocol.
Independent copies must be brought into agreement externally before an input
generation can be accepted. Matching paths or object names do not prove agreement.

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
- Reuse `ObjectStoreCoordinates.hpp`, store revision/annotation APIs and
  `ObjectStoreEvents.hpp`. UUIDs, stable point/control IDs and revision pins
  replace file aliases and document-array positions.

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
  inputs, not just filenames or a nominal store identity. Verify local display
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
  cannot start a fit, so the baseline comes from a checkout before `84757fd9d`
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
  errors; local and remote profiles with different paths to equivalent inputs.
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

Planned as one commit each on `spiral-object-store` after `b793e0b88`. Steps
2–10 change the service/VC3D protocol and land together (one `API_VERSION` /
`kApiVersion` bump, in step 3 and step 8); step 12 deletes old code only after
step 11's comparisons pass. Keep service and VC3D commits separate and include
focused regression tests with each functional commit. Before step 1, rerun the
CLI link-refresh check described under "Scope and dependencies".

Findings from the current tree that the steps build on (HEAD `b793e0b88`):

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
   (step 10 made the store mandatory; confirm the service still fits there),
   with file inputs matching the acceptance ingest, record:
   - display geometry: for a fixed set of patches (including invalid
     vertices), PCL collections (with point IDs) and fibers, the viewer-frame
     coordinates the workspace displays. Use a small dump entrypoint that step
     9 reuses against the store path;
   - service lifecycle transcripts for initialize → run → stop → preview →
     save → reconnect → load checkpoint: state/event sequences and the
     preview manifest and checkpoint provenance keys, from one real short
     session and the mocked-runtime `test_spiral_service_v2.py` harness.
   Commit the fixtures under `tests/data/` with a note naming the commit,
   commands and dataset. No production change.
2. **Service: bind the object store at launch.** Add required `--object-store`
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
3. **Service: replace the editing workspace with a session lease.** Build a
   lease from `WorkspaceLease`, `MutationCoordinator` and `workspace_storage`
   that keeps the token gate, `editing_lifecycle` command idempotency and
   teardown, but holds no input catalog and snapshots nothing at claim.
   `editing_lifecycle` no longer calls `refresh_clean`/`replay_resident`.
   Input-mutation routes (`input-changes`, `apply-inputs`, `commit-inputs`,
   `resolve-input`, `discard-inputs`, `input-content`, `input-catalog`,
   `input-commands`) answer 409 with a stable `inputs_frozen` code naming the
   full-input rebuild. Uploads accept only `checkpoint`. Bump `API_VERSION`.
   The old modules stay in the tree, unreferenced, until step 12.
4. **Service: torch-free discovery of store inputs.** Drop patches, fibers
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
5. **Runtime: build `FitContext` from the bound store.** Remove
   `input_bindings` from `create_session` through `_fit_main`; `FitContext`
   loads inputs in `load_host_inputs` through `load_fit_inputs()` (rank-zero
   refresh, failure aggregation, manifest agreement). Route the adapter's log
   and `LinkRefreshError`/`ObjectStoreInputError` reasons (including committed
   detector counts from `refresh_report()`) into session progress, events and
   the error state, rather than stdout only. Write `object_store_inputs.json`
   on the interactive path too. `apply_input_changes` (single and
   distributed) refuses before reserving a boundary. Test on CPU with the
   device stages stubbed against `tests/object_store_fixture.py`: the context
   receives the bound path, a model-only rebuild keeps `object_store_inputs`
   and does not refresh, and a two-rank startup refreshes once, propagates a
   failure and never starts partially (reuse the startup tests' distributed
   harness).
6. **Service: input provenance in status and artifacts.** Replace the status
   `input_manifest` file listing with the context's object-store manifest and
   digest, keeping external paths (shell, lasagna, tracks, checkpoint). Add the
   digest to autosave metadata (`write_autosave_metadata`), preview manifests,
   `PublishedPreviewIndex.record`, and the preview-restore matching in
   `_note_checkpoint_state`/`_restore_published_preview`. Remove the dataset-file
   PCL display artifacts (`_publish_pcl_artifact_locked`, the
   `*_winding_artifact` status keys): VC3D reads PCLs pinned to the manifest
   (step 9), so no unpinned copy remains.
7. **Service: checkpoints, autosave and rebuild against frozen inputs.**
   `_checkpoint_refusal` offers a full-input rebuild, or names
   `object_store_allow_changed_inputs`, for a manifest mismatch, and refuses
   a legacy checkpoint without `input_manifest.object_store` precisely.
   `_rebuild_stage_locked` no longer compares input paths. Add an explicit
   full-input rebuild (`POST /session/rebuild` with `inputs: "reload"`) that
   forces stage "all" to pick up new store revisions under an unchanged
   config. Preserve all-rank preflight. Test the refusal, the forced reload
   and that the `OBJECT_STORE_PCL_KEYS`/role toggles stay rebuild-only.
8. **VC3D: protocol, profiles and store binding.** Bump `kApiVersion`.
   Replace the editing claim/release calls with the session lease, and drop
   `pcl_inputs`, `paths.pcls` and input path rows from `SpiralPanel`'s request
   and `prepareSessionRequest`. Local auto-launch passes `--object-store` with
   the attached `CState::objectStoreSession()` store and restarts on an
   attachment change. A remote profile binds to the advertised store; both
   compare the local attachment's store UUID with the advertised one and
   refuse to present local data as the fitted inputs on a mismatch.
   `SpiralReloadComparison` compares the selection config instead of input
   paths and knows the input-reload rebuild. Old saved `form/pcls` and input
   path settings are ignored with a note, not mapped to store objects; brush
   working copies under `spiral-working/` are left on disk.
9. **VC3D: pinned store reads for input overlays.** Add a shared helper
   beside `ObjectStoreArtifacts.hpp` that reads patches, PCL collections and
   fibers at the revisions the status manifest pins (`RevisionSelection`,
   `readGeometryData`, aligned annotations, `fiberNetworkJson(selection)`).
   Convert them with `ObjectStoreCoordinates.hpp`, keeping invalid patch
   vertices and stable point IDs. `fiberTransportPaths` moves onto it rather
   than keeping a second current-heads export. Replace `loadInputSurfaces`'
   tifxyz reads and the PCL artifact overlay. Key surfaces and colours by
   UUID, not directory name. `mapServicePath` remains only for external
   inputs (shell, lasagna). A revision missing locally marks the overlay
   mismatched rather than falling back to current heads. Register a
   `spiral_store_inputs` ctest on a fixture store and compare against step
   1's coordinates with the same dump entrypoint.
10. **VC3D: disable input editing; show bindings and freshness.** Disable
    brush-to-file staging (`finalizeBrushPaint`), PCL replacement and new-PCL
    staging, row Add/Commit/Remove/Restore/conflict actions,
    `addPatchToCurrentFit`, `addFibersToCurrentFit`/`stageFiberInputs` and
    managed patch copies, each explaining that input changes need a
    full-input rebuild until stage 2. Keep the in-memory brush and point
    editors for stage 2. Show service connection, store binding, selection
    counts, manifest digest, resolved volume, and input freshness: store
    events touching selected objects (`ObjectStoreEvents.hpp`) mark the
    overlays and fit stale without reloading them. Previews show their
    digest; a preview from another digest is labelled as such.
11. **Acceptance before deletion.** Run step 1's comparisons against the
    store-backed service and workspace and explain expected differences.
    Check that the CLI and the service produce the same digest and
    constraint counts for one config. Cover empty selectors, metadata winding
    roles, non-unit scales, rejected/stale links and approved-only mode in
    service tests. Record commands, fixtures and results here.
12. **Remove obsolete file-input code.** Python: `EditingWorkspace`, the
    input-specific `input_workspace` classes, `service_uploads` input
    validators, the input entries of `FIT_INPUT_CATALOG` and `PCL_ROLE_FILES`,
    the runtime input-batch path (`_input_batches`, reserve/prepare/finish,
    `_run_input_batch`, `_apply_input_changes_at_boundary`), the rejecting
    routes, and their tests (`test_service_editing*.py`,
    `revisioned_client_service.py`, input parts of `test_input_catalog.py`,
    `test_workspace_cleanup.py` and `test_revisioned_runtime.py`). Stage 2's
    candidate generations build on the shared command barriers, not on this
    path. VC3D: the staging/upload half of `SpiralServiceManagerInputs.cpp`,
    `SpiralInputCopy`, input-draft upload state, the dead
    `*UploadFinished`/`pclCommitConflict` signals, `spiral_input_workflow` and
    the `check_spiral_*` scripts tied to removed code. Keep step 1's fixtures,
    `input_snapshot`, `PublicationTransaction` if still used, and the stage 2
    editors. Audit remaining references, then rerun the store-backed tests
    with no legacy inputs present.
13. **Docs and real-data acceptance.** Document service launch with
    `--object-store`, local and remote profiles, selection display, frozen
    inputs and the input-reload rebuild in README.md. Run a short fit through
    the service on the ingested real dataset, inspect it in VC3D, and record
    exact commands, config, revisions, counts, finite losses and
    checkpoint/preview provenance. This is the stage 2 gate.

Stage 2 gets its own sequence after the stage 1 gate; it depends on step 3's
lease, step 9's pinned reads and step 10's freshness tracking.

## Stage 2 — store-backed edits and live input updates

### 2.1 Establish edit and apply semantics

- Restore patch brush, PCL and fiber editing through the existing store session,
  typed geometry/annotation APIs and revision-checked transactions. Extract
  shared editor behavior where necessary rather than copy private implementations.
  Preserve metadata roles, stable IDs, coordinate declarations and branch links.
- A saved edit is a durable DB revision. Applying that revision to the running
  fit is a separate operation. UI labels must distinguish unsaved editor drafts,
  saved-but-not-applied data and the active fit generation. Discarding a local
  draft does not roll back committed DB history; undo is a new checked edit.
  Do not recreate the old filesystem workspace's Apply/Commit model accidentally.
- Initially offer explicit Apply pending inputs; store change events mark the
  session stale and coalesce notifications. Optional automatic application can
  use the same operation later. Ordinary DB changes must not mutate an active
  runtime or reinterpret a pinned preview underneath an edit.
- Carry expected revisions, active generation and an idempotent command ID in
  apply requests. When client/service copies differ, reject and report the
  mismatch; stage 2 does not synchronize databases.

### 2.2 Prepare and install a complete new input generation

- At an acknowledged iteration boundary, pause every rank. Coordinate writers
  across refresh and snapshot capture: extend the shared startup contract to
  enforce revision checks/retry or an appropriate write exclusion mechanism.
  Establish how non-Spiral editors participate before enabling live updates;
  the old file-workspace owner token does not lock the database.
- Refresh the resolved selection through the store detectors, then build a
  complete candidate snapshot and materialization through the shared adapter.
  Include metadata, approvals, deletion, relationships and selector membership
  changes, not just changed XYZ arrays. Full rematerialization is the initial
  correctness path; incremental updates are a later measured optimization.
- Reuse runtime prepare/install barriers, replacing their file-batch payloads.
  All ranks validate and agree on the candidate digest before installation.
  On preparation failure, discard candidate runtime state and retain the old
  active generation. Detector commits may already exist in the DB and must be
  reported; DB persistence and fit installation are distinct transactions.
  If installation fails after ranks diverge, stop the session and require
  recovery rather than resume a mixture of input generations.
- Specify which model and optimizer state can survive each update. Preserve it
  only with validated mappings; changes affecting atlas topology, model domain,
  parameter shapes or frame require an explicit rebuild. Never reuse runtime
  indices based on array position. Report any reset before accepting the update.
- Version previews, diagnostics, checkpoints, caches and input overlays with
  session/input generation and digest. A completed old preview may still be
  viewed as old; it cannot overwrite the active generation's display or authorize
  an edit against newer geometry. On reconnect, recover status before replaying
  commands; event cursors alone are not durable applied-state evidence.

### Stage 2 acceptance

- Before removing remaining non-store editing/apply paths, regression-test
  their replacement against captured pre-port behavior for edits, conflicts,
  apply failures, reconnect and preview freshness. Explicitly account for the
  new DB-save versus fit-apply semantics. Once comparisons pass, delete the
  replaced implementations and their unused callers, routes, options and tests;
  retain reference fixtures and rerun the store-backed regressions.
- Exercise create/edit/delete for patches, PCLs and fibers; annotation/approval
  changes; ordinary versus adjacent branches; empty selections and changed tags.
- Test edit conflicts, edits during preparation, repeated Apply, disconnection
  after commit/installation, delayed previews and failed detector refresh.
- Two-rank tests cover matching installation, one-rank preparation failure and
  fatal partial installation. Check model/optimizer preservation or explicit
  rebuild behavior, then resume from a checkpoint of the applied generation.
- Real-data VC3D smoke: edit → save to DB → apply at boundary → continue fitting
  → view new preview. Confirm old previews remain correctly identified and a
  failed apply leaves the previous resident fit usable when preparation failed.

## Stage 3 — optional shared database over SSH (sketch)

The store already has a [hosted hub and replicas](../vc_object_store/docs/README.md#hosted-hub-and-replicas),
including `sftp://` / `ssh://` addresses, an SFTP SQLite VFS, leases, conflict
handling and `ObjectStoreSession` background synchronization. Prefer integrating
this rather than inventing another database server or syncing live SQLite files.
An existing hub is deliberately not opened as an ordinary working store.

Candidate design: client and service use local working replicas of one
authoritative SSH hub. Expose this as an optional shared-store profile, retaining
the explicitly provisioned local-store mode. The existing HTTP SSH tunnel still
carries fit commands/artifacts; it is separate from object-store synchronization.

Before an apply, publish the client's committed edits, bring the service replica
to an acknowledged hub version, refresh on the service, publish resulting links,
and establish the exact manifest both sides will display/use. Reuse store lease
and conflict rules; decide ownership of detector writes and ordering with VC3D's
background sync. Pin the running fit independently of later hub changes.

This needs a fuller plan before implementation. Resolve replica lifecycle,
Python hub/session API coverage, lease scope for automatic relationships,
consistent snapshot capture during sync, concurrent authors, conflict UI,
offline behavior, interrupted/ambiguous pushes, credentials and reconnects.
Also determine whether the intended “shared copy” really requires direct remote
working access rather than the existing hub/replica model; that would be a
separate architectural change, not a path substitution. Validate on a real SSH
host, including interrupted connections and concurrent client/service writes.

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
