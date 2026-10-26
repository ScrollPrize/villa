# Object-store inputs for the Spiral service and VC3D workspace

## Scope and dependencies

Companion to [OBJECT_STORE_CLI_PLAN.md](OBJECT_STORE_CLI_PLAN.md). CLI steps
11–12 are complete as of 2026-10-26: cleanup landed in `84757fd9d`, acceptance
and documentation in `cb0e796d8`, and review fixes in `ebc43f070` (following
the lasagna-level fix `743d6d6d2`). Stage 1 can now begin.
This document is a plan, not a claim that the service already supports that port:
the current service cannot start a fit against the store-only `FitContext`.

The CLI acceptance notes record a real-data, single-GPU 1,000-step fit with no
patch/fiber/PCL files, finite losses, complete provenance and matching no-refresh
counts/digest. They also record 19 unrelated pre-existing suite failures; real-data
two-rank fitting and changed-input resume remain unrun (CPU startup/resume tests
cover those contracts). These limits do not replace the service acceptance below.
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
- Advertise the binding, supported capabilities, selectable objects and frame
  metadata through dataset discovery without loading Torch or refreshing links.
  The client supplies its local attachment separately. A local auto-launch
  profile can use its attached DB; a remote profile binds to the advertised
  store without assuming that a client path is valid on the service host.
- Use the actual CLI config names: `object_store_{patch,pcl,fiber}_{uuids,tags}`,
  refresh/approved-only options, `fit_volume_uuid`, normal-source settings and
  external frame settings. Preserve omitted versus explicitly empty selectors.
  Display the resolved UUID selection and counts, including excluded links.
  Reuse the adapter's lasagna normal-source resolution: `normal_zarr_group` is a
  dataset path, while the detector's `ct_level` is its multiscales level index
  (the acceptance dataset's group `4` is level `0`). Do not pass the group name
  as a level or duplicate that resolver in the service.
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
  workspace behavior against a recorded pre-port baseline on equivalent inputs:
  display geometry/coordinates, initialization, run/stop, previews, checkpoints,
  reconnect and resume. Record commands, fixtures and results; explain expected
  differences from the accepted CLI linking changes. Remove the replaced paths,
  obsolete routes, options and tests after these comparisons pass, then rerun
  store-backed acceptance with no legacy inputs available.
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
landed together. No new fitter/linker numerics are intended beyond the accepted
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
