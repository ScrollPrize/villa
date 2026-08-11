# Spiral interactive service refactor plan

Goal: a clean, maintainable interactive fitting service. The root problem is
that `fit_spiral.py` was wrapped, not refactored: `main()` is a ~2500-line
function whose operations are nested closures, so `spiral_runtime.py`
configures it by monkeypatching ~20 module globals (+ env vars + a disabled
wandb run) and controls it via five closures passed through `on_ready`.
Everything above it (stringly-typed states duplicated in three places, shadow
flags and generation counters in `spiral_service.py`, automatic-preview
detours, checkpoint load fused into session construction, and inconsistent
output channels) is compensation for that coupling.

Target architecture:

```
fit_spiral.py      FitContext: load_host_inputs / build_device_state / step /
                   apply_config / load_checkpoint / save_checkpoint /
                   export_preview / close.
                   Headless CLI = thin driver over the same object.
spiral_runtime.py  The sole execution owner of a FitContext (one fitter thread
                   per rank), typed commands, one enum session state machine,
                   and explicit DDP coordination/fail-stop behavior.
spiral_service.py  HTTP/auth and lifecycle orchestration. Artifacts, uploads,
                   and Lasagna publishing live in their own modules. Dataset,
                   output, and cache locations belong to the service instance.
events             One structured event stream (log | progress | metric |
                   error), plus a durable status snapshot for reconnects.
VC3D               A connection profile identifies a service instance and its
                   dataset/output roots. Locally, the user selects the dataset
                   in the connection UI before VC3D launches the service.
```

The work is deliberately ordered in three phases. **PR 1** introduces
FitContext and establishes its ownership/execution boundaries while retaining
the existing external protocol.
**PR 2** cleans configuration and service boundaries and makes independently
versioned client-contract changes. **PR 3** introduces the state/command model,
hardens DDP, adds safe session verbs, and only then switches to an always-loaded
service. A commit should remain independently usable unless it explicitly bumps
the coordinated service/VC3D API.

Workflow: prepare the three PRs as a stack using `gh stack`, locally only.
Executing this plan must not push branches/commits to origin or submit the
PRs; review and submission happen separately, at the author's initiative.

---

## PR 1 — FitContext and execution ownership

This PR establishes one owner for fitter state and resources before redesigning
configuration or service behavior. The important invariants are that all mutable
fit state moves together, Torch/CUDA work remains on the owning fitter thread,
and CLI and interactive drivers use the same context rather than parallel setup
paths. Configuration schema, HTTP behavior, and checkpoint policy change only in
later phases.

**Commit 0: golden-run characterization harness.**
A test that runs a short headless fit with fixed seeds on the existing
Scroll1 (PHercParis4) dataset restricted to a limited z-range
(e.g. `z_begin=10_000`, `z_end=11_000`), which keeps the run fast. The GPU
paths are not deterministic enough for bit-exact assertions, so the harness
does not attempt them. Instead it asserts on:

- structural invariants that must be exactly preserved: checkpoint key sets
  and tensor shapes, iteration counts, config round-trip, and the sequence
  of host-side RNG-order-sensitive decisions where they are cheap to record;
- numeric behavior within tolerance bands: per-loss-family loss traces at a
  few fixed iterations and the final losses, with tolerances calibrated by
  running the *unmodified* code several times first and setting bands from
  the observed run-to-run spread (plus margin).

This is the gate for every commit in this PR: hoisting is "purely
mechanical" only if this proves it within those bands. The harness does not
go away when the PR lands — it stays as the permanent end-to-end regression
test for the fit path; only its role changes, from refactoring gate to
regression guard (to be re-baselined explicitly whenever a numerics change
is intentional, which is out of scope for this plan).

The harness runs locally on the development machine, which has the dataset
and a GPU; it is not a CI job. Reviewing a PR 1 commit means running it on
that machine against the commit under review.

**Commit 1: establish `FitContext` ownership and lifecycle.**
Create `FitContext` around the current `main()` locals. Document four ownership
classes instead of treating setup as a simple CPU/GPU split:

- immutable dataset/path descriptors;
- host-prepared inputs and caches;
- session-owned CUDA/device resources;
- model, optimiser, scheduler, RNG, and iteration state.

`FitContext.close()` owns resource release, replacing
`release_interactive_resources`' module globals. All methods that touch Torch or
CUDA are executed only by the fitter thread for that rank; HTTP/coordinator
threads continue to submit commands rather than calling the context directly.

**Commit 2: extract host-side setup.**
Move the genuinely host-side setup closures and blocks to methods, converting
captured locals to attributes one at a time: patch/PCL loading, trusted-geometry
construction, track source loading, sampling-strata construction, and reusable
host caches. This includes `PatchGpuAtlas` construction despite its misleading
name and `device='cuda'` argument: its packed coordinates/indices are CPU
tensors, its native sampling atlas is ordinary host C++ storage, and `device`
only selects where each lookup's small interpolated result is delivered. Rename
it to `PatchAtlas` (not `PatchSamplingAtlas` — that name is already taken by the
native class it wraps as `self.sampling_atlas`) and expose this work through
`load_host_inputs()`.

`load_host_inputs()` is also the replacement for
`main(load_only_patches_and_point_collections=True)`: analysis tools that
today monkeypatch `fs.*` globals and early-return out of `main()` to reuse
the host-side loading (`find_inconsistent_windings.py`, `phase_tuning.py`,
and anything else ingesting spirals the same way) instead construct a
context, call `load_host_inputs()`, and read the loaded
patches/point-collections/etc. as attributes — no device state, no
monkeypatching, no special-case flag threaded through `main()`. Loaded host
inputs must therefore remain inspectable attributes, not locals of a setup
method. Port these tools in this PR (or at latest when the flag's globals
disappear) so they never break on an intermediate commit; they keep working
throughout, since they exercise only host-side loading plus
checkpoint-based model reconstruction (`SpiralAndTransform` +
`checkpoint_io`), both of which stay importable without a session.

**Commit 3: extract device/session setup.**
Move actual CUDA allocations, resident Lasagna stores, prepared device track
structures, model/optimiser/scheduler construction, and other device-dependent
setup behind `build_device_state()`. Patch-atlas lookup/upload is a runtime
host-to-device boundary, not evidence that the atlas itself owns CUDA storage.
Keep the host-side rebuilding/filtering in the host-input layer and bind its
lookup results to the session device when batches are produced.

**Commit 4: expose fitter operations behind compatibility adapters.**
Move `checkpoint_payload`, `save_model_to`, `load_model`,
`export_interactive_preview`, `incorporate_interactive_inputs`, and
`configure_interactive_run` to context methods. Keep the current `on_ready`
closure handoff temporarily as a thin adapter so this commit changes neither
runtime scheduling nor the public service protocol.

`apply_config()` retains the existing impact analysis and rollback behavior in
this PR. Redesigning the dependency catalog happens in PR 2 so ownership and
dependency-policy changes do not land simultaneously.

**Commit 5: extract `step()`.**
Move one training-loop iteration to `FitContext.step(iteration)`, including the
forward families, backward handoff, collective, optimiser/scheduler update, and
metric construction. The surrounding driver retains the current
`wait_for_iteration` / `iteration_completed` sequencing.

**Commit 6: make the headless CLI a thin driver.**
The `__main__` path constructs the same context and drives it to the configured
horizon. Collapse duplicated DDP initialization and batch splitting into shared
helpers, without yet changing configuration sources or wandb behavior.

**Commit 7: remove the legacy closure protocol.**
Once both drivers own a real context, delete the `on_ready` five-closure handoff,
the reverse `interactive_driver.status()` calls from fitter internals, the
unused `fit_session.SpiralFitSession` declaration, `session_finished()`
implemented as `raise RuntimeError`, and the unreachable
`wait_for_iteration`-returns-False exit. Shutdown remains an explicit command /
exception handled by the fitter-thread owner.

The automatic restored-checkpoint preview currently fires from inside
`on_ready` itself. Deleting the closure handoff must not change that
protocol behavior in this PR: the runtime triggers the same preview itself
via the context's `export_preview` after the session reaches ready. Its
actual removal (explicit preview verbs) happens in PR 3.

---

## PR 2 — Configuration, event contract, and service boundaries

This phase makes the context explicit to configure and reduces service coupling.
Client-visible slices carry an `API_VERSION` bump and matching VC3D changes in
the same commit/PR; `/logs` remains exactly compatible until that bump.

**Commit 1: one explicit fit configuration.**
Introduce `FitConfig` (wrapping the existing `Config` catalog) and pass it to
`FitContext`. Replace module-global fitter mutation, `FIT_SPIRAL_*` fit controls,
`wandb.config`, and `losses.py`'s `cfg/z_begin/z_end` globals with explicit
context/config references. wandb becomes an optional headless logging sink, not
a structural source of configuration.

The spiral-ingesting analysis tools (`find_inconsistent_windings.py`,
`phase_tuning.py`) migrate to constructing a `FitConfig` explicitly in the
same commit that removes the globals they currently assign (`fs.cfg`,
`fs.z_begin`, path globals, `umbilicus_z_to_yx`), so `fit_spiral` never has
to keep dual config paths alive for them.

**Commit 2: versioned scroll specification.**
Add one unambiguous conventional filename (for example `spiral-scroll.json`)
with `schema_version`, strict unknown-key validation, and paths resolved relative
to the dataset root. Parse it into a frozen, torch-free `ScrollSpec` in
`fit_session.py`.

The file contains physical/dataset facts: `name`, `voxel_size_um`,
`spiral_outward_sense`, umbilicus `coordinate_scale`, `normal_zarr_group`,
`surf_sdt_zarr_group`, `lasagna_scale`, and allow-listed overrides for paths that
depart from directory conventions. Conventional paths need no entry.
`outward_sense` leaves the load request and VC3D fit controls because it is a
property of the scroll.

`z_begin` / `z_end` become `Config` keys, but their metadata must record all of
their effects: host filtering, dense-store coverage, count scaling, rendering,
and model/checkpoint-domain compatibility. They are not ordinary cheap
run-boundary settings.

Adding keys to `Config` breaks the current strict checkpoint compatibility
check (stored `cfg` key set must exactly equal the schema), which would
reject every pre-existing checkpoint. When a checkpoint's stored `cfg` lacks
`z_begin` / `z_end`, default them to the values from the service start-up
configuration and emit a warning naming the checkpoint and the assumed
values; all other key-set mismatches remain strict errors.

Deployment and presentation values stay outside the scroll file: output/cache
roots are service startup arguments; `run_tag` is per-run metadata with
safe-path-component validation; storage backend is service implementation
configuration; render scale belongs to preview/output configuration.

The headless CLI takes `--scroll-spec` or discovers the single conventional
file, rather than requiring edits to module globals.

**Commit 3: one declarative input/dependency catalog.**
Describe each input with its path kind, enabling predicates, host/device rebuild
scope, and checkpoint-domain relevance. Make request validation, run planning,
and `FitContext.apply_config()` consume this catalog instead of maintaining
three drifting maps and the `outer_shell` special case.

**Commit 4: bind dataset/output/cache to the service connection.**
Make `--dataset` and `--output` required service arguments. `--output` must
resolve outside the dataset root. Add an explicit `--cache` argument or one
documented user-cache default outside the dataset; do not silently lose the
current cache derivation when removing it from `resolve_dataset_root`.

For a local connection, the dataset (and optionally output/cache defaults) moves
into VC3D's connection panel. VC3D launches its owned service with those values,
so selecting a different dataset creates/restarts a different bound service
instance. Remote connection profiles display the immutable advertised paths.
There is therefore no need for a general service-side "browse arbitrary dataset"
mode: resolution happens once at startup and `/dataset` advertises the result.

Generated state—named-session directories, run directories, autosaves,
previews, `.spiral-ephemeral`, staging, and uploaded checkpoints—lives under
`--output`. Dataset resolution describes inputs only. Preserve the named-session
exclusive lease under the corresponding output namespace.

**Commit 5: structured events without weakening status.**
Add `/events` with records containing at least `sequence`, timestamp, severity,
kind, source/rank, session generation, operation/command ID, text, and payload.
Kinds are `log`, `progress`, `metric`, and `error`. Define cursor-overrun and
reconnect behavior, rate-limit/coalesce progress and per-iteration metrics, and
route child-rank events explicitly to the parent process.

The durable `/session/status` snapshot continues to carry current state,
iterations, applied configuration, input manifest, latest metrics/error,
warnings, and current artifact/operation references. It stops synthesizing ETA
and presentation text; clients derive those from raw progress timestamps. An
event stream is bounded history, not a replacement for reconnect state.

Avoid duplicates: a structured progress/metric event must not also re-enter as
a tee-captured log record. Keep `/logs` as the old-schema compatibility endpoint
for the advertised compatibility window, rather than returning event-shaped
records under the old name.

**Commit 6: quiet access logging and update VC3D.**
Override `SpiralHandler.log_request` to suppress successful polling requests at
the source. Remove the string-matching filter from `ServiceLogBuffer`.

With the coordinated API bump, VC3D uses one event subscriber: the panel can
interleave all event kinds, popups are reserved for error severity, and popup
content also appears in the panel. Local-process stdout remains useful for
startup and terminal diagnostics but does not duplicate structured events.

**Commit 7: split service responsibilities.**
Move `ArtifactRegistry`/`Artifact` to `service_artifacts.py`; upload transfer,
validation, and publication to `service_uploads.py`; and flattened-preview work
to a `LasagnaPublisher` in `lasagna_publish.py`. Give the extracted components
small explicit interfaces back to the lifecycle orchestrator rather than moving
methods that still reach freely into `ServiceState`.

Lasagna publication owns one progress/event path, replacing the nested stage
callback, direct preview-status mutations, and log-line parsing.

**Commit 8: routing and mutation semantics.**
Replace `_dispatch` with a declarative route table. Specify idempotency per
operation rather than a single `needs_dedup` boolean: command-ID replay suits
logical mutations, upload PUT retries are content/offset/hash operations, and
finalize is naturally idempotent per upload ID.

**Ephemeral-input bookkeeping.**
Do not force persistence and incorporation into one linear enum. They are
independent: an input may be committed before it is incorporated into the live
fit. Keep an explicit incorporation state (`pending | incorporated | error`)
plus persistence state (`ephemeral | committed`), preferably in a typed record.
Centralize transition/cleanup helpers and validate commits under the file lock,
but treat this as local cleanup rather than a prerequisite for the architecture.
Expose committed-but-not-incorporated inputs in status.

---

## PR 3 — State machine, safe verbs, DDP hardening, always-loaded lifecycle

This phase changes session behavior. DDP coordination and fail-stop behavior
land before commands that mutate resident model state.

**Commit 1: one authoritative session state machine and typed command queue.**
Use `SessionState`: `Loading, Idle, Running, Saving, ExportingPreview, Error,
Closing`. Ready and Paused merge into `Idle`; `completed_iterations` records
whether work has run. One transition function owned by the runtime session is
authoritative. `ServiceState` and the distributed proxy observe it rather than
independently transitioning copies. Operation phase/progress remains separate
from lifecycle state where appropriate.

Replace `_idle_actions` tuples with command dataclasses carrying command ID,
session generation, expected iteration/config revision, completion result, and
cancellation/error state.

**Commit 2: DDP command barriers and fail-stop behavior.**
Every all-rank command carries a monotonically increasing epoch. All ranks
validate the same epoch, command kind, configuration revision, and pending count
at a safe boundary before entering another training step.

Do not model distributed aggregation as a generic "minimum" state. Collective
states become visible only when all participating ranks report the same command
epoch/state; rank-0-only publication and saving are explicit coordinator
sub-operations.

Generation checks do not catch exceptions after a rank enters a step. Add a
parent watchdog: any worker error, command timeout, or unexpected exit marks the
session failed, terminates/aborts sibling workers and the process group, and
returns a diagnosable bounded-time error instead of waiting for the 60-minute
NCCL timeout.

Pass rank/world-size explicitly through runtime/helper objects; environment
variables remain only at the torch rendezvous boundary. Replace
`_free_loopback_port`'s release-before-use race with a rendezvous that owns its
endpoint for the required lifetime (for example a file-store rendezvous in a
service-owned temporary directory, or an explicitly created TCPStore). Merely
having rank 0 discover and release another free port is not sufficient.

**Commit 3: strict in-session `load_checkpoint`.**
Add `POST /session/load-checkpoint`, valid in `Idle`. Prefer strict refusal over
implicitly rebuilding the resident model: an in-session load is allowed only
when the checkpoint exactly matches the live model/domain and all structural
invariants. A different model domain/config requires the explicit rebuild/new-fit
path, where teardown is visible as `Loading` rather than hidden inside a load
verb.

Loading is two-phase and atomic:

1. inspect on CPU and validate schema, scroll/dataset identity, exact model
   z-domain and tensor geometry, structural config, Lasagna/SDT identities,
   model keys, and optimiser/scheduler compatibility on every rank;
2. only after every rank accepts, apply model/optimiser/scheduler/RNG state on
   every rank and publish the new checkpoint iteration/config revision.

On failure before application, the live session is unchanged. A failure during
application is fatal to that distributed session rather than leaving a partially
loaded optimiser. Restore `completed_iterations` from the checkpoint and realign
the LR schedule to that durable step; do not reset it to zero. Initial autosave
restore uses the same preflight/application implementation after constructing an
exactly compatible service-domain model.

**Commit 4: explicit preview and save behavior.**
Add `POST /session/export-preview`. Remove the automatic checkpoint-resume
preview and hard-wired pause preview. Autosave-on-pause becomes a run-request
flag, default on. VC3D may preserve today's UX by explicitly requesting a
preview after checkpoint load or pause; inspecting a checkpoint no longer needs
a one-iteration priming fit.

**Commit 5: eager, always-loaded session.**
After startup dataset/spec validation, create the runtime asynchronously in
`Loading`. The HTTP service remains responsive while CUDA/model construction is
in progress and while in `Error`; `/health`, `/dataset`, `/configuration`,
`/events`, status, checkpoint upload, and recovery remain available.

For a named service output, select an autosave through explicit metadata rather
than filename ordering: require matching session namespace, validate container
and checkpoint identity, and define behavior for corrupt/incompatible candidates.
If a selected startup autosave cannot be loaded, enter `Error` with the cause;
an explicit rebuild-with-defaults command recovers.

There is no `Empty` state. `DELETE /session` is removed rather than redefined to
mean something surprising; use explicit `reload-defaults` / `rebuild` commands.
A rebuild is `Idle|Error -> Loading` and is the only path that may replace model
domain/structural configuration. Once this is authoritative, delete the old
`replacing`, `replacement_old_session_released`, and `pending_revision_target`
machinery.

**Commit 6: simplify generations and preview publication state.**
Retain counters only where they represent distinct public concurrency semantics:
service/process identity, session identity, status revision, configuration/input
revision, and command replay may not be interchangeable. Remove genuinely
redundant preview-generation shadow fields by making `LasagnaPublisher` own one
publication record. Do not promise an arbitrary reduction to two counters until
the reconnect, stale-plan, artifact, and command-dedup consumers have been
enumerated.

**Commit 7 (optional): reuse host-prepared inputs across rebuilds.**
Attach only immutable/cache-safe host data to `DatasetContext`, including the
host-resident patch sampling atlas where its filtering/config key matches.
Actual CUDA tensors, prepared device structures, and resident CUDA-backed stores
remain session-owned. Cache keys include path/content fingerprint, scroll spec,
z-range, and every preparation-affecting config value. Local commit notifications
alone are insufficient because another service/process can modify the dataset.
This optimization is not required for the always-loaded lifecycle.

---

## Explicitly out of scope

- SSH tunnel implementation changes.
- Changing the upload → incorporate → commit user workflow.
- Lasagna subprocess launch/port scraping; it remains contained inside
  `LasagnaPublisher`.
- Artifact and upload hashing algorithms.
- Loss-function or intended numerical behavior changes.
- Transparent checkpoint-driven model-domain rebuilding inside
  `load_checkpoint`; domain changes use the explicit rebuild path.

## Principal risks and gates

- PR 1 touches research-critical execution order. Preserve fitter-thread
  affinity, RNG/iteration ordering, cleanup ownership, and existing external
  CLI/service behavior while the new context becomes authoritative; the
  golden-run harness (PR 1 commit 0) is the mechanism that detects a
  violation, not just a stated intention.
- Each event/config/API change must update VC3D with the same advertised
  `API_VERSION`; do not defer a PR 2 contract change to PR 3's bump.
- DDP fail-stop and command barriers precede distributed checkpoint mutation.
- Checkpoint preflight must be exhaustive and application atomic. Reject an
  uncertain compatibility case rather than partially loading live state.
- The local connection UI must supply dataset/output before launching its owned
  service; otherwise mandatory service-lifetime dataset ownership would conflict
  with the current auto-launch flow.
- `find_inconsistent_windings.py` and `phase_tuning.py` consume `fit_spiral`
  as a library (host-input loading + checkpoint model reconstruction) and
  must work at every commit; each commit that moves or renames something
  they touch ports them in the same commit.
