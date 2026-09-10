# Plan: cooperative node resource manager and elastic Fiber3D training

Status: approved for implementation. All phases are required, in the order below.

## Current implementation and invariants

Reviewed merged baseline `6009bb7f4` on `fiber-training`, initially clean.
`fiber_trace_3d/train.py` owns loss statistics/global ratios, optimizer updates,
AMP, tests and snapshots. Its virtual-batch loop retains all microbatch graphs
until backward: useful as a mathematical oracle, not memory-bounded accumulation.
`sample_pipeline.py` owns canonical `VirtualBatchSampleIndex`, CPU shared slots,
pinned staging and CUDA preparation. Current batching assumes full local batches
and a fixed rank set; extract/extend these helpers instead of copying them.

Preserve `planning/specs.md` contracts: stream/data index distinction, sample-keyed
RNG, sampling/augmentation semantics, loss group weighting, test-only public
snapshots, best-by-test-loss and one writer. Dynamic mode rejects incompatible
batch-dependent models; never silently convert old BatchNorm checkpoints.

Correction to earlier discussion: legacy multi-branch positive-presence counts
can depend on prediction-selected nonempty branches. Counts for the reference
single-branch/conditioned paths are target-derived, but the general accumulation
design must handle prediction-dependent detached counts too.

## Phase 1 — reusable package, ownership and protocol draft

Create `gpu_manager/` with independent `pyproject.toml`, `src/villa_gpu_manager/`,
`tests/`, README, `docs/protocol-v1.md`, schemas and reference client example.
Proposed CLI: `villa-gpu-manager`. No Torch/VC3D/Lasagna imports in the core.

Mini playbook: Python/setuptools/pytest; Ubuntu and macOS amd64/arm64 Unix sockets;
CUDA integration on NVIDIA hosts. Standard-library JSON/socket/selectors service;
small process-inventory adapter using psutil and a driver-query GPU adapter that
does not initialize CUDA. Fake adapters/clocks support CPU-only tests. Install
only through the targeted workflow when authorized; none during planning.

One server runs in a user's foreground service process or a long-lived CPU
supervisor. No privileged daemon. Embedded server lifetime is independent of
its owner's training task: graceful shutdown drains clients or refuses while
they remain active. Automatic server migration/election and multi-node support
are outside the requested scope; server failure has defined recovery below.

Clients' CPU supervisors handle communication independently of GPU work.
Use nonblocking bounded socket handling so a slow client cannot freeze the
manager event loop. No Python thread is used to parallelize heavy training work.

### Protocol v1 — complete scope before coding

UTF-8 newline-delimited JSON over filesystem `AF_UNIX/SOCK_STREAM`; escaped
newlines in strings, partial reads/writes, maximum 1 MiB frame, no pickle/tensors.
Units: integer bytes, milliseconds, physical cores and GPU counts; stable UUIDs
for GPUs/tasks/server epochs. Reject invalid types, sizes, duplicate JSON keys,
nonfinite numbers and unknown operations.

Request envelope: `version: 1, id: <request UUID>, op, body`; after handshake
also `server_epoch` and task identity for task operations. Response echoes ID,
version, epoch and `ok`, with either body or structured error
(`code, message, details`). Hello negotiates version/capabilities before use.
Unknown required capabilities fail; optional diagnostics may be ignored.
Unknown constraint fields fail rather than silently weaken a request.

| Operation | Contract |
| --- | --- |
| `hello` | Negotiate versions/capabilities; return server epoch, inventory and peer identity. |
| `register` | Stable task UUID, label, root process identity and initial request; return task state and reconnect token. |
| `set_request` | Replace the complete resource/priority/exclusive request, with expected request revision. |
| `poll` | Last-seen generation and optional status; return complete directive/current state, including unchanged state. |
| `report` | Generation and phase: released, applied or failed; actual held resources and worker identities. |
| `set_paused` | Set/clear self-pause or controller pause; retain original request. |
| `set_override` | Atomic full allocation map, optional bound task, expected policy revision. |
| `clear_override` | Revert to automatic policy with expected policy revision. |
| `status` | Inventory, requests, target/granted/held allocations, reasons, phases, override and last contact. |
| `finish` | Terminal outcome after release; reconcile before forgetting resource ownership. |

Registration advertises `reconfigure_mode: full_release` for Fiber. A grant
change releases its complete old cohort, including devices in both old and new
allocations, before authorizing the complete new grant. Define this capability
in v1. Self-pause and controller-pause are separate latches: effective pause is
their union; a client cannot clear a controller's pause by resuming itself.

Poll replies deliver server changes, requests deliver client changes. No separate
push channel/event backlog. Work-step checks plus supervisor control polling
(default 1s) continue during startup, long steps, evaluation and pause; PID-tree
scan default 60s. Paused clients sleep between checks.

Retries: stable task UUID makes registration idempotent; request/override writes
use compare-and-set revisions; reports use generation+phase. Identical retries
succeed, contradictory repeats/stale revisions fail explicitly. Never reuse a
request ID with different contents. Reconnect retains task identity using token
and OS peer identity; task IDs cannot be resurrected by late messages. Define
bounded dedup retention and stale replies in conformance fixtures. Server epoch
changes on restart and rejects old-epoch mutations.

Reference Python client wraps these exact operations and supplies a context
manager example with polling, resize, pause, request replacement, failure cleanup
and finish. Manager clients and administrative CLI use the same protocol.

## Phase 2 — resources, deterministic policy and safe handoff

### Inventory and requests

Inventory: integer allocatable physical cores and GPU UUIDs with usable VRAM.
CUDA ordinals are local presentation only. Configure managed UUID allowlist,
CPU budget and driver/external-use reserves. Exclude nonparticipant GPUs.
Do not use currently free VRAM as the allocation ledger: promises count before
the recipient initializes. Unexpected inventory/use changes block affected new
grants and are reported.

Request: `max_cpu_cores, max_gpus, min_vram_bytes_per_gpu,
max_vram_bytes_per_gpu, allowed_gpu_uuids, priority, exclusive`.
Minimum VRAM is the minimum useful allocation, not permission to overcommit.
Use proportional initial CPU charging: for each allocated device with VRAM M,
f=M/max_vram_per_gpu; cores=ceil(max_cpu_cores * sum(f) / max_gpus).
Validate positive GPU ratios, minimum feasibility and integer rounding.
Manual allocation specifies explicit integer cores and device VRAM.

### Automatic scheduling (explicit interpretation of the request)

1. Exclude terminal or paused tasks. Larger integer priority wins, ties use
   stable registration order. Unsatisfiable minima remain visibly queued.
2. Highest-priority runnable exclusive wins over all non-exclusive tasks.
   Reserve the entire managed pool, granting only devices/capacity it can use
   within its request. Other tasks get zero; unused exclusive reservation is
   not handed to competitors.
3. Without exclusivity, place one minimum-useful GPU share per feasible task,
   preferring distinct GPUs; share a GPU's VRAM where needed. No compute-time
   fraction or performance-isolation guarantee accompanies a VRAM share.
4. Grow allocations in priority order toward requested maxima, enlarging VRAM
   shares then adding GPUs. Highest priority gets spare capacity until its max,
   then the next task. Retain feasible current UUIDs to reduce churn.
5. If minima cannot all fit, queue lower-priority tasks at zero rather than
   oversubscribe. No automatic time slicing/aging is introduced. Sustained
   higher-priority/exclusive demand may starve tasks; status must make it clear.

Manual override is an atomic full task-to-allocation map; omitted tasks get
zero. Validate host/device totals and request compatibility before accepting.
It supersedes automatic/exclusive policy, but cannot unpause explicitly paused
tasks. An accepted request replacement or pause that conflicts with the override
atomically clears it with an explicit `override_invalidated` reason and recomputes
automatic targets. Accepted changes must trigger withdrawal without waiting for
manual repair. `bound_task_id` clears the override
when that task finishes and releases, not on a transient disconnect or pause.
Task owners control their requests; same-user controllers may override all.
Cross-user controller privileges require explicit configuration.

### Handoff and liveness

Keep **held**, **target** and **granted** allocations distinct. A target is not
permission to use new resources. Pending grants already reserve capacity.

For a transfer, serialize the affected tasks' transition:

1. Send donors a generation-tagged release directive; Fiber releases its whole
   old GPU cohort. Continue charging old holdings.
2. Donors complete a safe work boundary, preserve CPU state, exit/join all GPU
   workers and acknowledge released for that generation.
3. Only then grant freed capacity to recipients; reserve it immediately. They
   initialize within the grant and acknowledge applied. Initialization failure
   must release before retry.

Unrelated tasks keep working; independent free capacity can be granted without
waiting for unrelated donors. Coalesce new requested targets while a transition
is pending, but never overwrite ownership/generations. For swaps, release every
donor before dependent grants. No client needs a new resource to release its old
one. Slow donors produce visible waiting states, not a blocked service thread.

For full-release clients even expansion or a VRAM change on the same UUIDs uses
release-to-zero then the entire new grant. During initialization keep the newest
target pending until applied/failed is acknowledged, then reconcile through the
next generation. Reports distinguish GPU release from draining CPU work; CPU
capacity is not reassigned before its corresponding acknowledgement.

Task identity includes root PID, process start time and host boot identity to
handle PID reuse. Verify peer UID/root ownership; record descendants and retain
their identities after reparenting. Supervisors report GPU children as additional
evidence; reject overlapping live task trees. Socket EOF, expired heartbeat or
root exit alone is not proof of release. Reclaim on a valid release report or
confirmed task/known GPU-descendant death. Inaccessible or ambiguous process
state stays reserved/quarantined and is reported. Periodic tree scans cannot
capture every short-lived-parent race: cooperating clients must report children
and must not detach GPU workers.

On client disconnect retain its holdings and allow reconnect. On server loss
clients finish the current update, unload, preserve CPU state and wait. No
indefinite work on an unverifiable grant. Persist a small atomic ownership
journal before grants (identities, reservations, overrides; no model data).
Persist the complete control state: task/reconnect identity, requests/revisions,
directive generations/phases, held/granted/pending reservations, pause latches,
override/revision and bound task. Persist state transitions before replying.
After restart reconcile journal/process state before granting affected capacity.
Reconnect replies present the new epoch and reconciliation directive; old-epoch
reports cannot mutate the new ledger. Direct survivors to release old holdings
before fresh grants, preserving requests and pause state.
A missing/corrupt recovery journal with possible live clients must not become
an automatic empty-ledger takeover. No automatic killing or sudo.

Owner lock/private socket directory prevent duplicate servers and unsafe socket
unlinking. Default modes 0700/0600; opt-in group sharing 0770/0660 with allowed
UID/controller rules. Verify kernel peer credentials using Linux/macOS adapters,
not a UID claimed in JSON. Cross-user process inspection failures are explicit.
This is cooperative access control, not isolation from nonparticipants.

## Phase 3 — memory-bounded fixed-effective-batch training

An update owns exactly B canonical stream IDs (reference B=128), independent of
rank/device/physical batch. Generate/partition just that update, supporting uneven
work counts, unequal microbatches, last partial microbatches, zero-work ranks and
single-sample forwards. No padding samples contribute to the objective. Seeds
derive from canonical sample identities. Preserve each effective batch's membership.

Use raw model forward, without DDP per-forward/buffer/backward communication;
fresh standard PyTorch process group per GPU cohort and explicit end-of-update
reductions. Launch one process per granted UUID with ordinary spawn/launcher
primitives and device visibility set before CUDA. No mutable DDP membership.
Reject BatchNorm/SyncBatchNorm and other batch-coupled modules in managed mode.

Shared existing loss statistics define three optimization components c:
direction, positive presence and negative presence.

`L = sum_c weight_c * N_c / max(D_c, 1)`

For each microbatch, evaluate shared loss code, retain detached scalar statistics
and accumulate `dN_c/dtheta` separately in three FP32 gradient banks. Perform
component VJPs on that one graph, retaining it only between the local VJPs, then
free the graph before the next microbatch. At effective-batch end SUM-reduce
component gradients and packed statistics; form the weighted globally normalized
gradient once, unscale/check overflow/clip once, and take one optimizer update.
All ranks execute the same boundary collective sequence, even with no samples.
Do not multiply/divide by DDP world size twice. Diagnostics do not affect gradients.

General reference overhead is three gradient banks (12 bytes/parameter plus any
separate combined grad) and up to three backward traversals. Measure it. An
implementation-stage optimization may combine components when exact full-batch
denominators are already known from compact targets, using shared formulas and
equivalence tests; it cannot guess counts, add sample-time collectives or retain
all graphs/targets in VRAM. This optimization is not assumed in correctness or
memory estimates.

Synchronize AMP overflow/loss-scale decisions across ranks. Preserve optimizer
moments, scaler, LR schedule, optional optimizer EMA and one step per B samples.
Maintain test-only public snapshots and best-by-test-loss. Evaluate numerical
tolerances for changed summation/kernel order; do not promise bitwise trajectories.
Both fixed and dynamic callers use extracted loss/state/update helpers.

Raw numerator VJPs can overflow with an AMP scale that is safe for normalized
losses. Use actual local component counts for reversible normalization before
backward, and undo it in FP32 when adding to the gradient banks. Dense upper
bounds are not safe: sparse FP16 gradients can underflow irreversibly. For FP16,
retry overflowing local VJPs with power-of-two attenuation on the same current
graph, undo that attenuation in FP32, and retain the global scaler decision.
Retain the loss scaler and single global overflow decision. Tests must include
finite normalized-oracle gradients with overflowing unscaled numerators; merely
reducing the scaler more often is not accepted as equivalent behavior. Validate
each supported loss/precision path before performance claims.

## Phase 4 — persistent supervisor and unload/relaunch

CPU supervisor owns manager connection, canonical consumed frontier, run identity,
log/TB writer and reconfiguration. Workers retain active model/optimizer replicas
between transitions. Extract the shared one-update/test/state logic from train.py;
do not create a parallel copy of the training implementation.

At reallocation, finish the current full update and stop admitting the next.
Capture one authoritative replica's complete state in RAM: parameters/buffers,
optimizer, scaler/scheduler, RNG/canonical frontier, step and test/log/checkpoint
progress. Validate replica agreement at the boundary. Stop CUDA preparation,
synchronize its streams, return shared leases, destroy process group, exit/join
all CUDA-owning workers and close GPU IPC. CPU parent/loaders never create CUDA
contexts. `empty_cache()` alone is insufficient; worker exit is the release.
Only then acknowledge release.

On a new grant, restart a cohort on the exact UUIDs, restore state and process
the next unconsumed effective batch. Audit stochastic model operations for
sample-keyed draws or reject unsupported batch/rank-sensitive randomness.
Run path, TB writer, schedule and test progress remain continuous.
RAM handoff is not a new public/disk training-step snapshot. Root failure resumes
from existing durable test checkpoints. Worker errors abort the cohort and report
ownership accurately; do not continue with partly updated replicas or silently
invent an automatic mid-update recovery.

Reuse the existing bounded CPU pool/slots beyond GPU cohort lifetime. Replace
fixed rank ownership with canonical sample descriptors and bounded delivery to
active workers. Drain leases on restart; keep useful CPU samples by stream ID or
deterministically recompute unconsumed samples. Current-step dependencies always
retain admission capacity; future lookahead cannot own all slots required by
the earliest outstanding sample. Workers claim runnable work, never tasks whose
dependencies require an occupied resource they themselves hold.

Resize the global CPU compute budget, counting loading/augmentation and meaningful
GPU-host/native CPU work. Set native limits before spawning/importing. Current
bounded CPU jobs may complete on shrink, then park/stop surplus workers. During
full pause CPU queues may fill but settle idle; no CUDA buffers/events/prepared
batches remain. Bound host RAM/credits independent of pause duration. Keep the
existing asynchronous H2D/augmentation pipeline inside each active cohort.

Evaluation, visualization and trace metrics obey the same grant and VRAM limit.
Poll during them and release at bounded test work units; preserve or deterministically
replay unfinished test work without duplicate metrics/snapshots. Display pending
reconfiguration phase/age: pause latency includes current update/test unit and
CPU state transfer, not instant revocation.

Preserve canonical logical test groups independently of physical microbatch size:
the evaluator currently averages logical-batch rows. Accumulate sufficient stats
within each original group, finalize that row, then apply the existing row
aggregation. Preserve canonical visualization selection and completed test groups
through pause/repartitioning so test loss and best-snapshot selection do not change.

## Phase 5 — memory calibration, configuration and user workflow

Managed mode is opt-in via training config/socket flag; unchanged unmanaged
defaults. Config records priority/exclusive, max CPU/GPU request, min/max VRAM,
microbatch cap, effective B and measured memory estimate: fixed bytes,
bytes/sample, preparation depth and safety reserve.

Measure warm and initialization peaks, first optimizer step, all gradient banks,
optimizer moments, CUDA augmentation/targets/workspaces, NCCL/driver overhead and
test/visualization. Record allocator reserved and driver-visible peaks with
GPU model, precision, patch/model/config and versions. Per-sample activation
memory alone is insufficient.

For grant M choose largest microbatch b satisfying the measured upper bound:
`fixed + b * per_sample + preparation(b, depth) + reserve <= M`.
Calibrate over multiple sizes; peaks need not be linear. If b=1 cannot fit,
request more or wait. Unexpected OOM/over-budget use is reported, resources
released, estimate corrected before further admission; no silent change of B,
precision, LR or augmentations. Sharing VRAM offers no compute-speed guarantee.

CLI: `serve`, `status`, `pause <task>`, `resume <task>`,
`request <task> ...`, `override set <json> [--until-task ID]`,
`override clear`. Reference Python client demonstrates all lifecycle operations.
Document exact available commands after implementation, not as working tools now.
Use targeted packaging/venv integration; no environment mutation in planning.

## Phase 6 — tests, measurements and protocol lock

**User revision, 2026-09-10:** ordinary tests must finish in a few minutes.
The three-repeat 100-update timing sweep below was stopped at the user's request
after its first full comparison. It is no longer a completion gate or available
through the normal test launcher. Default CPU/manager checks and explicit tiny
CUDA checks share a five-minute invocation deadline. Use existing completed
reference lifecycle/numerical evidence and a short fixed-cohort trace. A longer
pause-RAM settling study is omitted; bounded host queues are tested structurally.
Do not restart extended benchmarking without a new explicit request.

CPU-only protocol/manager tests precede GPU integration:

- Every message/error schema; partial/coalesced frames, malformed/oversize JSON,
  unsupported version/capability, retries/reconnect, stale epoch/revision/report,
  access checks and bounded dedup/connection storage.
- Policy: exclusivity, ties, baseline shares, overload, heterogeneous GPU memory,
  CPU rounding, manual splits/lifetime/reset, paused tasks and request changes.
- Handoff simulations: swaps, delayed release, updates while releasing/starting,
  no overcommit or premature reuse, unrelated progress and absence of allocation
  dependency cycles.
- Real child-process tests: normal finish, root death with surviving descendants,
  PID reuse, inaccessible process metadata, manager crash/journal recovery and
  competing startup. Fake credentials exercise multi-user behavior without sudo.
- Numerical oracle: full B versus decompositions of 16/8/1, unequal and nondivisible
  splits and zero-work ranks; conditioned/legacy branches, empty GT groups,
  FP32/BF16/FP16, clipping/overflow, multiple Adam steps. Compare sample IDs,
  augmentation draws, losses, component gradients, parameters and optimizer states.
- Evaluation equivalence covers unequal sparse counts and partial final groups;
  compare canonical test rows and best-snapshot decisions across GPU/microbatch
  changes and pause/resume.
- Integration: fixed versus resize/shrink/share/pause/resume, repeated cycles,
  no sample skip/repeat, single TB/test output, no task GPU contexts after pause,
  bounded CPU/preload RAM and measured per-device VRAM within grants.
- Failures during loading, forward, reduction, handoff, launch and release must
  surface stage/ownership errors, not silent waiting or fictitious release.

Reference environment: `/home/hendrik/.venv_las`, artifacts under
`/ephemeral/hendrik/fiber3d_train2`, archived S1A NML 8x8 data/test selection,
effective B=128 and current optimized GroupNorm/precision settings.
Check GPU availability before future runs; do not assume prior device availability.
Use existing approved reusable `vesuvius/scripts/run_fiber3d_profile_benchmark`
and pipeline callers/control files. No GPU work in this turn.

Record exact commands/commit/config/data/build, GPU type/count, microbatch per
GPU, effective B and CPU worker/core count. Matched warmup plus at least 100
measured updates, three repeats: step mean/p50/p95 and samples/s globally and
per GPU. Compare unmanaged, managed fixed allocation and dynamic transitions.
Separate component backward/reduction, polling, state-copy/unload/relaunch,
GPU-free pause latency and memory peaks. Compare unchanged held-out evaluation
after equivalent updates. Do not claim the overarching one-hour convergence
goal from a throughput test.

After full reference validation, mark protocol v1 locked with dated schema,
conformance fixtures and report. Add explicit repository instructions: no wire
shape, units, operations or semantic changes without explicit user instruction,
including supposedly compatible changes; no automatic version bump workaround.
Manager policy settings must remain within documented protocol semantics.

Commit implementation in reviewable pieces: core package/protocol/tests; shared
loss/state extraction; bounded accumulation; supervisor/pipeline integration;
calibration/client/CLI docs; validated results and protocol lock. Keep standalone
manager/training changes separate where practical. No commit in this planning turn.

## Spec update

On implementation update `planning/specs.md`: opt-in cooperative mode, canonical
effective batches, bounded component accumulation, end-of-update communication,
CPU supervisor, GPU-free pause and acknowledged handoff. Qualify existing
full-local-batch/retained-graph statements as old-path behavior. Extend shared
sample ownership across cohorts and document prediction-dependent legacy branch
group counts. Keep existing default configs, losses and snapshot rules.
Authoritative protocol specification resides in `gpu_manager/docs/`; link it,
do not maintain divergent copies.

## Docs updates

Package README/protocol schemas/reference client/operator guide (permissions,
priorities, overrides, recovery), Fiber training configuration/memory calibration
guide, measured validation report. Update `docs/code_structure.md` for shared
helpers, supervisor and pipeline lifecycle. Maintain status/task log by phase.

## Changelog update

After implementation add durable entries for standalone manager and dynamic
Fiber training with validation/performance impact. Do not describe planning as
an implemented feature.

## Sources checked during planning

Python filesystem socket support:
https://docs.python.org/3/library/socket.html#socket-families
PyTorch standard spawned process management:
https://docs.pytorch.org/docs/stable/multiprocessing.html
PyTorch launch/restart lifecycle:
https://docs.pytorch.org/docs/stable/elastic/run.html

These support Unix-socket and process-cohort design choices; they do not promise
in-place local GPU resizing. Confirm the installed APIs at implementation time,
as online stable documentation may describe a newer release than this venv.
