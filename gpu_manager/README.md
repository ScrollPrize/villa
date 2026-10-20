# Cooperative GPU manager

`villa-gpu-manager` coordinates **promised** CPU/GPU/VRAM budgets on one host.
It neither kills processes nor enforces resource usage. It has no Torch, VC3D
or Lasagna dependency. Clients must unload GPU workers before acknowledging
release; closing a socket does not release anything.

**Protocol v1 is locked (2026-09-10).** Standalone service/client, ownership and
failure/recovery tests pass. Fiber integration was validated on two H100 GPUs
with real S1A data, including GPU-free pause, partial VRAM and resize. This is
cooperative accounting, not isolation or enforcement. Other training/device
configurations require their own calibration. See the [protocol](docs/protocol-v1.md)
and [validation scope](docs/validation.md).

## Local development

From the Villa root, without installing:

```bash
python -m pytest -q gpu_manager/tests
PYTHONPATH=gpu_manager/src python -m villa_gpu_manager.cli --help
```

Explicit installation into your chosen Python environment:

```bash
python -m pip install -e ./gpu_manager
```

For development tests, install `./gpu_manager[test]` instead. Python 3.11+
and `psutil` are the only runtime requirements. This Unix-only package does not
support Windows; macOS can run the metadata service/tests but NVIDIA discovery
requires an available `nvidia-smi`.

Five optional CPU Torch/Gloo integration cases skip when those dependencies
are absent; the standalone tests do not require Torch. CI runs the integration
cases in a separate job with CPU-only Torch installed. With Torch/Gloo available,
run them explicitly with `python -m pytest -q gpu_manager/tests -m torch`.

No installer or GPU service is automatically started by importing the package.
Linux/macOS Unix sockets are supported; NVIDIA inventory monitoring requires
`nvidia-smi`. Tests use fake inventories and do not allocate GPUs.

## Service responsiveness

The production CLI separates socket multiplexing, policy/journal ownership, and
process-table collection into three OS processes (not Python worker threads).
One collected PID/parent table is reused across all task trees. Read-only polls
do not copy historical state or recalculate allocations. The state owner retains
the singleton lock even if the socket process crashes; recovery cannot race an
old journal writer. Wire protocol and resource-release verification are unchanged.

Slow collection does not block poll/status replies. Durable mutations still wait
for journal writes, and policy work remains serialized in its owning process:
this is not a guarantee against arbitrarily slow storage or policy computation.
See [measurements and tests](docs/responsive-service-validation.md).
Already-running managers retain their loaded implementation; deploying this
change requires a controlled service restart, not deletion of its journal.

## Operator commands

Python tasks may call `villa_gpu_manager.bootstrap.ensure_manager()` to reuse an
existing service or start a detached one automatically. It uses the same socket
selection and ownership journal/lock as the CLI. Simultaneous starters converge
on one service; no journal or socket is manually reset. Only absent/refused
connections permit startup; authorization/protocol failures remain errors.
Startup logs are under `~/.local/state/villa-gpu-manager/`. The auto service
inventories all host GPUs; client visibility restricts requests, not the shared
pool. Existing GPU processes are protected by the normal external-use monitor.
The service exits after 30 seconds with no connected clients and no live tasks.
This grace period permits the bootstrap probe to hand over to the real client.
A disconnected live task still owns its reservation; the regular PID-tree scan
retires dead tasks before idle shutdown. The socket is removed on clean exit,
but the recovery journal is retained. A later client automatically starts a new
service. Automatic inventory counts logical CPUs (224 on a 112-core SMT host),
so a request for 128 loading workers fits that host's CPU capacity.
On restart, a CPU-only inventory correction is accepted only for a fully retired,
empty-reservation journal without an override. Live ownership and GPU inventory
changes still fail closed; journals are never deleted to bypass this check.
Numeric visibility masks are explicitly resolved as physical `nvidia-smi`
indices and logged; UUID masks avoid CUDA ordinal-order ambiguity. Workers are
always launched with UUID visibility. Malformed/ambiguous masks are errors.

For an already prepared shared-group directory, auto startup allows its Unix group
members as clients; only the server owner is a controller. Existing directory
ownership/sticky-bit/world-permission protections still apply. It does not create
Unix groups, change permissions or acquire privileges automatically.

First identify UUIDs with `nvidia-smi --query-gpu=index,uuid --format=csv`.
Pass only the GPUs explicitly assigned to this cooperative pool. Example:

```bash
villa-gpu-manager serve --gpus GPU-UUID-ONE GPU-UUID-TWO --cpu-cores 32
villa-gpu-manager status
villa-gpu-manager pause TASK_UUID
villa-gpu-manager resume TASK_UUID
villa-gpu-manager request TASK_UUID request.json
villa-gpu-manager override set split.json --until-task TASK_UUID
villa-gpu-manager override clear
```

`--socket PATH` is a global option (before the command). Server and training
clients prefer `/run/villa-gpu-manager/manager.sock` when a listener is reachable
there or its location is writable/searchable (including a writable ancestor from
which the directory can be created). Otherwise they use
`~/.local/run/villa-gpu-manager/manager.sock`. Selection creates no directories
and makes at most a 100ms connect-only probe; the normal handshake still enforces
protocol and UID authorization. Errors after selection do not cause fallback.
Explicit paths override selection. Its directory must be private and
owned by the server user. For explicit cross-user cooperation, preconfigure a
shared group directory with sticky mode `1770` and use `--shared-group --allow-uid UID`; granting control
over other clients additionally requires `--controller-uid UID`. No sudo is used.

`request.json` contains the complete replacement request:

```json
{"max_cpu_cores":32,"max_gpus":2,"min_vram_bytes_per_gpu":8589934592,"max_vram_bytes_per_gpu":34359738368,"allowed_gpu_uuids":[],"priority":0,"exclusive":false}
```

`split.json` maps task UUIDs to `{"cpu_cores":N,"gpus":{"GPU-UUID":BYTES}}`.
Omitted tasks get zero. Invalid overcommit is rejected atomically. A request or
pause incompatible with an existing split invalidates it and restores automatic
allocation. A bound split ends when that task finishes, not merely disconnects.

Larger explicit priorities win admission and spare capacity (FIFO breaks ties).
Each feasible nonexclusive task gets a minimum share first. Exclusivity adds no
priority bonus: it requires the pool to be otherwise idle. Non-pausable running
jobs block preemption; waiting reasons identify them. Lower-priority tasks may
wait indefinitely. Partial GPU means a VRAM allowance, **not** a
compute fraction, CUDA partition, or speed guarantee.

## Recovery and shutdown

The service journals ownership beside the socket (`.json`), protected by a
separate `.lock`. Never delete these files while clients might still hold GPUs.
Restart with the same inventory; surviving clients must release before receiving
a fresh grant, except declared non-pausable clients whose grants remain intact.
Tasks whose journal already records both granted and held allocations as empty
recover directly to waiting, preserving pause flags; they need no additional
release acknowledgement. This also repairs empty tasks left in release by the
older recovery implementation.
Missing/corrupt recovery state fails closed rather than assuming
the GPUs are free. Unknown/inaccessible process state remains reserved.

The first Ctrl-C drains the pool and waits for release acknowledgements. A second
Ctrl-C forces shutdown, retaining reservations in the journal. Pausable clients
detect server loss and unload at a safe boundary. Non-pausable clients retain
their grant and reconnect; graceful drain waits for them to finish. Forced shutdown is not successful
draining. `--reserve-bytes` defaults to 1 GiB per GPU; reserve external/driver use
explicitly. The nonblocking driver monitor blocks new grants when observed
unowned usage exceeds that reserve or GPU inventory becomes unverifiable.

## Client responsibilities

See [the reference lifecycle example](examples/cooperative_client.py). A CPU
supervisor should poll every second and at safe work boundaries, even while GPU
workers are busy or paused. `initialize` grants permission; `target` alone does
not. Reallocation uses complete old-cohort release, including unchanged GPU UUIDs.
Report actual remaining CPU work while draining. Finish only after release.
Do not detach GPU descendants, and report their PID/start/boot identities.

The low-level `Client` never calls CUDA, launches workers, or claims release on
your behalf. A context-manager exit only closes the connection.

For a fixed, non-pausable evaluation job, register with
`client.register("evaluation", request, pausable=False, adaptable=False)` and set
`request["exclusive"] = True` with the intended explicit `priority`. Poll until
`phase == "initialize"` before starting any GPU work; display `task["reason"]`
while waiting. Report `applied` with the grant and worker process identities.
On completion, self-pause, unload/join all GPU workers, report `released` with
zero resources, then `finish`. Defaults remain pausable/adaptable for training.
Use `pausable=True, adaptable=False` for a job that can pause but must resume
with the identical allocation. See the protocol's authorized scheduling amendment.

`Supervisor` is the reusable CPU work-boundary coordinator. `ProcessCohort`
provides gated spawned children with bounded private controls and opaque CPU RAM
state handoff. Application adapters provide initialize/run-unit/capture/close;
only joined children count as released. These helpers do not import Torch.
On Ubuntu/macOS each worker has a dedicated session. Loader descendants must
remain in that session (no `setsid`/daemon detachment); failure cleanup stops the
whole owned session, including orphaned loaders, before allowing release.

`StopGuard` is an optional independent **client-owned** termination helper. Register
each compute session leader immediately after spawning (the cohort's
`on_process_start` callback supports this), call `check()` in the supervisor loop,
and call `stop()` on cancellation. It gives graceful cleanup 15 seconds, then
escalates TERM/KILL with five-second stages. A second stop accelerates escalation.
Call `close()` only after compute cleanup. Never register the supervisor/manager
as a compute root. Session descendants must not detach; unkillable survivors are
reported without claiming release. It does not alter the manager wire protocol.

`ExclusiveTask` is the reference helper for fixed-budget, non-pausable probes.
It waits for an exact exclusive grant, then the caller assigns a gated
`ProcessCohort` to `lease.cohort`. `lease.authorize()` confirms applied ownership
and worker identities before opening the gate; `lease.check()` polls for loss
of ownership. Context exit aborts/joins remaining owned workers before reporting
release and finishing. A failed join retains the reservation. It uses the existing
protocol, not a new scheduling API. Non-cooperating GPU processes remain outside
the manager's enforcement; callers must still check actual available memory.

`run_exclusive(command, ...)` uses that same lease with a `GatedCommand` adapter.
It reserves the full managed CPU capacity and selected devices' managed VRAM,
validates the complete grant, and reports an independent cleanup watchdog before
authorizing the command. Heterogeneous device capacities are supported through
the existing proportional CPU request calculation. SIGTERM/HUP cancel and join
owned workers; connection loss retains the fixed allocation while reconnecting
with the same task token. Commands must not detach their descendants.
Application launchers may call `ensure_manager()` first and supply a
`before_authorize` check for external GPU occupancy. There is no unmanaged
fallback on bootstrap, admission, or ownership failure.

For a fixed nonexclusive job, `wait_for_exact=True` returns incomplete admission
grants without starting workers and retries registration until the requested
allocation is available. This is opt-in; it can wait indefinitely and each new
registration joins the FIFO queue anew.

`ProcessOwner` provides a separate ownership root for sibling cohorts managed
by one coordinator. Pass its identity as `Client(registration_root=...)`,
track each worker session before authorizing it, and retire it only after joining
its workers. Closing the owner or losing the coordinator triggers cleanup of
tracked sessions. This uses the existing explicit-root/worker protocol and does
not claim unrelated coordinator children.
