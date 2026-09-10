# Cooperative GPU manager (development)

`villa-gpu-manager` coordinates **promised** CPU/GPU/VRAM budgets on one host.
It neither kills processes nor enforces resource usage. It has no Torch, VC3D
or Lasagna dependency. Clients must unload GPU workers before acknowledging
release; closing a socket does not release anything.

**Implementation status:** standalone service/client and ownership tests exist.
The Fiber supervisor is connected and CPU integration-tested, including a
two-process to one-process training restart. The protocol is a draft; calibration
and reference CUDA validation are not complete. Do not use this yet to coordinate
production training. See the [protocol](docs/protocol-v1.md).

## Local development

From the Villa root, without installing:

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q gpu_manager/tests
PYTHONPATH=gpu_manager/src python -m villa_gpu_manager.cli --help
```

Explicit installation into your chosen Python environment:

```bash
python -m pip install -e ./gpu_manager
```

No installer or GPU service is automatically started by importing the package.
Linux/macOS Unix sockets are supported; NVIDIA inventory monitoring requires
`nvidia-smi`. Tests use fake inventories and do not allocate GPUs.

## Operator commands

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

`--socket PATH` is a global option (before the command). Default:
`~/.local/run/villa-gpu-manager/manager.sock`. Its directory must be private and
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

Larger priorities win spare capacity. Each feasible nonexclusive task gets a
minimum share first. Exclusive tasks reserve the whole pool; lower-priority
tasks may wait indefinitely. Partial GPU means a VRAM allowance, **not** a
compute fraction, CUDA partition, or speed guarantee.

## Recovery and shutdown

The service journals ownership beside the socket (`.json`), protected by a
separate `.lock`. Never delete these files while clients might still hold GPUs.
Restart with the same inventory; surviving clients must release before receiving
a fresh grant. Missing/corrupt recovery state fails closed rather than assuming
the GPUs are free. Unknown/inaccessible process state remains reserved.

The first Ctrl-C drains the pool and waits for release acknowledgements. A second
Ctrl-C forces shutdown, retaining reservations in the journal. Clients must detect
server loss and unload at a safe work boundary. Forced shutdown is not successful
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

`Supervisor` is the reusable CPU work-boundary coordinator. `ProcessCohort`
provides gated spawned children with bounded private controls and opaque CPU RAM
state handoff. Application adapters provide initialize/run-unit/capture/close;
only joined children count as released. These helpers do not import Torch.
