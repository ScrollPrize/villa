# Cooperative resource management

The approved work spans a reusable [manager](../../../../../../gpu_manager/README.md)
and Fiber3D training. The default managed launcher is connected and validated with CPU
failure/recovery tests and real S1A CUDA lifecycle/gradient checks. See the
[validation report](cooperative_validation.md) for measured scope and limitations.
Validation used GPUs 0/1 only, below 20 GiB/GPU; other configurations need calibration.

## Implemented foundations

- `gpu_manager/`: independent Unix-socket client/server, deterministic CPU/VRAM
  reservations, release-before-regrant, priority/exclusive/manual controls,
  process identities, journal recovery, CPU conformance tests.
- `fiber_trace_3d/loss_statistics.py`: the shared statistics used by the existing
  trainer and bounded accumulation; no copied loss implementation.
- `gradient_accumulation.py`: target-normalized single backward for ordinary
  one-branch and conditioned training; unsupported losses fail explicitly.
  One graph at a time, preserved global loss normalization.
- `cooperative_resources.py`: explicitly calibrated memory bounds and canonical
  effective-batch partitions, including unequal VRAM grants. No implicit estimate.

## Fixed-cohort training option

Set `training.bounded_gradient_accumulation: true`, use GroupNorm, and set
`training.ddp_broadcast_buffers: false`. Existing defaults remain unchanged.
This enables bounded activation memory in the **existing fixed-cohort** trainer;
arbitrary/partial physical batches and zero-work ranks are supported in the shared
trainer. Physical partitioning does not change the effective batch's sample IDs.

Ordinary one-branch and conditioned training prepare the local macro-batch's
normalization counts before its first forward. The prepass uses the same target
voxelizer/device and compact geometry with a storage-free image view; it does
not run the model, transfer input images or rerun value augmentation. Counts are
globally summed once. Each microbatch then performs one combined backward, and
gradients are SUM-reduced once per update. An observed-count check prevents a
mismatched prepass from silently changing the loss. CPU queue slots are copied
into bounded owned host staging and promptly released; GPU inputs/graphs remain
microbatch-bounded. Geometry CUDA work honors the preparer's pause/stream barrier.

Embedding and ordinary multi-branch losses are temporarily rejected at training
startup and evaluation loss entrypoints. TODO: make their global denominators
target-only before re-enabling them. Their inference models remain usable.
Conditioned counts are 12 per GT voxel, two positive groups per GT-positive sample,
and two negative groups per sample, including empty-GT samples. No production
component-backward fallback remains. The old accumulator remains a numerical/
performance reference only. Floating-point trajectories are not bitwise identical
across GPU/microbatch counts.

### Stopping

Ctrl-C/TERM requests graceful stopping. An independent client-owned process
enforces a 15-second deadline, then TERM/KILL with five-second stages. A second
request accelerates escalation. It tracks registered compute sessions, including
orphan descendants, never the supervisor's shared manager service. Loader cleanup
uses shared deadlines and avoids abandoned queue feeder joins and dead-owner
counter locks. Unkillable kernel-I/O survivors are reported; the guard does not
kill the root to fake release. Forced stopping can lose updates since the last
persisted checkpoint. The manager wire protocol is unchanged.

### Precision and numerical validation

Production honors `training.mixed_precision` unchanged: BF16 model operations
remain under the shared trainer's BF16 autocast. FP32 master parameters, optimizer
state and gradient reductions are deliberate. Loss/reduction operations
may also use FP32 for stability; this is not an all-BF16 pipeline.
`cooperative_reference_validation.py` optionally sets `mixed_precision="off"`
in **generated test configs only** to isolate summation from BF16 kernel effects.
The frozen-gradient test additionally uses FP64 shadow sums, only in the oracle.

Real S1A B128 validation on GPUs 0/1 (GroupNorm, quarter-output, six updates):

- Fixed FP32 repeat and same-allocation FP32/BF16 pause/restart: byte-identical
  model, optimizer and final test metric. Pause during testing also covered.
- Frozen FP32 gradients: exact sample IDs/counts, matching one-/two-GPU FP64
  shadow sums; every component/layer within the computed FP32 summation bound.
- Resize trajectories are diagnostics, not a widened tolerance gate. Original
  pointwise comparisons failed for a small subset of weights. FP32 final model
  relative L2 difference was 1.39e-5; BF16 was 5.90e-5. Final test losses were
  2.242254823/2.242277861 (FP32), 2.239006564/2.239104435 (BF16), fixed/dynamic.
  The FP32 shadow test isolates accumulation roundoff, not every BF16 kernel effect.

Evidence paths and remaining gates are recorded in `planning/task_log.md`.

The existing approved profile caller accepts `--bounded-accumulation` via its
argument file. Native worker pools are bounded; HiGHS is optional, and its
dependent features are disabled when unavailable. The tested QuickBuild uses
`-O3` without LTO. Tests load isolated native bindings; they do not replace a
running training's venv. The shared single-GPU preparation path resolves the CUDA
device index before entering its background thread.

## CPU-only verification

### Reusable test caller

Run the fixed command (no arguments):

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
```

Test selection lives in `vesuvius/scripts/fiber3d_cooperative_tests.json`, so
changing suites/devices does not change the approved launcher command. Named
suites are `cpu`, `manager`, `cuda-amp`, `cuda-cohort`, `cuda-reference`, and
`native-build`, `native-configure` and `native-tests`; they run sequentially.
The default is CPU/manager tests (about 90 seconds here), with GPUs hidden.
Every invocation has a **five-minute total ceiling**, not five minutes per suite.
Long benchmark sweeps are rejected. CUDA tests remain explicitly selected.
`native-configure` selects QuickBuild -O3 without LTO and one autogen thread.
`native-build` builds the existing VC3D
Python/test targets without installing. Setting `native_build: true` selects
that build's Python bindings for isolated tests, leaving the venv unchanged.
`cuda-reference` implements calibration and a separate lifecycle phase, using the archived S1A
data selection, GroupNorm, quarter-resolution output and effective batch 128;
its control records microbatch sizes, steps and CPU workers. The lifecycle phase
requires `calibration` pointing to a prior calibration.json and at least six
steps; it compares fixed allocation with pause/shrink/expand, including pause
during an unfinished test wave and a driver check that paused GPU workers have
disappeared. Diagnostic end states are retained before numerical comparison so
failures can be inspected without repeating the entire run. One extended
unmanaged/fixed/dynamic comparison completed; the user stopped the redundant
repetitions. That historical benchmark is not part of the normal test command.
The `profile` phase accepts two updates and records a short trace of the same trainer; repeated
microbatch stage wall times sum over the update. Stage host times are not CUDA
kernel duration; use the recorded CUDA trace/operator tables for device time.
CPU suites hide GPUs. CUDA suites use tiny models and explicit allocator caps:
4 GiB for the numerical oracle, at most 6 GiB per cohort worker plus a 4 GiB
driver reserve. These caps are below the authorized 20 GiB total per GPU; they
are not a general CUDA driver-enforced memory quota. This caller does not run
uncapped reference benchmarks. Timeout cleanup targets only its test process
group. The shared driver monitor records only this test's process-tree memory
and aborts that test if observed use exceeds 20 GiB per GPU; the lower allocator
caps leave headroom. Reference calibration uses a 16 GiB allowance with 4 GiB
reserved outside the allocator (12 GiB allocator cap). Control, command, source
revision/worktree status, logs and results are
saved below `/ephemeral/hendrik/fiber3d_train2/cooperative-tests`.

### Direct CPU-only invocation

From `vesuvius/`, with GPUs explicitly hidden:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest -q tests/neural_tracing/test_gradient_accumulation.py tests/neural_tracing/test_cooperative_resources.py
```

The Gloo test needs local loopback socket access, but uses no GPUs. From the
Villa root: `python -m pytest -q gpu_manager/tests`.

## Managed launch

Explicitly install the independent package into the training venv:
`python -m pip install -e ./gpu_manager` from Villa. Nothing installs or starts a
service on import. Start the manager on explicitly available GPU UUIDs using its
README if you want to operate it separately. Otherwise normal training starts
or reuses it automatically. Run one CPU supervisor, **not torchrun**:

```bash
python -m vesuvius.neural_tracing.fiber_trace_3d.train CONFIG.json
```

Training automatically prefers `/run/villa-gpu-manager/manager.sock` when its
location is writable/searchable or a listener is reachable; otherwise it uses
`~/.local/run/villa-gpu-manager/manager.sock`. Optional `--manager-socket PATH`
overrides `training.cooperative.socket`, which overrides that shared default.
Only explicit `--unmanaged` selects the testing/debug path; failures never fall
back to unmanaged execution. Do not use a fixed-rank launcher for production.
No `training.cooperative` fields are required. Training reuses a compatible report
from `<run_path>/.memory-calibrations/<config-fingerprint>.json` (relative run paths
resolve against the config directory). If none exists, it requests exclusive,
non-pausable/non-adaptive startup ownership of all managed GPUs. Startup runs the
real initial evaluation and three real optimizer updates per attempted physical
microbatch size, ascending from one to the configured target when memory permits.
The logical macro-batch, samples, optimizer/LR progression and logging remain real:
there is no discarded probe training or replayed update.

After measurement, it derives maximum CPUs from the
configured sample worker count capped by manager inventory, GPU count from visible
compatible devices capped by measured distributed coverage, minimum VRAM from the
smallest measured microbatch, and maximum VRAM from inventory. Priority defaults
to zero, non-exclusive. Global batch remains fixed through accumulation.

Set only `training.cooperative.priority` to change training priority without
specifying resources, for example `"cooperative": {"priority": 10}` within
`training`. Higher integers take precedence. This shorthand overrides the older
`training.cooperative.request.priority` field when both are present.
Calibration defaults to that training priority **plus 100**; its `--priority N`
sets an exact override. Exclusivity itself never raises priority.

Only an absent/refused socket starts a detached manager, with host-wide inventory,
the existing ownership lock/journal and GPU monitor. Permission or protocol errors
never start a replacement or fall back. Logs are in
`~/.local/state/villa-gpu-manager/`. Existing external GPU use remains blocked by
the monitor; automatic startup does not grant ownership of somebody else's work.
An automatically started daemon retires once it has no clients/live tasks for its
idle grace period; a later training command starts it again safely.

The following `training.cooperative` object is an **optional advanced override**, not
required setup:

```json
{
  "request": {
    "max_cpu_cores": 128,
    "max_gpus": 8,
    "min_vram_bytes_per_gpu": 8589934592,
    "max_vram_bytes_per_gpu": 68719476736,
    "allowed_gpu_uuids": [],
    "priority": 0,
    "exclusive": false
  },
  "input_queue_depth": 2,
  "memory_calibration": "REPLACE WITH THE MEASURED JSON OBJECT"
}
```

Those byte counts illustrate units, not a measured recommendation. The calibration
must be an object, not the placeholder. Keep the numerical/data configuration,
including `optimizer_global_batch_size` (e.g. 128). Use sample-independent layers,
`sync_batchnorm: false`, `ddp_broadcast_buffers: false`, finite `max_steps`, and the
shared sample pool. `sample_pipeline_workers` is its maximum size, capped by grants;
`sample_pipeline_slots` must fit one whole effective batch. Initial lookahead can
remain small and uses the existing bounded ramp. `--resume` starts from a durable
test snapshot; live reallocations use RAM, not training-step snapshots.

### Automatic memory measurement and limits

New startup profiles include allocator/driver observations and a conservative
train-plus-initial-test envelope, with a 4 GiB reserve. Later dense evaluation
retains the measured **physical microbatch of one**; logical test grouping and
metric reduction are unchanged. This cap is conservative, not an evaluation speed
optimum. Only measured training microbatch sizes are eligible for adaptive grants.
OOM fails loudly without skipping/replaying a partially completed optimizer update.
Calibration cannot prove a universal bound for all future data-dependent workloads.

Manager communication failures log the actual exception and retry every5s after
a failed attempt. Already-authorized adaptive training continues for up to60s of
continuous outage, then pauses at a safe boundary, joins GPU workers and retains
state in RAM. After reconnecting, ownership is reconciled before automatic resume.
Known release directives, epoch changes and explicit stop bypass this grace.
Exclusive startup calibration can finish its current authorized microbatch size
offline, but must reconnect before starting a new cohort/size or adaptive task.
Each synchronous RPC has its own bounded timeout, so attempts can briefly delay
supervisor dispatch; grace does not make communication asynchronous.

The implementation preserves protocol v1's immutable registration capabilities:
the exclusive startup registration finishes after its GPU workers join, then an
adaptive registration continues the same training run with RAM model/optimizer/RNG,
sample ownership and TensorBoard writer retained. This changes manager task ID,
not run identity. Re-registration is the approved lifecycle; GPU validation remains
pending, and current CPU tests do not establish real-GPU performance.

Startup requires configured dense test data and a positive test interval so it can
measure a complete real evaluation. A missing report never falls back to unmanaged
execution. All-GPU visibility is required for uncalibrated startup.

### Optional explicit memory probe

Normal training no longer requires this command. It remains available for explicit
diagnostics/comparison, and unlike automatic startup its updates are probe work.

This measurement automatically connects to or starts the manager, then waits for
an exclusive grant before authorizing any GPU worker. Other cooperative tasks
must release first; an unpausable task can block the request until it finishes.

```bash
python -m vesuvius.neural_tracing.fiber_trace_3d.memory_calibration CONFIG.json --gpu-uuids GPU-UUID-ONE GPU-UUID-TWO --microbatch-sizes 1 4 8 16 --cpu-workers 8 --steps 3 --vram-limit-bytes 17179869184 --reserve-bytes 4294967296
```

The explicit VRAM limit is required. There is no unmanaged/shared bypass: external
compute processes are still rejected, since the manager cannot enforce cooperation.
Each microbatch-size probe registers as non-pausable/non-adaptable, reports its
gated workers before authorizing them, and joins them before releasing its grant.
Manager loss aborts the probe's own workers. Between sizes resources can be reassigned.
Use the intended maximum GPU count; a one-GPU calibration cannot authorize an
eight-GPU request. The shared trainer measures cold initialization, configured
initial tests/visualization, optimizer state, gradient banks, asynchronous prepared
batches and distributed communicator overhead. A fresh cohort is used per size.
The report records allocator peaks, polled driver-visible memory, GPU/PyTorch/CUDA
identity and config fingerprint. By default the report goes to the discoverable
location above; training reads it automatically. An explicit `--output` remains
available for experiments; those nonstandard reports require the optional embedded
`memory_calibration` override. No report is fabricated from model size or free RAM.
Manager control settings are excluded from the fingerprint; numerical/data config
changes require recalibration. Only measured batch sizes are selected. The reserve
is still needed: driver polling can miss transient peaks and other group sizes can
select different communicator algorithms. A two-GPU S1A BF16 calibration has
passed; this does not calibrate other model/data configurations or eight GPUs.

Runtime checks GPU/software identity, self-limits the caching allocator below the
grant minus reserve, and monitors its own driver-visible process memory. OOM or an
over-budget observation fails and releases the task, not an implicit change of B,
precision, LR or augmentation. Calibration is not a formal upper-bound proof.

## Ownership and pause

- `gpu_manager/supervisor.py` / `cohort.py`: gated spawned children, polled work
  boundaries, nonblocking private controls and join-before-release. Children are
  reported before CUDA initialization. Model handoff uses opaque CPU shared memory.
- `cooperative_training.py`, `cooperative_cohort.py`, `cooperative_worker.py`:
  connect the existing trainer to replaceable GPU processes. Each cohort has a
  fresh process group and exactly its granted UUIDs. The supervisor owns no CUDA.
- `sample_pipeline.py` / `sample_broker.py` / `ordered_sample_pipeline.py`: one
  persistent source-tagged elastic CPU pool for training and dense evaluation.
  Both managed and unmanaged paths use these factories and queue engine. Current effective-batch slots stay leased until all
  ranks commit; speculative GPU inputs can be discarded without consuming samples.
  Workers finish jobs then park on shrink. One active worker can execute both stages.
  Native compute, volume decoding and sampling use one thread per active worker.
  Admission stops before scheduled evaluation; future training cannot occupy its
  required credits. Source-specific tensor banks preserve uint8 deferred training
  transfers and float32 CPU-normalized evaluation, with one shared ownership bound.
  Logical test groups must fit the configured slot count; invalid sizes fail early.
- `training_state.py`: shared snapshot helpers and RAM model/optimizer/scaler/RNG,
  sample frontier, run identity, best metric and pending test rows.
- `cooperative_logging.py`: one CPU-owned TensorBoard writer across cohorts.

Dense evaluation preserves original logical groups while physically microbatching.
It yields after a wave of at most one group per rank; completed rows survive resize.
Trace evaluation yields between CP segments. Metrics and snapshots are not repeated.
Test visualization forwards one sample at a time. Managed training visualization
reuses the broker's first canonical samples before update commit, independent of
rank placement, and applies the same deferred augmentation helper as CUDA batch
preparation. It does not synchronously reload volume data.

Full pause joins all GPU workers/preparation streams/process groups. CPU samples,
writer and handoff may remain in RAM. Neither `empty_cache()` nor socket EOF proves
release. Queue feeder failures are surfaced; CPU draining also has a deadline.
Worker errors abort owned children rather than continue a partly updated step.

## Validation scope

CPU tests cover partial/arbitrary microbatches, interruption during evaluation,
actual two-process to one-process Gloo training restart, optimizer equivalence and
one writer. Relevant additional tests: `test_training_state.py`,
`test_sample_broker.py`, `test_cooperative_integration.py`, `test_memory_calibration.py`.
The small CUDA oracle now passes FP32/FP16/BF16 checks and the two-GPU to one-GPU
restart scenario. These use synthetic data/tiny models, not the reference UNet.
Real S1A BF16 resize/pause/expand completed, including driver-verified GPU-free
pauses; reference FP32 frozen-gradient and FP32/BF16 exact same-allocation restart checks
also pass. BF16 resized trajectory differences are not asserted bit-identical:
after 110 updates the fixed/dynamic test losses were 1.582692/1.606135. Exact
same-allocation restart and preserved gradient semantics do not establish equal
long-run convergence across changing physical batch shapes.
The completed extended case exercised repeated GPU-free pauses, partial VRAM,
shrink and expansion; further timing repeats were stopped at the user's request.
Host queue memory is structurally bounded/tested; no long pause-RAM plateau study
or eight-GPU throughput/convergence claim is made. No unrelated process was stopped.
# Visualization of actual forward results

Initial and periodic train/test sample sheets use the same bounded observer of
ordinary loss forwards. Inputs already include augmentation and targets;
predictions are detached for rendering. No visualization sample request, extra
forward, reaugmentation, or model-mode change is performed. Train captures up to
the configured count across rank-zero local microbatches; no-work ranks produce
no image. Test CPU sheets survive pending-test handoff across resource changes.
Conditioned sheets show actual zero/random-query outputs, not an extra recurrent
inference. Scalar/test schedules and loss/gradient calculations are unchanged.

The fixed reusable test command is:

```bash
/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests
```

It accepts no arguments. Edit `vesuvius/scripts/fiber3d_cooperative_tests.json`
to select named suites; `visualization` runs the CPU-only capture, evaluation,
broker and integration regressions. The five-minute total limit and per-run
logs/config snapshots remain active. GPU suites acquire an exclusive,
non-pausable/non-adaptable manager grant before subprocess startup. They wait
behind existing non-pausable work; the execution time limit starts after the
grant. Test descendants are joined before release, including new sessions.
