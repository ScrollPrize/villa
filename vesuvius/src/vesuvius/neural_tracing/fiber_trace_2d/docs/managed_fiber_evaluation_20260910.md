# Managed parallel fiber evaluation

Demand scheduling now uses RAM-only metadata, not SQLite. See
[RAM state and restart behavior](live_inference_ram_state_20260910.md) for the
current cache lifecycle, validation, limitations and profiling comparison.

The classic runner now defaults to a fixed exclusive GPU-manager request,
priority100. It never starts training. Only cooperative registered pausable jobs
can yield; it cannot evict arbitrary GPU processes. A missing service is an error,
not permission to run unmanaged. The standalone module supports the same config
and --manager-socket/--manager-priority options; --unmanaged is an explicit opt-out.

The CPU-only launcher resolves physical NVIDIA indices (or indices into an
existing CUDA_VISIBLE_DEVICES mask) to UUIDs, obtains the exact managed grant,
reports the gated child plus a separate watchdog, then permits execution.
The child receives granted UUIDs as CUDA_VISIBLE_DEVICES and local0..N-1 indices.
The fixed non-pausable job retains its grant across manager disconnect/reconnect.
Normal completion/cancellation verifies all owned non-detaching command-group
workers are gone before reporting zero resources. The watchdog survives launcher
and controller loss and retains the reservation until cleanup is verified.
Commands must not daemonize/detach workers. Killing every supervisor/watchdog
simultaneously or adversarial process escape is not an enforcement guarantee.

The merge is fed6dc24a, after native/shared/Fiber checkpoints f31412abc,
2c88b8ffd,ef16a59e5. The train.py conflict combined cumulative sample-axis logging
and checkpoint metadata with manager RAM handoff, bounded accumulation and
resumable evaluation. C++ code merged automatically; both README sections stayed.

## Running

With the service running, the reusable benchmark command is:

```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

This runs the real shared inference pipeline on the locked PHerc0139 test-fiber
intersections, not the100-direction native tracer. Model-only controls and
worker warmup precede the40-second pipeline measurement. It uses8GPUs,512-cubed
inputs, existing bf16 inference and existing float16 raw/float32 weight accumulation.
It records a timestamped workload/report beneath
/ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles.

The updated full-run config uses a separate managed_v3 output root. Run the
classic script's prepare/validate steps again before full evaluation; previous
audits intentionally do not accept changed workload hashes. Full evaluation has
not been restarted as part of this short profiling task.

## Concurrency

- Fast training config:128 shared CPU processes,87 load-affinity workers,
  2048sample slots,128initial credits. Its ordered loading/augmentation pipeline
  is not the priority-driven inference scheduler. Native selected-level sampling
  and the existing numerical inference/accumulation workers are shared.
- Classic inference:128tile-reader threads,256input slots,128result slots.
  Input512-cubed uint16 capacity64GiB plus8GiB output slots. Native float32/mask
  temporaries and completed reader arrays add memory; these are not the total RSS.
- Native process-wide cache:64source-read workers,8decode workers. These are
  separate from tile-reader threads, not multiplied by the number of collections.
  The Python runtime setters cannot exceed the underlying worker capacities.
- All100direction threads could already wait on inference simultaneously.
  workers32 constrained active trace CPU permits, not waiting jobs. workers0 now
  chooses automatic active CPU concurrency, bounded by cases and hardware.
- Ready inputs and results have independent lifetimes. Remaining queued tiles
  retain foreground/remaining-length priority. Admitted reads cannot be preempted;
  no claim of strict priority throughout all downstream queues.

Live inference_queues.jsonl samples roughly once per second while tracing or
draining; no fsync is introduced for this buffered diagnostic file. Currently
warm-cache/no-pool runs create no queue file, and failure paths rely on the
existing incomplete summary rather than a terminal queue snapshot. These are
explicit observability limitations, not completion or numerical changes.

Validation:64CPU regression tests include128concurrently blocked tiny readers,
216exact serial-oracle output tiles, small-result-pool backpressure, manager
donor/release barriers, failed/uncertain startup, reconnect, SIGTERM cancellation
and independent watchdog cleanup after launcher SIGKILL. Synthetic concurrency
is not a native I/O/GPU throughput measurement. No model precision changes.
