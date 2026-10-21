# Fiber3D shared train/test queue integration

The trainer now uses the existing `FiberTrace3DSampleQueue` and
`QueuedBatchView` for both training and dense evaluation. This is the
source-tagged scheduling from development commits `653fac7e0`/`3a4a7b2c1`, with
the asynchronous preparation hookup from `3f0aeecaa`. It does not introduce a
second loader, a new augmentation recipe, or a new training objective.

## What stays the same

The dataset selection, canonical random seeds, fixed-sigma augmentation,
validity masks, targets, model/precision defaults, rank-local batch size, DDP
loss reduction, test selection and metric/snapshot scheduling are unchanged.
Each training rank still consumes its original canonical sample stream:
`((step - 1) * world_size + rank) * batch_size + offset`.
Checkpoint resume advances that stream from the saved update, as before.
The full initial test and periodic tests use the same queue/evaluator.

This stage does not enable cooperative resource handling, dynamic microbatch
sizes, global effective-batch accumulation, or new model output resolutions.
The legacy whole-batch DataLoader helper remains available to the existing
standalone benchmark, not production training. GPU callers must already hold a
manager allocation; these loading components never acquire resources.

## Worker and buffer controls

Existing dataset and training configs remain usable. These development controls
configure the queue:

| Training setting | Meaning / default |
| --- | --- |
| `sample_pipeline_workers` | Total CPU workers across ranks. Zero/omitted uses `loader_workers`, with at least one worker per rank. |
| `sample_pipeline_slots` | Shared sample credits across ranks; default 512. Each rank needs at least one train/test batch, so tiny requests are raised to that minimum. |
| `sample_pipeline_initial_in_flight` | Startup lookahead; default 16, bounded by local slots. |
| `sample_pipeline_cuda_prefetch_depth` | Future batches on the private preparation stream; default 0 (synchronous). Positive values require CUDA. |
| `sample_pipeline_cuda_value_augmentation` | Optional `"blur"`/`"all"` suffix; requires positive CUDA depth. Omitted uses the existing masked CPU recipe. |

Workers and shared-slot memory persist across train/test phases. Each CPU worker
initializes each source's dataset and sampler lazily once. Queue workers stay
CPU-only and use the existing single-core native/Torch limits. The per-rank
worker partition is deterministic. For a single worker, the existing queue's
one-worker affinity is used; it can perform both load and augmentation stages.
This replaces `loader_worker_device`, `loader_prefetch_factor` and whole-batch
loader multiprocessing choices in production, without changing their standalone
benchmark behavior. The common queue uses `forkserver` (available on the
supported Ubuntu/macOS hosts); the lower-level queue API also supports `spawn`.

CPU preprocessing is the shared production path. An old configuration that
deliberately generated coordinates/value augmentation on CUDA workers (or used
the old serial GPU fallback with zero workers) now preprocesses on CPU too.
CPU and CUDA RNG implementations are not bit-identical even with the same
canonical seeds. Exact comparisons must use the same preprocessing backend and
device, rather than claiming cross-device augmentation identity.

## Evaluation cannot be blocked by speculative training

Training admission stops at the next test update, regardless of the configured
lookahead or CUDA preparation depth. Evaluation uses finite views over the same
owner, including its partial final batch and ranks with no test batches. Only
after that test phase does the training limit advance. A preparation reader
waiting at that limit holds no queue lock, so evaluation can consume/release
the same credits. No queue or worker rebuild is needed.

CUDA preparation is paused during evaluation and resumed afterwards. Its retained
device batches still occupy memory; this local pause is **not** a manager GPU
release. Transfer and target preparation timing come from the existing stage.

Returned CPU batches own their tensor storage before releasing a shared-slot
lease: `.to("cpu")` alone would alias that storage. Synchronous CUDA transfers
complete before releasing the lease; asynchronous preparation first takes a
pinned owned copy. Captured test batch zero therefore survives ring reuse for
visualization. Closing a test view does not close the shared workers.

`max_steps=0` retains open-ended training: the existing lazy batch-index mapper
provides a practically unbounded stream without materializing a job list.
Shutdown closes a waiting preparation reader, then the queue, writer and process
group, including when the initial test or a later update fails.

## Validation

From `vesuvius/`, use an existing environment and the real CT/annotation fixture
variables described in [asynchronous preparation](async_training_preparation.md):

```bash
python -m pytest -q tests/neural_tracing/test_training_queue_integration.py -k 'not cuda'
```

Manager-reserved CUDA children can run the optional tests:

```bash
FIBER_ASYNC_CUDA_TEST=1 python -m pytest -q \
  tests/neural_tracing/test_training_queue_integration.py -k cuda
```

The reserving launcher must set `CUDA_VISIBLE_DEVICES` and
`FIBER_COOPERATIVE_TEST_GPUS` before the child initializes CUDA. Two granted
devices are needed for distributed cases. Tests run the actual small U-Net,
production losses and AdamW on real Paris4 CT/annotation. They compare serial
and queued updates, test metrics and snapshots, including resumed sample IDs,
tiny rings, partial/empty test ranks, repeated phases, visualization, exceptions,
open-ended training and shutdown. CUDA tests include deferred `blur`/`all`
augmentation and two-rank NCCL. No full-model throughput or convergence claim is
made; Linux runtime validation does not establish macOS/arm64 runtime parity.
The single-GPU optimizer oracle fixes cuDNN convolution algorithm selection in
the test only. Production determinism/precision flags are unchanged.
