# Ordered CPU sample queues

`vesuvius.neural_tracing.ordered_sample_pipeline` provides the CPU scheduling
foundation for shared training and evaluation loading. It is independent of
fiber annotations, augmentation recipes, model outputs and GPU allocations.
This extraction does **not** switch the existing trainer to this queue yet.

## Fiber3D preparation integration

`fiber_trace_3d.sample_pipeline` supplies the existing load/augment factories,
rank-local `FiberTrace3DTrainingBatchPipeline`, and source-tagged
`FiberTrace3DSampleQueue`. They use `FiberTrace3DLoader.load_raw_sample` and
`augment_raw_sample`; the serial `load_sample` composes those same stages.
Serial and queued batches both use `collate_fiber_trace_3d_samples`.

This layer is extracted from the CPU preparation stage before CUDA augmentation
and the later annotation/refinement/embedding features. Its masks, fixed-sigma
augmentation, targets and sampling remain that stage's existing behavior.
The normal training CLI's replacement by the latest unified owner is deferred
until its GPU preparation and effective-batch dependencies are extracted;
the old trainer has not been rewritten as an alternative integration.

`geometry_backend` selects `auto`, `native` or `eager`. Native CPU generation
fuses affine/inverse-smooth coordinates and validity using `vc.volume`;
eager generation remains the reference and non-CPU fallback. `geometry_tile_depth`
can bound the eager temporary grid. `value_augmentation_backend` selects
`auto`, `native` or `torch`. Explicit native selection reports a missing or
ineligible binding as an error; auto may use the existing fallback. Rebuild
`vc_volume` to expose all three [native kernels](../../volume-cartographer/docs/native_training_kernels.md).

Value operations keep the existing order: mask-aware normalization, contrast,
brightness, gamma, seeded noise, isotropic blur, directional blur, then invalid
pixel zeroing. Native float32 results are tested against the Torch reference
with the source implementation's existing rounding bounds; same-backend serial
and queued results are exact. This does not import the later randomized-sigma
GPU augmentation recipe or remove validity masks.

`volume_io_threads`, `volume_decode_threads` and `volume_sample_threads` expose
the shared sampler's native thread controls. Queue load workers initialize the
native cache before opening volumes and force one I/O, decode and sampling
worker. Native CPU compute and Torch workers are single-threaded. Separate
loader processes retain their own caches; this API does not imply a cross-process
decoded cache.

The shared tensor bank keeps large volumes **and validity masks** out of queue
payloads; only metadata and generation-checked descriptors cross the queues.
Aligned batches are zero-copy views. A partial batch wrapping the ring copies
only its two ring spans. Canonical sample indices drive random seeds, while
transport ordinals identify slot ownership independently.

For a shared train/test pool, construct resolved `FiberTrace3DConfig` objects
once and pass them through named sources:

```python
owner = FiberTrace3DSampleQueue(
    sources={"train": SampleQueueSource(train_config),
             "test": SampleQueueSource(test_config, sample_mode="flat")},
    workers=4, slot_count=8, initial_in_flight=2,
)
try:
    train_view = QueuedBatchView(owner, "train", train_batches, limit=next_test_boundary)
    # Consume the permitted train batches, release_current(), then consume a
    # finite test view over the same owner before extending the train limit.
finally:
    owner.close()
```

Keep views and their in-flight batches within the owner's lifetime. Returned
batch storage is leased, not a persistent snapshot: copy tensors retained for
visualization or analysis before releasing their ticket. Closing a view wakes
its reader; closing the owner stops/drains its workers. Dataset parsing and
sampler objects are lazy per source per persistent worker, not rebuilt at phase
boundaries. GPU lifetimes and training interruption handling remain the caller's
responsibility at this extraction boundary.

## Ownership and progress

- Stage one maps an integer sample index to a value; optional stage two maps
  that value to its final form. Factories create each operation inside its
  worker, so native readers and caches need not be pickled.
- Fixed mode assigns separate loader and augmenter processes. Shared mode
  uses one persistent pool with stage affinity and worker-side work claiming.
  A shared worker processes its loaded value itself if publication would block
  on a full intermediate queue. One active worker can execute both stages.
- One credit window bounds all claimed samples, including queued, executing,
  completed out-of-order and leased samples. Results are always delivered in
  index order; the coordinator drains other completions while awaiting that
  index. A slow early sample therefore cannot deadlock result publication.
- Shared mode can grow lookahead from `initial_in_flight` to `max_in_flight`.
  Every completed load advances the admission frontier by two, within the
  finite stream and credit limits. This avoids filling a large window before
  augmentation starts.
- `release_credit=False` retains a lease until `release(index)`. With a custom
  transport, retain it until all consumers have finished reading its storage.
  Slot ownership prevents wrapping around onto a still-leased sample.

Use `spawn` (portable) or `forkserver` where available. `fork` is rejected to
avoid inheriting an initialized Torch/native runtime. Apply
`native_thread_environment()` before creating workers if native libraries are
imported at startup; workers also limit native/Torch thread counts to one.
Do not call CUDA from these CPU workers.

## Basic use

Factories and operation classes must be importable, pickleable definitions,
not lambdas or nested functions. Launch from an `if __name__ == "__main__"`
guard when using multiprocessing:

```python
from vesuvius.neural_tracing.ordered_sample_pipeline import (
    OrderedTwoStageSamplePipeline,
)

if __name__ == "__main__":
    with OrderedTwoStageSamplePipeline(
        stage1_factory=LoadFactory(),
        stage2_factory=AugmentFactory(),
        start_index=0, item_count=100,
        stage1_workers=0, stage2_workers=0, shared_workers=4,
        initial_in_flight=2, max_in_flight=8,
        raw_queue_capacity=2, result_queue_capacity=2,
        multiprocessing_context="spawn",
    ) as pipeline:
        for _ in range(100):
            result = pipeline.next_ordered()
            consume(result.index, result.value)
```

For load-only work, set `shared_workers=0`, positive `stage1_workers`,
`stage2_workers=0` and `stage2_factory=None`. For separate stages, set both
stage worker counts positive and leave `shared_workers=0`.

`OrderedPipelineResult` includes per-stage wall/CPU service times, queue and
reorder waits in milliseconds, worker IDs and monotonic timestamps in
nanoseconds. These describe queue operations, not end-to-end GPU throughput.

### Pause, resize and phase boundaries

In shared mode, `elastic_workers=True` allows `set_worker_limit(n)` with
`0 <= n <= shared_workers`. Current work finishes before surplus processes
park; `workers_within_limit()` confirms that transition. Keep calling
`poll_ready()` while draining/paused so completed workers can publish. CPU
datasets and worker PIDs stay alive. Resuming even with one worker preserves
progress and deterministic delivery. GPU allocation remains the caller's job.

`manual_admission=True` starts with no work admitted. Publish initialized
descriptors with `admit_until(exclusive_end)`; this frontier only advances.
Credits and startup ramp-up still apply. Polling does not admit work.

`fiber_trace_3d.queued_batches.QueuedBatchView` provides an ordered batch view
over a shared owner exposing:

| Owner method | Contract |
| --- | --- |
| `submit(source, indices)` | Return a ticket, or `None` if credits are unavailable. |
| `poll()` | Drain available work without blocking on future batches. |
| `batch(ticket)` | Return the ready batch, or `None` while pending. |
| `release(ticket)` | Release the completed batch's storage lease. |

Views own no workers. Advance the training view's `limit` only through the next
test boundary; release its current batch before reading test batches from the
same owner. This prevents speculative training from consuming all credits
ahead of evaluation. A view supports a single consuming reader; another thread
may advance its limit or close it. `close()` wakes a waiting reader, but does
not release pending in-flight tickets: the owner must drain/reclaim them.
`TrainingBatchIndices` maps physical batch sizes lazily; `test_batches` maps
fixed evaluation batch indices without changing their selection.

### Transport and shutdown

Optional transports expose `encode(index, value, final=...)` and
`decode(index, descriptor, final=...)`. Shared transports additionally expose
`slot_count >= max_in_flight` and `_location(index) -> (slot, generation)`;
their location mapping must agree with the queue's cyclic slot ownership.
Transport storage is caller-owned; the generic queue never allocates a GPU
buffer or knows the payload layout.

Worker exceptions (including stage/index in shared mode) and unexpected exits
are reported to the coordinator. Startup and ordered-progress timeouts bound
waits. For an intentionally paused pool or an unadmitted phase, use polling
rather than blocking `next_ordered()` across its progress deadline.

Always close the queue, preferably with a context manager. Shutdown sets a
cooperative stop event, then escalates to terminate/kill after bounded waits.
It avoids queue feeder joins and locks potentially left held by dead workers.
An unkillable worker is reported as an error rather than falsely releasing its
resources. The coordinator API itself is single-owner, not thread-safe.

## Validation

From `vesuvius/`, using an existing environment:

```bash
PYTHONPATH=src python -m pytest -q tests/neural_tracing/test_ordered_sample_pipeline.py tests/neural_tracing/test_queued_batches.py
```

Regression coverage includes out-of-order completion with tiny queues,
exceptions and hard/clean worker exits, saturated shutdown, retained storage
leases, one-worker operation, zero-worker parking/resume, explicit admission,
and train/test/train phase transitions.

To include real CT transport, set `VC_QUEUE_TEST_VOLUME` to a local array
readable by `vc.volume.Volume.open` and `VC_QUEUE_TEST_ORIGIN_ZYX` to an occupied
crop origin in that array's coordinates. Four overlapping 32³ patches and a
deterministic axis flip must match direct serial reads byte-for-byte in fixed
and shared modes, using both `spawn` and `forkserver` where supported. These
tests skip without an explicit fixture; no downloads or source writes occur.
They validate transport and scheduling, not a throughput or model-quality gain.

`tests/neural_tracing/test_sample_pipeline.py` carries the original CPU-stage
regressions plus source-switch, ring-wrap/partial-batch, failure cleanup and
resume-index tests. Set both `VC_QUEUE_TEST_VOLUME` (a concrete local CT array)
and `VC_QUEUE_TEST_FIBER` (an actual annotation in its original base coordinates)
to include the Paris4 loader check. It exposes the concrete array as a temporary
level-zero group without copying CT chunks and compares fully augmented native
train/test/train batches exactly. The fixture's affine scales the annotation
by one half for the existing level-one CT array. Neither source is changed.

The extraction was validated on Ubuntu/Python 3.14 with 27 passing tests,
including real Paris4 CT patches at pyramid level 1. Both scheduling modes
and both start methods matched direct reads exactly. macOS/arm64 runtime
validation remains pending; no GPU workloads or numerical settings changed.
