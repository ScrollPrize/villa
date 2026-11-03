# Persistent Fiber3D sample ownership

`sample_broker.CanonicalSampleBroker` owns the existing shared
`FiberTrace3DSampleQueue` across replaceable training consumers.
`cooperative_worker.QueueBatchSource` is the existing cancellable worker-side
transport. The [shared training loop](queued_training.md) accepts it through
`training_units(..., batch_source=source)`; it does not create another queue or
another implementation of training, losses, targets, tests or snapshots.

These components come from `3a4a7b2c1`. Test-sample lease reference counts come
from the frozen `2192cc5ac` implementation. This extraction keeps the current
train/test sources, validity masks and augmentation, without importing later
historical evaluation, refinement, embedding or tracing workflows.

## Ownership and completed updates

The CPU owner constructs the broker with an existing queue, or calls `create`
with a resolved loader config, a finite training horizon and the original test
selection. `create` uses the common loader/config parser and shared queue, not a
second loading path. Its CPU workers start parked until `set_worker_limit(n)`.
Parking finishes current jobs and keeps workers, datasets and samplers alive;
it does not discard queued samples or their canonical IDs.

The owner must call `poll()` regularly. Training batches are nonempty contiguous
canonical ranges within the live stream. `batch(indices)` returns `None` until
all their existing samples are ready. Reading a batch, delivering it through an
IPC queue, completing H2D, or finishing one microbatch **does not** release its
CPU slots. Only `commit(completed_step)`, after the complete global optimizer
update, advances the frontier by exactly one effective batch and releases those
slots. All ranks must have completed that update first.

The broker requires enough shared slots for the entire effective optimizer
batch and at least one logical test batch (or its smaller final-only group).
It bounds training admission at the next test phase before workers claim jobs.
This avoids blocking a test behind speculative training holding all credits.

If a consumer is replaced at a completed boundary, its GPU/IPC lookahead can be
discarded. The replacement starts from the broker's canonical frontier with a
new physical microbatch partition and the trainer's RAM handoff. Retained CPU
samples are reused, not randomly sampled or augmented again. Consumer shutdown
never closes the broker; the owner explicitly closes it after all consumers and
their IPC publishers have joined/drained.

The worker queue's `received` counter counts successful IPC receives only. It
helps its owner drain the old publisher and is not an optimizer-update commit.
`QueueBatchSource` validates every received batch against
`EffectiveBatchSampleIndex`; stale/out-of-order IDs raise instead of silently
altering sample membership. Closed views and an exited supervisor interrupt
polling waits. CPU staging owns tensor copies before returning test credits.

## Train/test transitions

The same workers and slot ledger service initial and periodic tests. Worker
sources call `begin_test(step)` and `end_test(step)` around the existing test
routine. The owner acknowledges each boundary only after all current ranks
agree. `begin_test` commits that update if it has not yet been committed.

Test views request original logical groups, including partial final groups.
`test_batch(step, indices)` grants a lease only when the sample data is ready.
`release_test` releases that reader, not every reader of those indices. The
per-sample reference count keeps overlapping readers' storage valid until the
last one releases it. Already-completed test samples can be requested again,
without counting them twice toward phase completion.

Pending test rows and the retained visualization batch live in the trainer's
RAM handoff. A replacement consumer skips completed groups, requests the rest
from the same broker, and ends the test only when all selected samples have been
consumed and every test lease returned.

The existing internal pipe messages are:

| Worker request | Owner response/action |
| --- | --- |
| `("begin-test", step, ())` | Coordinate ranks, call `begin_test`, reply `("begin-test", step)`. |
| `("test", step, indices)` | Wait for `test_batch` and send the CPU batch. |
| `("release-test", step, indices)` | Call `release_test`; no reply. |
| `("end-test", step, ())` | Coordinate ranks, call `end_test`, reply `("end-test", step)`. |

This is the existing internal data transport, **not** the locked public GPU
manager protocol. This stage does not implement its owner-side cohort pump,
manager registration/allocations, startup calibration or automatic GPU launch.
Callers must already hold an appropriate manager allocation before GPU work.
Aborted consumers with outstanding IPC requests require owner-side draining or
whole-owner cleanup; view cancellation alone is not a global lease rollback.

The loader workers retain their dataset/sampler objects. Construction of trainer
metadata objects inside a replacement training process is not eliminated by
this stage. No dataset metadata serialization, model defaults, precision,
supervision or learning-rate changes are introduced.

## Validation

With the cached real Paris4 fixture environment from
[asynchronous preparation](async_training_preparation.md):

```bash
cd vesuvius
python -m pytest -q tests/neural_tracing/test_sample_broker.py \
  tests/neural_tracing/test_worker_batch_source.py -k 'not cuda'
```

Optional CUDA cases use the existing manager-reserving test launcher, which
sets `FIBER_ASYNC_CUDA_TEST=1`, `FIBER_COOPERATIVE_TEST_GPUS` and the UUID device
mask before starting its child. Do not run them without a manager reservation.
They exercise the actual U-Net, AdamW, target prepass and asynchronous preparation
through a persistent CPU owner, replacing the GPU consumer during updates/tests.
The tests compare parameters/optimizer with the full effective-batch reference,
verify unchanged loader PIDs, and cover tiny credits, overlapping leases,
canonical IDs, cancellation and shutdown. Convolution backends are fixed for
strict numerical comparisons only; production optimized kernels stay unchanged.
Runtime validation is Linux-only, not a macOS/arm64 or performance claim.
