# Plan: TensorStore whole-volume inference prefetch

## Implementation

1. Add a shared input-reader abstraction to `lasagna.tiled_predict3d` with
   `tensorstore` and `python-zarr` backends. Open one TensorStore Zarr driver
   and one bounded context per inference run after child-process startup. Keep
   ordinary Python Zarr open for metadata, shape, support/resume checks, and as
   an explicit fallback.
   TensorStore import/open failures are explicit errors when that backend is
   selected; never silently change backends. Construct its context only after
   all GPU and flush children start because TensorStore starts native threads.
2. Split tile reading into exact clipped source bounds plus a shared finalizer
   that preserves current NumPy dtype and reflect-padding semantics. TensorStore
   submits `.read()` futures for clipped bounding boxes; the coordinator polls
   them without blocking and materializes only within a bounded prefetch window.
   Python-Zarr fallback uses the existing reader executor through the same
   event state machine. One-device inference also keeps a bounded canonical
   future window so TensorStore reads progress during synchronous GPU forwards.
3. Decouple prefetch events from reusable GPU input/result shared-memory slots.
   Lazily generate at most `prefetch_tiles_per_gpu * device_count` canonical
   events, default four per GPU. Completed TensorStore arrays wait in that
   bounded window. Copy a ready tile into shared memory only when an input slot,
   result slot, and GPU queue are available; release its temporary array
   immediately after the copy. Preserve two GPU/result slots per GPU by default.
   CPU/single-device mode counts as one device. Reject resume/skipped work before
   submission and never mutate coordinator state from TensorStore callbacks.
4. Expose shared/front-end controls:
   `--input-reader {tensorstore,python-zarr}` (TensorStore default),
   `--prefetch-tiles-per-gpu` (default 4),
   `--input-cache-gib` (finite, nonnegative; default 4),
   `--input-io-threads` (positive; default 16), and
   `--input-copy-threads` (positive; default 4). Keep `--prefetch-workers` for the
   Python-Zarr fallback and reject non-positive/negative values consistently.
   Serial inference uses the same selected reader synchronously without a deep
   queue.
5. Preserve bounded memory. Conservative peak is prefetch-window times padded
   input-tile bytes (every outstanding read may own a full result), plus the
   TensorStore cache, existing input/result shared memory, and request/cache
   overhead. Cancel all futures on failure/interrupt and wait/drain them before
   releasing referenced state. TensorStore is eligible only for local Zarr-v2
   filesystem array paths; unsupported/custom arrays fail clearly unless the
   caller explicitly selects Python Zarr. Do not create the full Cartesian list.
6. Extend normal statistics with input backend, submitted/completed reads,
   maximum live/ready reads, materialization/copy time, aggregate read latency,
   bytes/throughput, ready-queue high-water, GPU starvation/wait-for-input,
   submission-to-ready latency, and configured cache/I/O/copy concurrency.
   Keep existing GPU and commit metrics.

## Tests and measurement

- Compare TensorStore and Python-Zarr tiles byte-for-byte for interior,
  partial-border, fully outside, endian uint16, and size-one border axes while
  preserving existing NumPy reflect behavior (including errors).
- Compare complete serial and multi-device inference outputs exactly across
  both reader backends, including skipped/resumed tiles and canonical
  out-of-order completion.
- Prove lazy bounded read-ahead: a controlled reader may never exceed the
  configured event window, reads advance while GPU workers are busy, and GPU
  slots do not cap the prefetch window.
- Test TensorStore import/open/read errors, future errors, interrupts, explicit
  fallback, local-Zarr eligibility, validation, cleanup, child-before-context
  startup order, and both CLI forwarding paths.
- Run compilation, focused shared/front-end tests, diff checks, and repeated
  Python-Zarr versus TensorStore benchmarks on the same local Zarr. Report
  command, shape/chunks/dtype/crop/tile/overlap, warmups, iterations, and
  mean/p50/p95 or min/median/max for raw reader throughput and controlled
  end-to-end inference. Sweep prefetch depth, I/O concurrency, and cache before
  leaving the user's representative eight-GPU run as authoritative validation.

## Spec update

Replace CPU/Zarr-reader language with the shared asynchronous TensorStore
bbox-reader contract, independent bounded prefetch and GPU-slot windows,
fallback behavior, exact padding/dtype guarantees, cache/concurrency controls,
and failure cleanup.

## Docs updates

Update `fiber_trace_2d/docs/code_structure.md`, `lasagna/README.md`, and
`lasagna/docs/3d_unet_training.md` with the new pipeline, memory accounting,
CLI controls, metrics, and backend fallback.

## Changelog and task log

Record the TensorStore reader integration, measured controlled results,
validation commands, deviations/limitations, and the need for a representative
eight-GPU rerun.
