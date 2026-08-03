# Plan: process-parallel rolling-mmap flush

## Implementation

1. Replace `ThreadPoolExecutor` flush execution with explicit persistent
   spawn-context `Process` workers and bounded task/result queues. Do not use
   `ProcessPoolExecutor`/`Pool` feeder machinery. Start GPU workers first, then
   flush workers, then CPU tile reader threads. Serial inference starts flush
   workers before traversal/read activity. Partial startup terminates and joins
   only successfully started children and closes queues.
2. Describe each `_CircularZBand` by immutable absolute mmap path, dtype,
   shape, ring-depth, and channel metadata. Flush
   tasks contain scale, product/chunk bounds, logical-to-physical Z mapping,
   raw/weight mmap descriptors, and product identity only. Workers lazily open
   and cache read-only memmaps by path. No NumPy array or decoded/finalized
   chunk crosses a multiprocessing queue.
3. Validate production adapter pickling before traversal and initialize each
   worker with picklable model/output adapters; fail early and clearly for an
   unsupported custom adapter. Keep native
   BLAS/OpenMP/PyTorch CPU threading at one for the worker lifetime without
   changing the coordinator process. Each task reads one frozen chunk,
   normalizes/finalizes its dirty products, and atomically writes only that
   globally unique output chunk.
4. Preserve one runner-wide frozen batch across all inference scales. Keep its
   descriptors in coordinator memory but pump at most `2 * workers` through
   IPC. Pump opportunistically in serial and multi-GPU loops; at the following
   frontier block-pump until every result arrives. Each task exclusively owns
   one scale/chunk origin and all dirty products sharing its denominator. Empty
   intervals still advance frontiers. Validate frozen generation ownership
   before enqueue, collect the full batch before clearing/releasing any
   rectangle, and never force `memmap.flush()`.
5. Add `flush_workers` to the shared API and `--flush-workers` to both CLIs.
   Positive values select that many persistent processes; normal CLI default is
   the available CPU count capped at 64. Zero selects the exact old synchronous flush
   path, including its immediate-release planner, for repeatable A/B baselines.
   Reject negatives and keep scheduling bounded to one frozen batch.
6. Associate batch/task IDs with results and poll child exit codes even when no
   result arrives. On task error or hard exit stop enqueueing, terminate/join
   all workers, cancel queue join threads where necessary, then clean queues
   before mmap cleanup. Never clear/reuse the frozen interval or advance
   `final_z`. Preserve a primary coordinator/interrupt exception and attach
   shutdown failures as secondary notes.
7. Report flush process count, aggregate worker work, coordinator wait, chunks,
   and throughput. Remove the threaded `threadpoolctl` context entirely.

## Tests

- Assert tasks and pool queues contain descriptors/paths only, never ndarrays.
- Spawn process workers against real temporary mmap files and verify direct
  wrapped logical-Z reads, exact persisted bytes, distinct-process execution,
  and one-chunk allocation bounds.
- Compare `flush_workers=0`, one process, and multiple processes for exact
  output bytes/chunk presence across wraparound, multiple products sharing a
  weight, multiple scales, partial edges, sparse unsupported chunks, resume,
  and final drain.
- Use controlled per-chunk delay to prove the process pool both overlaps the
  next inference band and parallelizes chunk flushes. Measure repeated sync,
  one-process, and multi-process timings.
- Inject task exceptions, partial coherent-product writes, hard worker exits,
  and interrupts; verify error propagation, no premature ring release, mmap
  cleanup after worker shutdown, and resumable incomplete output.
- Verify CLI/default/zero/positive validation and forwarding for both Fiber and
  Lasagna. Run focused shared/front-end tests, compilation, and diff checks.
- Cover bounded task/result backpressure, empty and undersubscribed batches,
  sequential wrapped batches, pool startup failure, actual production-adapter
  spawn pickling, and tolerant concurrent invalidation of shared coarser
  pyramid chunks.

## Spec update

Replace thread-flush language with persistent spawn-process workers, mmap-path
descriptors, one-chunk-per-worker bounds, default/zero worker semantics,
process-local native thread limits, and batch completion/failure guarantees.

## Docs updates

Update `docs/code_structure.md`, Lasagna README, and 3D inference documentation
with process-pool architecture, `--flush-workers`, memory scaling, A/B baseline
mode, metrics, and failure behavior.

## Changelog and task log

Record the measured thread regression, reviewed design, implementation,
validation commands/results, controlled timing, representative benchmark still
required after implementation, and all deviations/limitations.
