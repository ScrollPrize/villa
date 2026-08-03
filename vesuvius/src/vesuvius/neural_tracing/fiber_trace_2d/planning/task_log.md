# Task log: asynchronous rolling-accumulator flush

## Findings

- The current accumulator is already a bounded circular mmap band; it never
  allocates full logical output Z.
- `_record_commit` calls `_flush_group` synchronously at every canonical Z-row
  frontier. `_flush_group` reads, normalizes, finalizes, writes, clears, and
  releases before ordered result commit can continue.
- Multi-GPU workers can run briefly during a flush, but the coordinator stops
  recycling result slots and the bounded result window eventually stalls all
  devices.
- The current physical depth planner assumes immediate post-row flush and
  release. Async operation therefore requires capacity for one retained frozen
  interval in addition to the subsequent active write span.
- `_CircularZBand.read` already bounds temporary allocation to one requested
  output chunk. A background flush can preserve that bound while reading the
  frozen mmap directly; no band snapshot is required.

## Plan review

- Capacity planning will simulate the precise runtime schedule rather than
  conservatively but unjustifiably doubling the existing depth.
- One flush future is runner-wide and combines every scale group advancing at
  a row frontier.
- Submitted, completed, and released frontiers are separate; progress and slot
  reuse occur only after successful join.
- Coordinator-detached immutable activity descriptors prevent the worker from
  mutating live scheduling state. Exact dirty regions clear only after success.
- Every exit path must join the non-cancellable reader thread before mmap
  cleanup, while preserving an earlier coordinator exception.
- Tests and docs must cover per-chunk allocation bounds, adapter concurrency,
  exact persisted output/chunk presence, partial writes and rerun, final drain,
  progress timing, exhaustive planner agreement, and controlled overlap timing.

## Implementation and validation

- `_plan_circular_z_depth` now simulates one retained submitted interval and
  releases it only at the following advancing frontier. The real 256/32/96
  geometry with 64-voxel chunks is covered at inference factors 1, 2, and 4.
- The coordinator submits one combined, single-threaded flush future for every
  scale group advancing at a row. Activity is detached before submission;
  completion alone clears dirty rectangles, discards ring generations, updates
  counters/progress, and prints completion.
- Flush jobs carry descriptors and read/finalize/write one chunk at a time
  directly from the frozen mmap. No band-sized arrays or second mmap were
  introduced.
- Normal completion drains the final interval. Error/interrupt cleanup waits
  for the non-cancellable future before closing mmap files and preserves an
  existing coordinator exception with any shutdown failure as a note.
- Added aggregate `flush stats work=... wait=...` output and a final progress
  row after the final async completion.

## Validation results

- Focused overlap test blocked the first output write until inference entered
  the following Z band; it verified one active writer, unchanged `final_z`
  during the blocked write, exact three-chunk completion, and mmap cleanup.
- Forced background write failure reached the caller and left no mmap temp
  files after the reader had stopped.
- Multi-scale shared-runner testing submitted both scales in one flush and
  passed. Spawned two-worker output remained byte-exact against serial output;
  hard worker-exit handling still passed.
- Planner independence, initial-prefix, realistic 256/32/96 scale geometry,
  compilation, and diff checks passed.
- Controlled timing used five 4-voxel Z bands, 50 ms model delay and 50 ms
  output delay per band, three repetitions. Runtime was 0.313/0.304/0.305 s
  (mean 0.307 s) versus a 0.500 s serialized delay floor, demonstrating actual
  overlap in the synthetic workload.

## Deviations and limitations

- Representative whole-volume eight-GPU throughput was not measured because
  that dataset/runtime was not available to the agent test environment. The
  controlled benchmark establishes overlap but not production speedup.
- Several pre-existing Zarr-v3-backed tests intermittently stalled inside
  `zarr.open`'s global async I/O loop before entering the shared runner. They
  were terminated and are not counted as validation; array-backed shared-runner
  tests, including serial/spawn exactness, completed normally.
- The test environment has no pytest installation; validation used the existing
  unittest entry points and direct compile/diff checks without installing new
  dependencies.
