# Task log: explicit Z-band shared inference pipeline

- Live Paris4 evidence showed that its already-started GPU, accumulator, and
  flush workers were alive and idle while all three result queues were empty;
  the coordinator continued its generic 50 ms GPU-result poll. This identifies
  scheduler quiescence, independently of the host's current thread pressure.
- The problematic flush payloads had all reached disk (305 current cosine
  chunks and 89 chunks for each normal channel), so the observed terminal state
  is coordinator bookkeeping/quiescence rather than compression or output I/O.
- Static inspection found that a fully acknowledged flush remains frozen until
  a later canonical Z-row calls `_advance_flushes()`. This couples ring release
  to a different event-state transition and leaves the generic idle poll unable
  to identify the actual unresolved owner.
- The task will replace that coupling with an explicit current-Z-band barrier
  and one previous flush lifecycle. No full-band copy, extra mmap, global tile
  list, numerical reorder, or Fiber/Lasagna implementation split is permitted.
- Independent review required active-band-only model reads/GPU/accumulation,
  a separate bounded live-cache descriptor ledger, explicit slot and flush
  ownership, nonblocking accumulator submission, and exact rollback of only
  the faulty elapsed-time watchdog/process-limit diagnosis.
- The shared runner now lazily schedules one Z band at a time. Its live-cache
  descriptor ledger may retain the configured 10,000 cheap future requests,
  while active full-tile reads and events remain bounded by the existing
  slot/read window and cannot cross the band barrier.
- Input/result slots now have checked `(band, sequence, stage)` ownership.
  Accumulator descriptors enter stable-owner queues incrementally; full queues
  yield to reader, GPU, accumulator-result, and flush-result pumping.
- Combined multi-scale flush batches now track immutable plans and disjoint
  queued/inflight/completed task IDs. Fully acknowledged and zero-task batches
  use the same immediate finalization routine as synchronous flush; failures do
  not clear/release frozen ring generations.
- Removed the elapsed flush watchdog and RLIMIT guess. The existing per-worker
  Zarr executor and BLAS/OpenMP/NumExpr/Blosc limits remain in place.
- Validation:
  - The focused shared-runner/ownership selection passed: 4 tests, 65
    deselected, in 21.26s.
  - The complete native/Zarr worker-limit file passed: 8 tests in 1.52s.
  - The live materialization/cache suites passed: 9 tests in 1.68s.
  - The main regression spans four Z bands and 16 dense tiles (> two slots and
    > the two-tile read window), forces out-of-order GPU completion, keeps a
    ten-descriptor live-download lookahead across band boundaries, uses two
    accumulator and two flush processes, and exactly matches serial output. It
    passed by itself with delayed live-cache completions in 12.53s.
  - Four Fiber 3D live-cache/prefetch/slot CLI tests passed, with 202
    deselected, in 2.41s, proving the Fiber caller still reaches the shared
    interface correctly.
  - `py_compile` and `git diff --check` passed.
- A wider Zarr-writing test run could not be used as final evidence because the
  shared host currently has about 18,222 threads, above the observed 16,384
  user task limit; more than 10,000 belong to a separate active training job.
  Even an isolated fresh `zarr.open_group` stalled in that host state.
- The production baseline supplied by the user did not complete: Paris4 became
  quiescent at 23,700/606,208 tiles with all workers alive and idle. Full
  Paris4 multi-GPU completion and throughput remain user-side validation because
  this sandbox does not expose the GPUs or the running dataset namespace.
- The first production rerun reached 27,747/606,208, then remained unchanged
  for more than an hour in the middle of a 4,096-event Z band. The coordinator
  main thread repeatedly polled a process connection with a 50 ms timeout.
  All eight GPU workers and all 32 accumulator workers were alive and blocked
  in `pipe_read` waiting for new input; all eight inference CUDA contexts were
  resident but showed no GPU work. Flush workers were idle and no flush was
  active at the terminal log frontier.
- The coordinator dump's parent `QueueFeederThread` instances all had empty
  buffers and were asleep. That refutes the proposed queue-to-pipe replacement;
  the evidence does not show a feeder losing a submitted descriptor.
- The failed run explicitly had `live_fetch.enabled=false`. Nevertheless,
  live-fetch commit `7a2a84d36` had changed the disabled path from atomic
  discovery/read submission to the new `awaiting_read` lifecycle. The repair
  restores the pre-live normal path and confines live materialization to an
  upstream bounded ledger feeding the same worker pipeline.
- TensorStore completed the exact first uncommitted tile and a 200-tile
  neighborhood in under 0.53 seconds, ruling out a persistent source/read
  defect at that coordinate. Stage-aware waits, per-worker task ownership,
  slot ownership, and liveness checks now make unresolved scheduler state an
  explicit invariant failure rather than an endless generic GPU-result poll.
- A Python-aware stack/locals capture of the next production freeze identified
  the actual cycle. `next_commit` and `next_accum_dispatch` were both 160589;
  the active events were 14 `ready`, 16 later `done_result`, and two
  `done_skip`, with no reads or assigned GPU work. All 16 result slots were
  therefore held by later results which canonical accumulation could not
  consume before ready sequence 160589, while that sequence could not enter a
  GPU without a result slot. No accumulator task or completion was missing.
- Rejected and removed an unproven acknowledged-completion retry experiment.
  Worker task/result queues retain their single fire-and-forget messages.
- GPU admission now reserves one input/result pair for the effective canonical
  frontier after a contiguous sparse-skip prefix. A direct full-16-slot
  reservation test, shared worker descriptor test, delayed/out-of-order exact
  shared-pipeline test, and TensorStore/Python-Zarr equivalence test passed (4
  tests, 58 deselected, 19.13s). Both live-cache suites passed (9 tests, 1.55s).
- The full combined Lasagna files are not green independently of this repair:
  the first isolated failure is an existing stale test patch target
  (`preprocess_cos_omezarr._auto_download` moved to the shared module), and a
  later pytest failure-rendering path segfaulted in NumPy array formatting.
