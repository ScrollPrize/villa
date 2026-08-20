# Task log: reliable and parallel on-demand fiber replay

## Initial state and reproduction

- The worktree already contains the uncommitted on-demand fiberlet storage and
  cache implementation from the preceding task. Those changes are retained.
- The CLI currently prints `FiberTraceProgress` directly. Its `step/maxSteps`
  fields are local to one greedy restart segment, while failure arc/fraction is
  global to the replay interval. The missing segment identity makes successive
  denominators appear to be changing whole-run targets.
- The main thread joins the greedy future first and then the fiberlet future,
  without printing independent evaluator completion.
- Reproduced the user's exact final greedy sequence against the existing full
  Paris4 sparse caches. This checkout then terminates with
  `cached fiberlet adjacency failed: required fiberlet anchor chunk did not
  resolve to data`. The fiberlet generator currently discards the dependency
  chunk key, `ChunkStatus`, and underlying `ChunkResult.error`, obscuring the
  actual failure and making the silent interval look like a hang.

## Captured deadlock

- The user captured the stalled live process with GDB. The fiberlet cache worker
  was blocked waiting for an anchor chunk; the anchor chunk parent was joining
  its fit workers; one fit worker was asleep in the ready-cell condition
  variable.
- Scheduler state was already terminal: `nextReadyCell=7776`,
  `readyCells.size()=7776`, `completedTiles=64`, and
  `partitionTileCount=64`. This proves a missed completion notification rather
  than I/O, missing workers, or unfinished computation.
- `remainingCells` and `completedTiles` were atomically modified and notified
  without holding `readyCellMutex`. A waiter could evaluate its predicate as
  false while holding the mutex, then miss a notification issued before it
  actually entered the wait.
- Both completion counters now use the same mutex as the condition-variable
  predicate. No heartbeat, polling, timeout, or recovery behavior was added.
- Built `test_fiber_anchors` and `vc_fiberlets` with `-j32`; all 84 anchor tests
  pass.

## Expanded scope

- The user also requires globally coherent replay progress and parallel work
  inside one on-demand fiberlet chunk. The current candidate enumeration phase
  in `traceFiberletPaths` is serial even though later preparation and search
  phases use `pathConfig.parallelThreads`.

## Implementation and focused validation

- Independent plan review required exact serial/parallel artifacts, isolated
  per-chunk CPU measurement, explicit evaluator terminal states, and preserving
  serial semantics for arbitrary predicates. Source ownership is evaluated in
  canonical order; generic point/pair predicates keep serial enumeration.
- Candidate enumeration now assigns canonical source indices to workers and
  merges per-source candidates/counters in source order. The selected output,
  graph ordering, and per-candidate arithmetic do not change.
- Greedy replay progress now reports global reference arc/fraction, restart
  segment, and explicitly local step/budget. Fiberlet graph replay reports the
  same global axis. Both async evaluators print completion or failure before
  the main thread joins them.
- Generated cache rows report stable schedule index, nearest global reference
  arc, monotone generated count, scheduled count, and internal fiberlet phase.
  Completed rows include measured wall/CPU time and candidate-generation
  effective cores. Cached chunks are not misreported as newly generated.
- The first instrumented Paris4 chunk revealed a serial teardown after search:
  151,802 independent prepared-candidate vectors retained about 11 GiB and were
  destroyed on the caller thread after the phase timer stopped. Each search
  worker now releases its own prepared candidate after producing the result.
- For Paris4 chunk `107,34,45` with 105,730 input anchors and 63,932 stored
  fiberlets, wall time changed from 34.7949 s to 4.3440 s. The new run used
  122.665 CPU-s overall; candidate generation used 0.1075 s wall and 1.2423
  CPU-s, or 11.55 effective cores. The prefix and route payloads are
  byte-identical (`8b1cec2b...` and `806eefaf...`).
- The 64-tile/32-worker anchor regression completed 50 consecutive runs. All
  85 anchor tests, 54 trace tests, 10 storage tests, and 28 cache tests pass.
  `test_fiberlet_paths` has no new failure from this work but still reports 295
  existing bit-exact local-scoring checks at line 406 in this checkout.

## Validation command

```bash
volume-cartographer/build/bin/vc_fiberlets fiberlet-replay /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json /tmp/fiberlet-replay-debug --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --anchor-cache /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiberlet-replay-full/cache/anchors.zarr --fiberlet-cache /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiberlet-replay-full/cache/fiberlets.zarr --radius 64
```
