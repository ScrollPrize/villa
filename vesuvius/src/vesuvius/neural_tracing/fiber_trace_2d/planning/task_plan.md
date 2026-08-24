# Plan: explicit Z-band shared inference pipeline

## Invariants and band model

1. Introduce a coordinator-owned Z-band record covering exactly one canonical
   model-tile Z position. It records the lazy sequence interval, expected and
   completed tile events, outstanding reads, assigned GPU work, outstanding
   accumulator tasks, and owned shared result slots.
2. Keep the band's Y/X lattice lazy. Materialize and read only the existing
   bounded windows within the active band; never retain the whole plane or the
   volume-wide Cartesian tile list. Do not read, assign GPU work, or accumulate
   a later band until the active band completes.
3. Complete a band only after every event is skipped or accumulated, every
   accumulator acknowledgement is received, and every input/result slot owned
   by the band is returned. Assert these invariants at the transition.
4. Advance progress and live-cache safe Z only from the completed band
   transition. Preserve canonical tile order and current numerical accumulation
   order.
5. Track input/result slots in an explicit ledger mapping each owned slot to
   `(band, sequence, stage)`. Reject duplicate or foreign releases, and require
   every slot to be free at a band transition.

## Flush lifecycle

1. Keep at most one previous frozen flush batch while the current Z band is
   computed, using the existing single enlarged circular mmap and no band copy.
2. Represent each chunk-aligned, possibly empty, multi-scale flush interval as
   an explicit immutable-plan batch with expected task count and disjoint
   queued, inflight, and completed task IDs.
   Drain chunk acknowledgements continuously. As soon as the entire batch is
   complete, normalize its accounting, clear only its dirty rectangles, release
   the exact circular generations, and advance durable `final_z`.
3. Before submitting the current band's newly flushable interval, join/finalize
   the preceding batch if it has not completed. This is the only intended flush
   backpressure point.
4. Preserve descriptor-only spawn-process tasks, stable chunk ownership,
   chunk-sized float32 temporaries, atomic OME-Zarr writes, multi-scale combined
   batches, resume behavior, and `flush_workers=0` synchronous semantics.
5. Use one finalize/release/accounting routine for synchronous and process
   flushes. A failed batch never clears or releases frozen generations, and a
   zero-task batch finalizes immediately.

## Pipeline failure and diagnostics

1. Replace the generic no-progress wait on the GPU result queue with
   stage-aware pumping of GPU, accumulator, reader, and flush completions.
   Submit accumulator tasks incrementally and nonblocking: preserve stable-owner
   FIFO while a full owner queue yields to the other stages.
2. Treat a quiescent coordinator as an invariant failure when unresolved work
   exists but no read, worker task, accumulator task, or flush task can produce
   progress. Report the band bounds, event-status histogram, sequence
   frontiers, free/owned slots, outstanding reads, accumulator tasks, flush
   batch counts, queue depths where supported, and worker liveness.
3. Worker exceptions and hard exits remain immediate failures. Diagnostics must
   identify the actual stalled stage and must not claim process-limit exhaustion
   without a caught process/thread creation error.
4. Remove the workaround flush timer and its unsupported process-limit claim.
   Retain one native BLAS/OpenMP/NumExpr/Blosc thread and the bounded Zarr
   executor per process.

## Live-fetch scheduler regression follow-up

1. Keep the established bounded `multiprocessing.Queue` worker transport. The
   process dumps show empty parent feeder buffers and do not support replacing
   the transport as the cause or cure.
2. Restore the pre-live-fetch normal-input lifecycle: discovering a supported
   local tile and submitting its bounded read happen atomically in one
   coordinator pass. The optional live cache is only an upstream bounded
   materialization ledger; after materialization it enters the same reader,
   GPU, accumulator, flush, and canonical-commit stages as a local tile.
3. Keep explicit per-worker sequence/task ownership and exact input/result slot
   ownership while using the existing queues. Validate ownership and worker
   liveness on every coordinator pass, and reject duplicate, foreign, or
   mismatched acknowledgements immediately.
4. Retain stage-aware quiescence handling. Never wait only on the GPU queue
   when the producer that can advance the head event is a live-cache future,
   input read, accumulator task, or flush task.
5. Keep Z-band ordering, TensorStore, stable accumulator ownership, asynchronous
   flush overlap, mmap layout, and numeric behavior unchanged.

## Canonical-frontier capacity invariant

1. Preserve canonical accumulator dispatch and its existing floating-point
   accumulation order. Do not process later GPU results out of order merely to
   release their shared result slots.
2. Compute the effective accumulation frontier by advancing past the contiguous
   `done_skip` prefix beginning at `next_accum_dispatch`. When that effective
   frontier still needs GPU processing (`fetching`, `awaiting_read`, `reading`,
   or `ready`), reserve one free input slot and one free result slot for it. A
   later ready event may be admitted only when the reservation remains
   available. Once the effective frontier is assigned, its owned pair satisfies
   the reservation and all other free pairs may be used.
3. Express the reservation decision in a small coordinator helper so its
   behavior is directly testable. The reservation consumes no additional RAM,
   mmap space, workers, or queues and reduces peak later-tile occupancy by at
   most one slot only while the frontier read is outstanding.
4. At every no-progress pass, reject the proven circular-wait state immediately:
   an effective frontier that still needs GPU work, no free result slot, and
   every result-slot owner belonging to a later GPU-stage event. A slot owned by
   the current/earlier accumulator is legitimate backpressure and must not
   trigger the assertion. Report the raw/effective frontier, skipped prefix,
   status, free-pool counts, producer counts, and complete slot ledger. Do not
   label that state an accumulator hang.
5. Keep work/completion delivery fire-and-forget. A successfully queued task has
   one terminal result; do not add acknowledgement/retry state or make optional
   input-release notices part of a retry protocol.

## Tests and validation

1. Add deterministic scheduler tests with delayed/out-of-order fake GPU work,
   process-parallel accumulation, sparse skips, multiple products/scales, and a
   delayed real Zarr-v2 flush. Use enough events to cross several bounded
   windows and Z-band transitions.
2. Assert that parallel output matches the synchronous runner, every shared
   slot is returned once, a completed flush releases immediately, only one
   flush overlaps compute, and no Cartesian tile list is materialized.
   Cover initial no-op prefixes, circular wrap/reuse, a band that does not
   advance a chunk frontier, divergent multi-scale frontiers, and a dense real
   Zarr-v2 flush whose tasks queue in waves.
3. Add a forced quiescent-state test that validates the stage-rich diagnostic,
   plus existing worker exception/hard-exit coverage.
4. Run the focused Lasagna tiled-inference, Zarr thread, live-cache, Lasagna
   predict3d, and Fiber 3D inference suites. Run a bounded representative smoke
   using TensorStore plus process accumulation/flush if available; record its
   command, input, wall time, and throughput before/after. Report if the full
   Paris4 GPU workload is left for user validation.
5. Exercise both local and live-cache modes beyond each bounded read/slot
   window, force out-of-order GPU and accumulator completion, and assert exact
   serial equivalence plus one acknowledgement, slot release, and canonical
   commit per descriptor. Retain hard-exit coverage for both worker stages.
6. Add a focused capacity regression in which the canonical frontier read is
   delayed while at least one full result-slot window of later reads completes.
   Assert that later admission leaves one pair reserved, the frontier is then
   admitted, canonical accumulation advances, and all slots are ultimately
   released without retries. Cover a contiguous `done_skip` prefix before the
   delayed frontier, one total slot, and a current/earlier accumulating owner
   that must be treated as recoverable backpressure.

## Spec update

Replace the conflicting shared-runner, ordered-event, and process-flush bullets
in `specs.md` with an
explicit one-Z-band coordinator contract: canonical completion barrier, exact
slot ownership, one previous asynchronous flush, immediate completed-flush
release, canonical-frontier slot reservation, and stage-aware quiescence
failure. Clarify that the coordinator owns scheduling/frontiers while flush
processes only execute distinct chunk writes.
Retain all existing memory,
numeric, lazy-generation, sparse/resume, shared Fiber/Lasagna, and live-cache
requirements.

## Docs update

Update the shared 3D tiled-runner section in `docs/code_structure.md` to explain
the explicit Z-band lifecycle, read-ahead boundary, slot ownership, asynchronous
flush overlap, and diagnostic behavior.

## Changelog

Record the replacement of the cross-stage event/flush scheduler with a bounded
explicit Z-band pipeline and removal of the incorrect timeout diagnosis.
