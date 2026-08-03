# Plan: asynchronous flush over one enlarged mmap ring

## Implementation

1. Separate flush preparation/completion from the existing synchronous
   `_flush_group` operation. The coordinator will identify and detach the
   finalized dirty-chunk activity in canonical order, freeze that logical Z
   interval, and submit exactly one background flush job that reads the
   accumulator mmap directly one output chunk at a time.
2. Retain the existing single `_CircularZBand` per product/weight. Extend ring
   planning by simulating the exact runtime order: accumulate the whole current
   canonical Z row; at an advancing frontier complete/release the prior flush;
   submit the current finalized interval without advancing the origin; then
   accumulate the following row while that interval remains frozen. Capacity
   is the maximum write end minus oldest unreleased origin under that schedule,
   not an assumed doubling. Do not copy live overlap, finalized bands, or whole
   mmap regions. Keep physical capacity bounded and independent of full Z.
3. Permit only one runner-wide flush job at a time, including across distinct
   inference-scale groups. At each row frontier, prepare every advancing group,
   join the previous combined job once, then submit one combined job. This is
   the bounded backpressure point when storage cannot keep up.
4. Track submitted/frozen, completed/written, and released/origin frontiers
   separately. The coordinator detaches an immutable canonically sorted list
   of chunk activity before submission. The worker may read only those frozen
   mmap regions and write their unique output chunks; it must not inspect or
   mutate live activity/resume/support state, clear/discard slots, update
   counters, or update progress. On successful join, the coordinator clears
   the exact detached dirty rectangles, discards generations, advances the
   released and completed frontiers, and applies returned counts/timing in that
   order. On failure it performs none of those release mutations.
5. Drain the final job before metadata/pyramid creation and during normal
   completion. Because a Python thread cannot be cancelled safely, every error
   and interrupt path must wait for an active flush reader before mmap cleanup.
   Preserve an original coordinator exception if flush shutdown also fails and
   attach/report the flush error as secondary context. A repeated interrupt
   must not race mmap close/unlink against the reader. Limit native worker
   threading to one so finalization/compression does not fan out unexpectedly.
6. Preserve per-output-chunk RAM bounds: `_CircularZBand.read`, `np.stack`, and
   finalized channel arrays may exist for only the chunk currently written;
   jobs contain descriptors, never decoded chunks or finalized arrays. Record
   separate flush work and coordinator wait timing so overlap and residual
   backpressure are visible.
7. Treat asynchronous adapter use as a shared-runner contract: output
   completeness checks may overlap distinct-chunk writes, while model inference
   may overlap stateless product finalization. Document it and keep the shipped
   filesystem/output and model adapters safe for these disjoint operations.

## Tests

- Add a deterministic slow output adapter test proving inference begins
  accumulating the next Z band while the previous band write is blocked.
- Assert only one flush is active and the next flush frontier waits for it.
- Compare synchronous-reference and asynchronous outputs exactly for wrapped
  rings, multiple products/scales, sparse unsupported chunks, and resume.
- Inject a background write/finalization failure and assert it reaches the
  caller before frozen slots are reused; verify cleanup completes.
- Assert planned mmap depth remains independent of full Z and exactly matches
  an exhaustive small-lattice simulation of one-flush lag. Cover the real
  tile-size 256, border 32, overlap 96 and supported scale/chunk combinations.
- Compare persisted channel bytes and chunk presence across sync-reference and
  async runs covering wraparound with a frozen interval, shared-weight products,
  multiple scales in one combined job, partial edges, unsupported chunks,
  independently resumed sibling products, and final end-of-stream drain.
- Assert final-Z progress, written/cleared counters, and completion messages do
  not advance while a flush is blocked and advance exactly once on join.
- Inject a partial coherent-product write failure and verify the frozen region
  is retained through shutdown and a rerun detects/rebuilds the incomplete
  product through existing atomic output semantics.
- Run focused shared-runner tests, both frontend forwarding/smoke tests, Python
  compilation, and diff checks. Measure repeated fixed synthetic workloads with
  controlled write delay before/after overlap and report elapsed/work/wait
  timing; explicitly note that representative whole-volume throughput remains
  unmeasured in the test workspace.

## Spec update

Update `planning/specs.md` to specify one enlarged circular mmap ring, at most
one frozen asynchronous flush interval, direct mmap chunk reads, completion-
gated reuse/progress, and the prohibition on band-sized RAM snapshots.

## Docs updates

Update `docs/code_structure.md` to describe the shared asynchronous flush
pipeline, its bounded backpressure, timing output, and failure semantics.

## Changelog and task log

Add a dated changelog entry. Record plan-review findings, implementation
details, validation commands/results, performance evidence available from
synthetic timing, and every deviation or limitation in `planning/task_log.md`.
