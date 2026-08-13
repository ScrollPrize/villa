# Task log

## Findings

- `replaceViewDemand()` previously removed old per-view priority slots but left
  their unresolved tasks in the schedulers as background work. Repeated renders
  could therefore grow the queue faster than it drained.
- Probe, source-read, and decode schedulers already have separate interactive
  and background lanes. The missing state was request ownership on each cache
  entry, not another scheduler split.
- The scheduler supported reprioritization and cache-wide epoch cancellation,
  but not targeted cancellation of one pending keyed task.
- A stale asynchronous GUI miss was previously accepted as background work when
  its view generation had already been superseded.
- Clearing by view ID alone could not reject a render allocated before closure
  but publishing afterward. VC3D now supplies its latest render request serial
  as the cleared generation watermark.

## Implementation notes

- Entries retain a background-demand flag and independent per-view generation
  slots. An unresolved entry is removable only when both are empty.
- Snapshot replacement and task cancellation happen under the shared scheduler
  selection gate, so workers cannot observe a partially published render.
- Pending work is canceled by task ID and removed from unresolved counters.
  Running work finishes its current stage; stale probe/download results are
  discarded before another stage is submitted, while an already-running decode
  may populate the cache.
- View closure removes that view from every registered source and retains a
  closed generation watermark. A strictly newer render generation reopens the
  stable view ID after a cache rebuild.

## Test cleanup

- The decode-priority test previously released a worker immediately after
  creating two requests. Persistent probes could race that release, so the test
  sometimes asserted before both decode tasks existed. Cache stats now expose
  the decode scheduler's pending count, and the test waits for both candidates
  before releasing one worker.

## Validation

- `test_chunk_cache`: 60 cases passed.
- Complete `test_chunk_cache` executable passed 20 consecutive runs.
- `VC3D` target built successfully.
- `git diff --check` passed.
