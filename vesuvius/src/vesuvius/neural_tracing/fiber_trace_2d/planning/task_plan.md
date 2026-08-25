# Plan: lock-free parallel Fiberlet crop tracing

## Contract

- Preserve canonical seed order, exact selected routes, output order, geometry,
  limits, counters, and deterministic serial coverage semantics.
- Use the stored chunk infrastructure only during one bounded bulk preparation
  stage. After preparation, seed workers must query an immutable in-memory graph
  without cache locks, waits, I/O, or repeated route reconstruction.
- Keep only candidate tracing parallel. Coverage sampling, active-anchor
  mutation, result acceptance, and progress reporting remain serial.
- Discard a speculative candidate if an earlier accepted line covers its seed.
- Use the host CPU count by default; no separate low worker cap may hide graph
  access contention.

## Implementation

1. Extract the existing private eager replay adapter into a reusable public
   immutable replay source. Port its existing caller to the shared class rather
   than copying the implementation.
2. Extend the existing chunk-route bulk materialization path to construct the
   immutable replay graph for a base-coordinate crop:
   - load crop anchors once;
   - load every incident prefix owner chunk once;
   - load required route and endpoint-anchor chunks once;
   - reconstruct every selected physical route once;
   - build stable adjacency and allowed join transitions once.
3. Expose bulk materialization from `FiberletStoredReplayGraphSource`. Its
   direct cache-backed queries remain valid serial/on-demand APIs, but it must
   no longer advertise itself as a scalable concurrent trace source.
4. In `vc_fiber_trace_chunk`, materialize the crop graph before tracing and
   report preparation time separately. Pass the immutable graph to the crop
   tracer and use `--threads` for both preparation and trace workers.
5. Retain bounded parallel seed batches and serial canonical integration.
   Remove the separate `--trace-threads` workaround and the four-worker cap.

## Tests

- Preserve the synthetic exact serial/parallel equivalence and speculative
  covered-seed discard regressions.
- Add stored-dataset coverage proving bulk materialization preserves anchors,
  outgoing arcs, route geometry, cost profiles, and transitions relative to
  the cache-backed source.
- Instrument a synthetic immutable source to prove trace work overlaps and no
  mutation occurs during worker execution.
- Build the optimized Release CLI and focused tests.
- Benchmark the same Paris4 crop and 16-attempt workload with one and all host
  threads. Report bulk-preparation time, trace time, wall/CPU time, effective
  cores, peak memory, and exact OBJ equivalence.

## Spec update

Specify a bulk immutable crop graph boundary between chunk-backed storage and
parallel tracing. Clarify that thread-safe cache APIs are not used by crop seed
workers and that deterministic serial integration remains authoritative.

## Documentation updates

Document bulk graph preparation, separate preparation/trace timing, host-CPU
default parallelism, and the serial integration/discard behavior.

## Changelog

Replace the preliminary bounded-cache-query implementation note with the final
lock-free immutable-graph implementation and measured Release results.
