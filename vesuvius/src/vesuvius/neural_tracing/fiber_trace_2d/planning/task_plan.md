# Plan: finish staged Fiberlet reduction performance and reporting

## Baseline and invariants

1. Build the current tree in the ordinary `volume-cartographer/build/`
   directory as Release and run the established hot Paris4 two-stage workload.
   Record repeated wall/user/system time, actual effective cores, stage phase
   timings, retained-ID digests, payload digests, and all output counts.
2. Treat the existing canonical ordering, floating-point operations, retained
   ID sets, and serialized overlay bytes as invariants. No numerical or
   acceptance change is allowed.

## Chunk-granular graph materialization

1. Replace per-anchor `incidentEdges`, per-direction `directedEdge`, and
   per-transition cache queries in `materializeChunkRouteGraph` with a shared
   bulk path:
   - load each seed anchor chunk once;
   - union the exact per-inside-anchor owner-reach cubes, load each resulting
     prefix-owner chunk once, and select prefixes incident to the transformed
     canonical inside-anchor-key set;
   - load each required endpoint-anchor chunk once through the same anchor view;
   - construct the same sorted physical Fiberlet and directed-arc arrays from
     those immutable payloads.
2. Build transitions from already materialized arcs and shared anchors, using
   the existing `storedTransition` implementation. Parallelize this cache-free
   per-incoming-arc work with indexed result slots and merge counts in canonical
   order, including deterministic lowest-index exception selection.
3. Keep one shared materialized graph for analysis and simplification. Do not
   duplicate scoring or transition logic in the CLI.

## Reporting without redundant full graphs

1. Extend stage populations to retain both the canonical union of all incident
   Fiberlet IDs and the canonical union of Fiberlets interior to at least one
   complete stage box. This retains the existing per-box interior definition;
   a Fiberlet crossing between adjacent boxes is `all`, not `interior`.
2. Keep stage domains equal to the union of that stage's complete boxes. Report
   `anchors`, `all`, and `interior` for original/input/output and both stage and
   cumulative reductions.
3. Snapshot the original and inherited input populations before any box in the
   stage writes, then snapshot output after all writes. All three use the bulk
   no-transition path; sequential box analysis cannot substitute for the input
   snapshot because later boxes observe earlier writes. Reuse the final
   selected-region population instead of collecting it twice.

## Optimized default build

1. Confirm `volume-cartographer/build/CMakeCache.txt` uses `Release` and
   `-O3 -DNDEBUG`; explicitly reconfigure that build directory as Release if
   needed.
2. Build and document `volume-cartographer/build/bin/vc_fiberlets` as the
   canonical performance binary. Debug CI directories remain test-only and are
   not silently changed.

## Tests and validation

1. Add a regression comparing bulk results at one and multiple threads against
   the existing public point-query behavior on a multi-owner fixture with a
   non-identity anchor view. Cover both edge-cost views, anchors,
   all/interior Fiberlet IDs, arc cost fields, successor order and join fields,
   entries/exits, route distributions, retained sets, and simplification
   structures; timing fields are excluded.
2. Add stage-domain reporting coverage for an offset stage with incident
   Fiberlets crossing its boundary and between adjacent stage boxes, proving
   `all` differs from per-box `interior`, IDs are counted once, and untouched
   geometry is excluded.
3. Build `vc_fiberlets`, `test_fiberlet_storage`, and
   `test_fiberlet_paths` with 32 jobs. Run the focused tests in Debug and the
   storage suite in Release.
4. Run the exact hot Paris4 command at least three times after warm-up. Report
   min/median/max and actual CPU use. Verify retained-ID and payload digests are
   unchanged from the recorded baseline.
5. Capture a same-input Release hotspot profile with an available local
   profiler. If sampled `perf` remains unavailable, use an instrumented
   profiler build and record that limitation explicitly. Keep hashing and
   reporting timings separate from graph materialization, search, and writes.

## Spec update

Specify chunk-granular local graph materialization, cache-free parallel
transition construction, complete stage-local `anchors`/`all`/`interior`
reporting, and reuse of already materialized populations.

## Docs updates

Document the three stage scopes, the optimized default build command, the
chunk-granular implementation, and exact before/after performance results in
`volume-cartographer/docs/fiberlets.md`.

## Changelog update

Record the completed staged-reduction speedup and restoration of stage-local
all-incident Fiberlet statistics.
