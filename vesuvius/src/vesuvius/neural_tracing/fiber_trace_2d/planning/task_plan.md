# Plan

1. Generalize the existing filter-stage planner so its final-stage selection
   is the complete global-lattice boxes intersecting the exact half-open target
   crop, not the lookahead box. Preserve repeat order and duplicate stages,
   normalize offsets modulo positive side, recursively select every preceding
   box intersecting later boxes expanded by endpoint reach, and expand source
   support once more to cover cross-boundary endpoints.
2. Extract transient stored-dataset stage orchestration into reusable core
   code. Build separate anchor/path overlay views over a combined source,
   process each stage monotonically, and expose the final read-only graph view
   while deleting temporary layers on teardown.
3. Add repeatable `--stage SIDE,OFFSET_X,OFFSET_Y,OFFSET_Z` plus filter join
   angle, stored-cost profile, and per-entry state-limit controls to crop
   tracing. Show a separate deterministic stage/box progress phase and stage
   reduction counts. Materialize the existing lookahead-padded graph from the
   final transient view; outside written overlays, missing upper chunks fall
   through to the original combined dataset.
4. Preserve sparse tuple semantics: a wholly absent source tuple is empty, a
   missing upper tuple inherits, an explicit empty upper tuple shadows, and a
   partial prefix/route tuple is corruption. Process overlapping boxes serially
   in canonical Z/Y/X order while retaining existing within-box parallelism.
5. Keep one shared bounded write-back LRU for every layer, drain it on exit,
   retain overlays through graph materialization, and remove all temporary
   layers on success or exception. Include ordered stages and filter policy in
   trace artifact provenance so incompatible outputs cannot be mixed.
6. Add planner and stored-overlay regression coverage, including the exact
   1024 crop and `256/0`, `256/128`, `512/256` schedule (27 final boxes),
   malformed CLI values, cross-boundary endpoints, monotone writes, tuple
   behavior, cleanup, and unchanged unstaged behavior.
7. Build the optimized targets, run focused tests, then run the requested crop
   trace using the existing input and report filtering/trace results.

## Spec Update

Specify crop-driven final-box selection, recursive dependency/source closure,
ordered monotone execution, exact sparse tuple behavior, combined-source
fall-through overlays, shared bounded lifecycle, filter controls/provenance,
and the relationship between target crop and separately padded search graph.

## Docs Update

Document the crop tracer `--stage` option and provide the concrete aligned
1024-crop command.

## Changelog

Record transient staged Fiberlet filtering for crop tracing.

## Validation

Run focused Fiberlet storage/graph tests, CLI parsing smoke coverage, an
optimized build, and staged/unstaged versions of the same requested 1024 crop.
Compare stage populations, trace geometry/counts, wall/user/sys/RSS, and verify
all source payloads remain unchanged. Record that sparse absence cannot prove a
partial mirror is complete without an external inventory/build marker.
