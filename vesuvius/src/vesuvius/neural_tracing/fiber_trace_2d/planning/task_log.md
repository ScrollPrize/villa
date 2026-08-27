# Task Log: Fiberlet crop lookahead graph halo

## Findings

- The crop command currently materializes only the requested crop's incident
  graph. A seed near a crop face can therefore rank lookahead routes against
  graph exhaustion caused solely by the requested output boundary.
- Trace first-exit clipping is already separate and must remain tied to the
  requested crop.
- One lookahead-distance halo is sufficient because bulk materialization
  retains the complete final Fiberlet incident to every in-search-box anchor,
  including its outside endpoint.
- Staged Fiberlet filtering is recorded in `planning/todo.md` as a separate
  follow-up and is intentionally not part of this implementation.

## Plan Review

- Independent review found that graph expansion alone was insufficient:
  speculative lookahead also clipped at the requested crop. The plan now gives
  rollout clipping the expanded search box while retaining requested-crop
  clipping for committed geometry.
- Review confirmed exact lookahead padding is sufficient because a complete
  final incident Fiberlet is retained even when its endpoint lies beyond the
  search box. It requested explicit long-final-edge and boundary-choice tests.
- Review also required distinct graph/seed use throughout materialization and
  complete sparse halo coverage. These are now explicit in the plan.

## Deviations

- None.

## Implementation

- Added one validated exact-lookahead search box shared by crop tracing and the
  command's graph preparation.
- Split bulk graph and seed bounds: the expanded box owns immutable graph
  content while only requested-crop anchors become seed candidates.
- Speculative rollout now clips at the expanded search boundary. Committed
  geometry still clips at the requested crop boundary.
- Added graph-preparation diagnostics for requested bounds, search bounds, and
  padding.
- Added the staged-filtering experiment to `planning/todo.md`; it is not active
  in crop tracing.

## Validation

- Built `vc_fiber_trace_chunk`, `vc_fiberlets`, `test_fiberlet_storage`, and
  `test_fiberlet_crop_trace` from `volume-cartographer/build` with `-j32`.
- `volume-cartographer/build/bin/test_fiberlet_storage`: 40 cases passed.
- `volume-cartographer/build/bin/test_fiberlet_crop_trace`: 74 cases passed.
- The new route-choice test proves evidence beyond the requested crop changes
  the selected continuation while emitted geometry remains clipped to the
  requested crop. The storage suite covers separate graph/seed boxes and
  required partial sparse tuples in the expanded halo.
- A Release smoke trace on the Paris4 combined Fiberlet dataset used a
  128-base-voxel crop, the default 384-base-voxel lookahead, and one attempt.
  It reported crop `[10240,22016,6144)..[10368,22144,6272)` and expanded search
  `[9856,21632,5760)..[10752,22528,6656)`, then accepted and published one
  bidirectional crop-clipped trace. Graph preparation took 9.44 s and tracing
  took 0.063 s; the temporary artifact was removed.
