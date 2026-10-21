# Fiberlet graph replay

`vc/fiber_tracer/FiberGraph.hpp` exposes reference-guided evaluation of a
precomputed fiberlet graph. It is a CPU library API, not a full-volume tracer
or a model-inference launcher. The reference is used for acquisition/restarts
and measuring departures; route selection uses the graph's stored costs.

```cpp
using namespace vc::fiber_tracer;
auto graph = buildFiberletGraph(pathReport);
FiberletGraphReplayConfig config;
config.expansionThreads = 4;
auto result = traceFiberletGraphReplay(
    graph, referenceBaseXYZ, normals, normalWorkingToBaseScale, config);
auto report = fiberletGraphReplayJson(result, config);
auto geometry = fiberletGraphReplayObj(result);
```

## Coordinates and costs

References, returned geometry, lookahead/step distances and acceptance radii
use base-volume voxels. Stored edge lengths/cost densities use prediction
voxels; `predictionToBaseScale` supplies the conversion. Normal sampling uses
the explicit `normalWorkingToBaseScale`. Do not substitute millimeters for
either coordinate system.

Replay uses persistent beam lookahead, strict graph transitions and shared
normal/tangential error measurements. Aggregate mode consumes stored edge and
join costs. Stepped mode consumes the stored per-segment cost profile; it does
not reconstruct a different objective from route coordinates. A route-state
limit raises `FiberletRouteStateLimitError`, rather than silently truncating
search. Diagnostics are optional and can be restricted to reference intervals.

Failures and restarts are explicit. Partial terminal edges, uncovered tails,
reference intervals, forced seeds and optional excursion grace retain their
existing semantics. OBJ output keeps independent trace segments separate.
`fiberletGraphReplayFailureWindows` returns focused windows for inspecting
failures, without joining across restart gaps.

## Ownership and parallelism

For repeated replay, construct `FiberletImmutableReplayGraphSource` once and
use the source overload. It owns immutable graph data and exposes pinned views
of routes and costs. A view's lease keeps its storage alive; callers must not
retain raw spans after releasing that lease. Custom graph sources default to
non-concurrent query support and must explicitly opt in for parallel queries.

This API is synchronous. Progress/failure callbacks execute within the replay
call; it does not start processes, reserve GPUs or reload experiment configs.
This first extraction does not add the on-demand storage adapter, crop seeding,
graph visualization, or the experimental layout/consensus algorithms. Those
can use the same source interface without copying the search implementation.
Existing format-3 fiberlet storage and publication remain unchanged.

## Tests

Build `test_fiberlet_paths` in an existing configured Release build, then run:

```sh
ctest --test-dir build --output-on-failure -R '^test_fiberlet_paths$'
```

Tests cover lookahead, exact stored cost integration, persistent-prefix
accounting, deterministic parallel expansion, bounded search, diagnostics,
elliptical acceptance, restarts, partial edges and failure-window validation.
For the optional real-data regression, set `VC_TEST_GRAPH_ANCHORS` to an
existing `anchors.json`, `VC_TEST_GRAPH_PREDICTIONS` to its prediction manifest,
and `VC_TEST_GRAPH_NORMALS` to the corresponding normal manifest. It generates
routes from the real predictions and compares serialized serial/four-worker
replay results. Its reference is a generated edge: this checks pipeline and
parallel equivalence, not tracing accuracy against annotated ground truth.
