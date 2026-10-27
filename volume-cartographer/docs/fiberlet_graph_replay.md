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
The storage adapter below connects the same search to stored and generated
chunks. Crop seeding/coverage is described below; graph visualization and
experimental layout/consensus algorithms remain separate. Format-3 encoding
is unchanged.

## Stored and on-demand graph sources

`vc/fiber_tracer/FiberletChunkGraph.hpp` provides the storage boundary:

```cpp
auto dataset = FiberletChunkDataset::openExisting(fiberletDirectory);
FiberletStoredReplayGraphSource source(dataset);
auto replay = traceFiberletGraphReplay(
    source, referenceBaseXYZ, normals, normalWorkingToBaseScale, config);
```

The single-dataset constructor accepts complete format-3 published data and
existing combined construction caches. It recovers route reconstruction and
join-cost parameters from authoritative processing metadata, not defaults.
Incomplete published datasets are rejected on open. Sparse absent chunks are
empty; corrupt chunks and missing route records are errors.

Published prefix-cache entries retain the owner chunk's embedded endpoint
table. Reconstructing a route does not fetch a neighboring anchor owner merely
to recover its endpoint. Referenced endpoints do not become duplicate seeds.
Endpoint memory counts toward the prefix cache's byte budget; leases preserve
payload lifetime after cache eviction.

For a `FiberletOnDemandPreprocessor`, reuse its existing datasets and caches:

```cpp
FiberletStoredReplayGraphSource source(
    producer->anchorDataset(), producer->anchorCache(),
    producer->fiberletDataset(), producer->fiberletCache(),
    pathConfig, 45.0F,
    [producer](const auto& key, auto payload) {
        return producer->evaluationAnchorChunk(key, std::move(payload));
    });
```

Use the producer's actual path configuration. The evaluation anchor view is
required when canonical float anchors feed compact path generation; it reuses
the producer's quantization/scoring rather than inventing a second transform.
The caller owns producer shutdown. Read-only stored sources do not generate
data or mutate the dataset.

`FiberletChunkGraphSource` exposes pending/ready/error queries for incident
edges, anchors, routes and transitions. Its complete reach-cube lookup accounts
for lexicographic endpoint ownership across chunk boundaries. Optional wait
executors must synchronously complete the supplied wait and be configured
before queries start. Stored replay is blocking and conservatively reports
non-concurrent query support; do not enable parallel expansion for this adapter.
Use immutable materialization below for parallel crop tracing.

## Crop tracing without a reference line

`vc/fiber_tracer/FiberletCropTrace.hpp` provides callable graph and dense
crop backends. Both start from canonical anchors, trace in both directions,
clip at the first crop exit and suppress covered seeds using normal/tangential
distance and sign-agnostic direction agreement. They do not use a GT line.

```cpp
FiberletCropTraceConfig crop;
crop.minimumBaseXYZ = minimumBaseXYZ;
crop.maximumBaseXYZ = maximumBaseXYZ;
crop.parallelThreads = 4;
const auto search = fiberletCropTraceSearchBox(crop);
auto materialized = source.materializeBaseBoxForSeeds(
    search.minimumBaseXYZ, search.maximumBaseXYZ,
    crop.minimumBaseXYZ, crop.maximumBaseXYZ, 4);
auto traced = traceFiberletCrop(
    *materialized.graph, materialized.insideAnchors, normals,
    normalWorkingToBaseScale, crop);
```

The graph box includes lookahead outside the output crop. Materialization
retains incident edges and their exterior endpoints, but does not allow joins
at exterior endpoints. It preserves degree-zero seeds and stored geometry,
cost profiles and join costs. Loading reuses the existing chunk adapter;
only join construction is parallel. Irrelevant halo edges are filtered before
resolving their endpoints. Required missing/corrupt data still fails explicitly.
The returned immutable graph owns its data independently of chunk-cache leases.

Seeds are ordered by decreasing presence, then stable storage key. A bounded
window computes candidates concurrently and commits in that same order.
Coverage, output geometry and cost are deterministic across worker counts;
speculative-work counts and timing are not. Custom sources/samplers that do
not support concurrent access automatically run serially. Progress callbacks
run on the calling thread. Limits on attempts, accepted lines and graph search
states are explicit configuration fields.

The dense backend uses the existing production beam tracer:

```cpp
crop.reverseConsistency = true;
auto traced = traceDenseBeamCrop(
    predictions, anchors, predictionToBaseScale, normals, denseConfig, crop);
```

Here normal sampling uses `denseConfig.traceToBaseScale`; prediction-grid,
trace and base coordinates are distinct. Forward/backward checking reuses the
shared consistency implementation, optionally checks prefixes, and returns
independent retained pieces with original-seed provenance. Results report
reverse attempts/failures and skipped base-voxel length. Optional accepted-trace
stopping uses indexed segments and revalidates speculative candidates at commit.

Backend restrictions are explicit, not silently ignored:

| Option | Fiberlet graph | Dense beam |
|---|---|---|
| Bidirectional tracing / seed coverage | Yes | Yes |
| Forward/backward consistency / prefix checks | Rejected | Yes |
| Stop at previously covered anchors | Yes | Rejected |
| Stop at accepted trace segments | Rejected | Yes |
| Graph ambiguity / stored-cost-density threshold | Yes | Rejected |

`importFiberCropLines` clips external polylines without connecting reentry
gaps. Imported zero costs mean “unavailable”, not perfect quality.
These APIs do not reserve GPUs, start inference processes or write artifacts;
the caller owns source readiness, output publication and visualization.

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

Also build/run `test_fiberlet_storage` for cross-chunk leases, on-demand reuse,
published endpoint independence and metadata recovery. Its optional
`VC_TEST_STORED_GRAPH` fixture is a read-only cache directory containing
`anchors.zarr` and `fiberlets.zarr` with Paris4 chunk `12/44/20`; the test
checks reconstructed geometry and exact reversal without generation. With
`VC_TEST_GRAPH_NORMALS` and `VC_TEST_GRAPH_PREDICTIONS`, it also compares
serial/four-worker graph crop and dense consistency-filtered crop results.
These are mechanical equivalence checks, not annotated accuracy benchmarks.

Build `test_fiberlet_crop_trace` as well, then run the three suites:

```sh
ctest --test-dir build --output-on-failure -R '^test_fiberlet_(paths|storage|crop_trace)$'
```

Crop regressions cover bidirectional starts, crop exits, deterministic parallel
coverage, ordered failures, search-state limits, ambiguity, quality rejection,
dense accepted-trace stopping, scale conversion, reverse gaps and imported runs.
