# Forward/backward consistency for native fiber tracing

`verifyFiberReverseSpan` checks whether a new backward trace follows an actual
forward prediction. It never compares against a ground-truth fiber, and does
not modify or automatically trim the forward path. Periodic scheduling,
reference replay, restart selection, visualization and Python bindings are
separate callers, not part of this API.

## Usage

```cpp
#include "vc/fiber_tracer/FiberReverseConsistency.hpp"

// forward.points use trace coordinates; geometry must use base coordinates.
std::vector<cv::Vec3d> basePoints;
for (const auto& p : forward.points)
    basePoints.push_back(p * config.traceToBaseScale);
auto geometry = vc::fiber_tracer::makePolylineArcGeometry(basePoints);
auto reverseConfig = config;
if (config.widthRelativeTolerance > 0)
    reverseConfig.widthReferenceMean = forward.widthMean;
double remaining = vc::fiber_tracer::verifyFiberReverseSpan(
    predictions, normals, reverseConfig, geometry, geometry.length(), 20, 40);
// remaining == 0: verified back to the beginning.
// Otherwise only the suffix [remaining, geometry.length()] was verified.
```

The prediction source and normal sampler use trace coordinates. Forward
geometry, return value and the two ellipse **radii** use base voxels. The normal
radius constrains displacement perpendicular to the sheet; the tangent radius
constrains displacement within the sheet. Invalid normals use the shared
metric's Euclidean fallback at the normal radius.

Each committed reverse point is projected onto the preceding forward interval
`[max(0, previousArc - 2 * reverseStepLength), previousArc]`. It must both advance
backward and remain within the anisotropic tolerance. Equality to the tolerance
passes. Verification stops at the first failure or upon reaching arc zero;
there is no excursion grace. Duplicate points do not advance verification.
The search budget is three times the requested reverse span, with a minimum
of two tracing steps; exhausting it does not imply success.

An optional final `historySteps` argument (0–64, default 0) initializes reverse
smoothness from preceding forward edges. When enabled, a start between vertices
snaps to the next forward vertex. Direction history does not add path points or
consume tracing steps. Width gating should use a fixed forward mean, as above.
Embedding scoring, if enabled, starts a fresh reverse history.

A missing prediction direction at the reverse seed is a failed check, returning
the starting arc. Invalid configuration and other sampling errors propagate.
The lower-level `ReverseSpanVerifier` can verify committed points from another
backend with the same matching/threshold rules; stop feeding it when `observe`
returns false. Construct its geometry with `makePolylineArcGeometry` and keep
both geometry and the normal sampler alive throughout verification.

## Committed-point callbacks

`traceFiberExtrapolation` optionally accepts a callback after its existing
`fixInitialDirection` argument. It receives a trace-coordinate point,
cumulative tracing loss, and one-based step index. Only shared, irrevocable
beam prefixes are reported during search. The selected path's remaining nodes
are reported on return; the seed is excluded and each node appears once.

Returning false stops at that node with reason `observer_stop`. The returned
path includes it, and width accumulation includes no later nodes. Callbacks see
native path nodes before optional final distance clipping, and can be delayed
while surviving beam branches disagree. This reuses committed-width tracking
rather than creating a second notion of path ownership.

## Validation

Using an already configured native build, from the checkout root:

```sh
cmake --build volume-cartographer/build/pr-native --target test_fiber_reverse_consistency test_fiber_trace3d test_fiber_width_scoring -j 8
ctest --test-dir volume-cartographer/build/pr-native --output-on-failure -R '^(test_fiber_reverse_consistency|test_fiber_trace3d|test_fiber_width_scoring)$'
```

Tests cover anisotropic acceptance, invalid-normal fallback, no backward
progress, inconsistent paths, committed callback ordering, stopping inside
lookahead, width accounting, exhausted frontiers, error propagation, scale
conversion and optional vertex history. Existing tracing defaults are unchanged.

The optional real-data check reads cached inference only:

```sh
export VC_TEST_WIDTH_MANIFEST=/path/to/width-enabled/inference.lasagna.json
export VC_TEST_WIDTH_FIBER=/path/to/covered/reviewed-fiber.json
export VC_TEST_REVERSE_NORMAL_MANIFEST=/path/to/normals.lasagna.json
volume-cartographer/build/pr-native/bin/test_fiber_reverse_consistency
```

Tested with GCC 13 Release on Paris4 reviewed piece
`0039_anon_20260815T025206849_000041_run0.json`, checkpoint
`a56e9dc2116af3db535d0c6283b009833bd788eb77d1056ab175c11edc46db10` and existing
Lasagna normals. The piece supplies only the starting point/direction; the
verification geometry is the predicted forward trace. A 64.0011-base-voxel
forward path verified fully backward (remaining arc 0), with identical serial
and four-thread results. No GPU inference was run. This is integration
evidence, not a claim of improved tracing accuracy or throughput.
macOS/arm64 have not been tested locally.
