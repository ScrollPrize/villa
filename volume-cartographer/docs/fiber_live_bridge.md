# Live prediction publication bridge

The native tracer can wait for an inference coordinator to publish prediction
chunks rather than interpret not-yet-written chunks as empty. This extracts the
control transport used by live fiber evaluation. Prediction data still travels
through the ordinary Zarr/native sampling path, not through the socket.

This is a client library, not an inference service or a GPU scheduler. The
coordinator must implement the protocol below and must acknowledge a dependency
only after the entire coherent prediction product is readable. It must preserve
acknowledged chunks for the lifetime of the session. Do not reuse a bridge or its
publication proofs across inference sessions or changed models.

## Python

```python
from vc import fiber_trace as ft

bridge = ft.LiveBridge(port, token, timeout_seconds=600, heartbeat_wait=True)
field = ft.open_prediction_field(prediction_manifest, enable_width=True)
field.set_live_bridge(bridge, source="paris4", case_index=0)
# Use the same field in trace_segment, trace_extrapolation, or ChunkCache.
# Configure the field before starting any concurrent sampling.
# To interrupt blocked readers:
bridge.cancel("evaluation stopped")
```

The field retains the bridge even if its Python handle is discarded. Native
requests release the GIL; the reader thread never invokes Python. Multiple
native callers may wait concurrently; replies may arrive out of order.
Cancellation, disconnects, invalid responses and timeouts fail all outstanding
waiters. There is no silent reconnect or reuse of stale publication proofs.
An empty cancellation reason uses a nonempty default error.

The optional `remaining_base_voxels` value on `set_live_bridge` is a fixed
coordinator scheduling hint, not a dynamically updated route distance.
`proof_cache_capacity=4096` bounds the field-local proof cache. The bridge has
a shared 100,000-key proof cache. Eviction only requests another acknowledgement;
it does not invalidate prediction payloads or imply that inference must repeat.
For direct requests, `bridge.wait(source, case_index, remaining_base_voxels,
array_paths, indices_zyx)` uses one array path and one ZYX index per dependency.

## C++

Include `vc/fiber_tracer/FiberLiveBridge.hpp` and link `vc_fiber_tracer`.
`FiberLiveBridge::await` accepts the dependencies already produced by
`FiberPredictionField::setReadinessGate`. `FiberCaseReadiness` supplies the
optional bounded case-local proof cache. Its activity check runs even on cache
hits; failed waits never establish proof. Concurrent callers must leave its
optional profiling pointer null. Profiling values are counts and seconds.

Construct before callers start and destroy only after they join. The bridge
blocks the calling OS thread while waiting; coroutine case scheduling and CPU
permit handoff are deliberately separate. No additional compute worker pool is
created here. This transport supports Ubuntu/macOS sockets; other platforms
compile but report an explicit unsupported-platform error on construction.

## Wire protocol (version 1, preserved from live evaluation)

TCP to IPv4 loopback only. Each UTF-8 JSON object ends with a newline.
The session starts with:

```json
{"version":1,"type":"hello","token":"session-secret"}
```

A request contains a session-unique integer request ID:

```json
{"version":1,"type":"demand","request_id":1,"source":"paris4","case":0,"remaining_base_voxels":256,"chunks":[{"array":"/prediction/presence/0","index_zyx":[0,1,2]}]}
```

The coordinator replies only after publication:

```json
{"version":1,"type":"ready","request_id":1}
```

It can fail the session with `{"version":1,"type":"error","message":"..."}`.
With `heartbeat_wait=true`, `{"version":1,"type":"heartbeat"}` extends the
liveness deadline but **never** marks any chunk ready. Without heartbeat mode,
each request has a fixed timeout. In heartbeat mode a healthy coordinator may
wait indefinitely for resources. Cancellation remains available at any time.
Messages larger than 1 MiB, unknown IDs/types and unsupported versions are
errors. A token is a shared session secret, not encryption; do not expose this
transport beyond loopback.

## Validation

From the checkout root, with an existing configured build:

```sh
cmake --build volume-cartographer/build/pr-native --target vc_fiber_trace test_fiber_live_readiness -j 8
ctest --test-dir volume-cartographer/build/pr-native -R test_fiber_live_readiness --output-on-failure
python -m pytest -q volume-cartographer/python/tests/test_fiber_live_bridge.py --vc-build-python volume-cartographer/build/pr-native/python
```

The tests use a loopback coordinator to cover concurrent out-of-order replies,
session proof caches, errors, cancellation, timeouts, heartbeats and Python
ownership. Native tests cover cache eviction, failed waits and shared readers.
The optional cached Paris4 test uses the three fixture environment variables
documented in [chunk tracing](fiber_chunk_tracing.md). It runs the real 256-base-
voxel replay through the publication bridge and verifies the same single
segment, zero failures and exact disk reload. It does not run a GPU model or
claim end-to-end live inference throughput.

Extraction validation (GCC 13 Release, Python 3.14): 14 native regression
suites passed; the bridge, chunk and existing tracer Python suites passed all
17 tests, including the cached Paris4 replay. No GPU was used. macOS and
unsupported-platform compilation have not been exercised locally.
