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
a shared 100,000-key proof cache keyed by source, array path and chunk index.
Identical array paths in different sources require independent acknowledgements.
Eviction only requests another acknowledgement;
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

Construct before callers start and destroy only after they join. Ordinary
callers block their OS thread while waiting; inside a `FiberCaseQueue`, the
same bridge yields the case stack so another case can use that worker.
The bridge itself creates no compute worker pool. This transport supports
Ubuntu/macOS sockets; other platforms
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
cmake --build volume-cartographer/build/pr-native --target vc_fiber_trace test_fiber_live_readiness test_fiber_case_queue -j 8
ctest --test-dir volume-cartographer/build/pr-native -R 'test_fiber_(live_readiness|case_queue)' --output-on-failure
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

## Bounded cases and CPU permits

`FiberCaseQueue(workers, cases, work, contextsPerWorker=8)` runs a fixed number
of root callbacks. Each admitted case has an 8 MiB virtual stack with guard
pages and stays on one OS thread. Only the configured number of stacks exists;
the remaining cases are lightweight counters. The constructor starts workers;
`join()` waits and reports the first failure. `cancel(reason)` and destruction
cooperatively unwind waiting cases. Cancellation cannot forcibly interrupt
arbitrary user I/O or a CPU loop that never yields.

`await(nonblockingPredicate)` yields until the predicate succeeds or the queue
is cancelled. Call `notify()` after external readiness changes; a 10 ms poll
also covers predicates without notifications. Predicates run on the scheduler
stack: they must not block, yield, or invoke Python. Do not yield from a C++
exception handler or while holding a mutex needed by another case.

`parallelFor(count, indexedCallback)` uses the same workers. Parents help
their own children from their existing stacks, even when every context is
occupied, and join all children before returning or unwinding. Idle workers
remain available for children submitted later. Failures cancel sibling waits.
Callbacks must not depend on arbitrary not-yet-admitted root cases: bounded
admission cannot guarantee progress for such a dependency graph.

`FiberCaseCpuPermit` wraps a counting semaphore. Its `suspend(wait)` releases
the permit during a dependency wait and reacquires it on success; an exception
unwinds without a permit. `FiberConstructionCpuPermit::current()` is scoped
to the case, not merely thread-local, so another suspended case cannot borrow
the wrong permit. The Python field's bridge attachment uses that scope when
present. Child work can acquire its own permits as required by the caller.

The native chunk-cache wait and parallel callbacks accept these functions:

```cpp
cache.get(key, predictions, trace, normals, scale,
          FiberCaseQueue::parallelFor,
          [](const auto& ready) { FiberCaseQueue::await(ready); });
```

Python offers ordered multi-key construction, with the GIL released:

```python
trace_config.parallel_threads = 1  # no nested OpenMP worker pool
cache = ft.ChunkCache(field, normals, trace_config, chunk_config,
                      output_directory, source_identity)
chunks = cache.get_many(keys_xyz, workers=4, contexts_per_worker=8)
```

Concurrent duplicate keys still share one build. Publication and cache-future
waits yield their cases. No Python callback runs inside a stackful context.
Ordinary `get`, `route` and `replay` remain synchronous. Higher-level
evaluation orchestration and GPU/resource management are not part of this API.

Stackful scheduling uses Linux/macOS `ucontext`. Check
`FiberCaseQueue::supported()` or Python `case_queue_supported()`; unsupported
platforms compile but explicitly reject queue construction (ordinary tracing
and serial `parallelFor` remain usable). Context creation failures cancel the
queue. A failure of the OS context-switch operation itself is unrecoverable
and terminates the process rather than freeing stacks still referenced by work.

Regression tests cover bounded admission/thread affinity, fully occupied
nested pools, late child submission, exception/destructor unwinding, permits,
and a one-worker live build with concurrent duplicate-key waiting. Real cached
Paris4 replay also prebuilds duplicate chunks through the queue. No GPU is used.
macOS and unsupported-platform compilation have not been exercised locally.
