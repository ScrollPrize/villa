# Unified native tracer cache — 2026-09-11


### Initial run observation

Native PID246841; shared decoded budget216408444518bytes (201.54GiB).
At152.4s:772 inferred tiles and1345 published chunks. All8GPUs showed100%
utilization in a snapshot. This is startup/inference progress, not completed
trace timing. Native startup log: /tmp/fiber-live-control-4q52uv61/native.log
(copied into the output directory by the runner at termination).
## What changed

The native tracer's Lasagna scalar/normal samplers no longer have their own
decoded chunk cache. All these paths and the existing fast corner sampler now
use regular VC3D ChunkCacheService source handles, coalescing, decoded uint8
payloads, eviction and memory accounting.

The canonical source identity and bulk-consumer eviction preference match the
previous regular corner path. No new payload representation or copy is added.
The previous plane sampler's8-entry local reference helper was extracted to
LocalChunkCache.hpp; its new blocking mode is used by scalar cube resolution.
Cubes and batches retain shared references while sampling. The regular cache
protects and counts active pins; they can temporarily put it over budget until
readers release them.

The old LasagnaChannelChunkCache, singleton, LRU, in-flight/read pool, payload
copy and sampler cache-size parameters are removed. One native process budget
covers all fiber and normal sources. Python inference runs in separate processes
and retains its own input caches; this does not merge those across processes.

## Configuration and numerical behavior

Batch replay uses10% physical RAM by default, detected with Linux sysconf or
macOS sysctl. Explicit positive `decoded_cache_bytes` overrides it. The effective
value is printed at startup and persisted in the result config.
The obsolete `cache_bytes_per_channel` setting now throws instead of silently
pretending each source has an independent capacity.

Other standalone tools apply their existing cache-size option to the regular
service at startup. Sampler constructors never modify the GUI's service budget.
Scalar double interpolation, tensor normal decoding, uint8 values, requested
levels, live-readiness gates and tracer parameters were not changed.
Absent declared fill remains unavailable for scalar reads; explicit fill is
sampled without allocating a duplicate fill chunk. All-fill elision is disabled
for sources without declared fill so stored zeros remain valid data.

## Verification

Existing GCC13 RelWithDebInfo CMake build, no installs or numeric flag changes:

```bash
cmake --build /tmp/vc-live-native-cmake --target vc_fiber_trace_metric test_lasagna_normal_sampler test_fiber_trace3d test_chunked_plane_sampler test_fiber_anchors test_fiber_replay test_fiber_trace_review test_fiber_reference_replay_benchmark -j 8
cmake --build /tmp/vc-live-native-cmake --target vc_fiberlets vc_fiber_trace_chunk vc_lasagna_normal_align test_chunk_cache -j 8
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
```

315 native cases and69 Python tests passed (Python30.18s).
New coverage exercises both scalar/corner warm orders with shared payload
identity and no repeated codec work, concurrent cold readers, pin protection
and eviction after release, missing/no-fill versus stored zeros, malformed
payloads and invalid fill. Existing tests cover normal/scalar numerics,
live readiness, reference replay and rendered strips.

Independent plan and code review completed. `git diff --check` passed.
macOS runtime and the GUI executable were not tested in this headless Linux
build; regular viewer plane sampling is covered by the CPU suite.

## Authorized full rerun

Same approved entrypoint; only its workload config changes:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_unified_cache.json
```

Preparation/validation succeeded for the locked classic best91_5k checkpoint,
50 fibers,100 directed cases,6 collections over5 scans. PHerc1229's embedded
PHerc1299 volume mapping and both Paris4 collections are unchanged.
The manager granted exclusive8GPU/224logicalCPU ownership at priority100,
task232a816c-345a-4e17-b7ae-ac9260cdcc6c.
No direct training/process interventions or unmanaged fallback.

Output:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_s1_91500_20260911_unified_cache_v1`

Fresh inference namespace:
`classic_s1_91500_unified_cache_20260911_v1`

CT/normal fetch caches are reused, previous inference chunks are not.
Prior cancelled-run outputs are retained. Full timing/results are pending;
no speedup is claimed from the migration before measurement. Reference
matching's whole-reference traversal was not changed by this cache patch.
