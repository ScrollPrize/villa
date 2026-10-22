# Reusable tracing recovery

Reusable native tracing, cache and evaluation APIs. This documents code
integration, not a new quality result or a prescribed experiment configuration.

## Inventory and scope

| Component | Integration |
|---|---|
| CPU beam search, candidate cutoff, width scoring, FW/BW verifier | Already on main; retained |
| Shared case queue, live readiness, pinned regions, reference selection/replay | Main APIs retained; not replaced by historical monolithic runners |
| Chunk seeds and tracing, routes and immutable cache | Main implementation retained and extended |
| Five-point initial matching, anisotropic initial ellipse, joint overlap/distance/direction/width joins | Recovered as optional chunk settings |
| Seed position/direction/width refinement | Recovered as optional native construction setting |
| Normal-oriented path conflict filtering | Recovered into `FiberChunkFilter.cpp`, standalone and cache-load API |
| Resident CUDA beam tracing and online fixed-plane refinement | Optional CUDA plugin; shared portable tensor/scoring primitives |
| RAM-only top2 spatial-companion 2x prediction pooling | Native read-only Zarr view, explicit source channel mapping |
| Generic evaluation | Existing callable `vc.fiber_evaluation` extended; native `replay-set` accepts `chunk-beam` |
| Construction/filter microbenchmarks and verified legacy cache import | `vc.fiber_chunk_benchmark` |

Historical parameter sweeps, model/checkpoint paths, cohort configurations,
results, plots, exported data and machine-specific launchers are **not** imported
or committed here. Keep them in external experiment archives. Experimental
skeleton/BP algorithms and training changes are outside this tracing API;
the repository's separate implementations remain in place.

## Settings and units

All chunk distances, movement caps and filter ellipses are in **base voxels**.
TraceConfig step sizes remain in trace coordinates. New features default off.
`ChunkConfig.to_json()/from_json()` round-trip the settings; unknown keys fail.

- `initial_alignment_power`: 0 preserves nearest matching, 1 cosine, 2 cosine².
  Optional initial normal/tangential radii override only the initial path search.
- `joint_join_score` and `join_weights`: overlap, width, distance and direction.
- `refine_seed`, `seed_movement_base`: bounded location/direction/width fitting
  before CPU tracing. A successful fit initializes that seed's width mean.
- `conflict_normal_base`, `conflict_tangent_base`: both positive enable filtering.
  Shared and separated run lengths are configurable (`conflict_agreement_base`,
  `conflict_separation_base`); sampling 8, angle gate 25°, endpoint guard 16.
  Conflicting paths keep their IDs but are emptied in the filtered view.
- `cache_only`: reject absent construction chunks; never silently retrace them.
- `presence_peak_restarts`: select the historical aligned local presence peak
  before matching a cached path, including explicit no-initial-path stops.
  Default false retains the generic cached-path seed search. For direct native
  `replay-set`, the corresponding top-level option is `prediction_based_seeding`.
  Peak selection rejects invalid/nonpositive width when width modulation is
  enabled, rather than selecting a seed that the first trace step cannot use.
- `gpu_library`, `gpu_devices`: explicit plugin and granted visible device count.
- `online_refine_iterations`: 0 disables; positive refines after each commit.
  Window 0 includes the whole committed path, positive values limit its tail.
  Each point's transverse plane stays fixed from its commit. Epsilon, step and
  displacement cap are explicit. This is experimental, not a default promotion.

Construction cache identity includes construction/trace settings and source
identity; matching/filter changes do not require constructing paths again.
The builder saves **unfiltered** paths. Filtering never overwrites these files.
Original main cache specifications are upgraded in memory with disabled defaults
for the new options; mismatches still fail. No historical cache is rewritten.

## Evaluation and cached import

Use the existing [callable evaluation](fiber_evaluation.md) config with
`chunks` settings. It retains main's shared queue and live-inference boundary:

```sh
python -m vc.fiber_evaluation evaluation.json results.json
```

For the native CLI, `vc_fiber_trace_metric replay-set --config evaluation.json`
accepts `backend: "chunk-beam"`, `chunk_trace` settings, and per-source
`chunk_cache` and `prediction_identity`. Beam remains supported unchanged.
No GPU is needed to replay existing paths against cached prediction chunks.

Forward replay retains candidate presence filtering inside beam selection;
committed-step checks alone cannot prevent infeasible speculative candidates
from displacing feasible paths. Reverse consistency checks retain width and
direction scoring but do not inherit the forward presence cutoff: they measure
geometric reversibility.

The optional `benchmark_fiber_chunk_cuda --device-tests` runs standalone CUDA
selector and two-stage tracing regressions without a dataset. Run it with
`compute-sanitizer --tool memcheck --error-exitcode 9` inside a granted GPU job.
It must be built with optimization as well as tested with real cold-cache
inputs; host-only scoring tests cannot validate device memory safety.

The CUDA winner reduction synchronizes after all warps snapshot its shared
winner, before the next reduction reuses scratch. Its scoring helper takes the
immutable parameter block by reference: the previous second by-value copy
caused invalid local-memory reads in optimized CUDA 12.8/H100 execution at the
second expansion. Keep the device regression in addition to host math tests.

Old experimental chunk files without main's specification need **explicit
provenance attestation**, not an unchecked compatibility switch:

```sh
python -m vc.fiber_chunk_benchmark import-legacy provenance.json NEW_DIRECTORY
```

The version-1 manifest contains `specification` from
`ft.chunk_cache_specification(chunks, trace, scale, source_identity)` and `files`
entries containing `path` and `sha256`. The source identity must identify the
immutable predictions, normals and transforms. The caller must verify the
attested construction settings against the original run. Import checks hashes,
native geometry and bounds, rejects duplicate keys and existing destinations,
and writes new copies. It cannot establish missing provenance by itself.

For historical native replay documents, keep using
`vesuvius.neural_tracing.fiber_trace_3d.replay_metrics` for minimum-piece pruning,
retained percentage, mean/median and restart-episode accounting. Do not interpret
all `failures` as GT-distance errors: report `distance_above_threshold`,
`chunk_no_initial_path`, `chunk_no_overlap`, limits and other reasons separately.
Filtered stops remain separate. The callable evaluator exposes raw counts and
geometry, not an implicit cohort-specific score.

## Microbenchmarks

Time native construction with caller-owned dataset handles, no path-cache IO:

```python
from vc.fiber_chunk_benchmark import time_construction
report = time_construction(field, normals, trace, chunks, [0, 0, 0],
                           workers=(1, 4, 16, 64), repeats=3)
```

An untimed warmup fills dataset caches. Timed runs rebuild paths and include
queue setup, pinning, seed processing and tracing. Worker-count parity checks
exact paths and stop reasons. Returned stage timings are backend-specific;
do not infer pure tracer time by subtracting unrelated runs.

Time only conflict filtering of already loaded path geometry:

```sh
python -m vc.fiber_chunk_benchmark time-filter \
  --normal-manifest normals.lasagna.json --config filter.json \
  --trace-to-base 1 --workers 1 4 16 64 --repeats 3 \
  --output filter-timing.json chunk-path-cache/0_0_0.json
```

Reports min/median/max/mean, exact rejected-ID parity, and throughput. Normal
data are warmed but not pinned: any subsequent normal-cache miss is included.
Neither helper starts inference or acquires resources on the caller's behalf.

## CUDA build and limitations

CPU builds do not require CUDA. With existing dependencies and a compatible
CUDA host compiler, configure your normal build with:

```sh
cmake -S volume-cartographer -B <build> -DVC_BUILD_FIBER_CHUNK_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build <build> --target vc_fiber_chunk_cuda benchmark_fiber_chunk_cuda -j 8
<build>/bin/benchmark_fiber_chunk_cuda --host-refinement-tests
```

Supply the resulting shared library via `gpu_library`. GPU execution requires a
manager grant owned by the caller; it does not obtain a grant itself. Persistent
device workers are keyed by library and visible device count. Admission covers
**pinning, host preparation and execution**, with 128 admitted chunks, not just
device launches. Wait callbacks cooperate with main's shared queue.

The recovered kernel supports beam8/lookahead2/parent-cap32, 81 angular
candidates, an even step budget up to 160, one prediction option, unfrozen width
and modulated cutoff. Unsupported settings fail explicitly. CPU seed fitting
and embeddings are not implemented in this GPU backend. Online refinement is
GPU-only. Defaults and unsupported settings must not silently fall back.
It uses float arithmetic with FMA disabled, not a CPU/GPU bit-identity promise.

The native C++ `benchmarkExport` option writes an explicit single-chunk fixture
for the standalone CUDA benchmark. Use a distinct directory per chunk; it is
not a shared export destination for concurrent builds. GPU execution is separate
from the host-only refinement tests.

## Native pooling views

`openLasagnaChannelArray` recognizes `.native_pool.json` beside an output
`.zarray`. Recipe revision 1, factor 2, mode `top2_spatial_comp` uses the original
spatial-companion weighting, float widths and axial tensor direction/normal
averaging. `source`, `source_level` and/or a `channels` mapping are explicit;
channel paths may be relative to source. Compatibility default level is 1.
Seven required channels: presence, nx, ny, width, sheet_nx, sheet_ny,
sheet_grad_mag. Output width metadata must be `<f4`; other channels `|u1`.
Output is raw, C-order, slash-separated chunks with half the source chunk size
and ceiling-half shape. Source chunks must be even-sized and aligned.

Views share a bounded 8 GiB RAM bundle cache and coalesce concurrent requests.
No pooled chunk files or scratch files are written. Missing source data raises
an error: readiness/inference belongs to the existing producer boundary.

Eigen is portable. On supported Linux builds with LAPACK, the recovered rare
degenerate/hemisphere-edge fallback uses DSYEVD. Without it, Eigen handles all
cases. **Byte-identical pooling across these solver policies is not promised**;
record the build/solver in prediction provenance when exporting or constructing
path caches. Do not reuse one source identity for different numerical policies.
macOS/arm64 runtime validation remains outstanding.
