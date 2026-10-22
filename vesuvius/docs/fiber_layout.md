# Fiber constraints, winding/UV layout and minimum flow

This extracts the existing AFV-to-constraint and fiber-layout implementations.
Inputs and earlier stage solutions are immutable; preparation records source
identities, hashes and coordinate conventions. Preparation, search, experiment
matrices, diagnostics and tifxyz surface export are documented in
[fiber-layout workflows](fiber_workflows.md). Run-specific configs and data are
external; portable usage samples are included.

## Native fixed evidence

`vc_fiber_constraints CONFIG.json` reads an AFV collection and normal inference,
checks scan identity/base shape against inference provenance, and exports local
parallel/perpendicular evidence. It retains continuous evidence; downstream
stages decide rejection/quantization. Optional normal orientation and axial
priors reuse the same shared normal field and geometry implementations.

Required extraction config entries are `afv`, `coordinate_space`,
`normal_manifest`, `normal_provenance`, `remote_cache`, `output`, and either
`full_volume: true` or `minimum_xyz`/`maximum_xyz` (base XYZ). Use `database` to
write a constraint database, with `repetitions: 1`. Existing outputs are not
overwritten. `threads` controls CPU extraction; this executable uses no GPU.
`load_threads` independently controls read-only AFV reader connections (default1).
Each reader pins a snapshot and consumes complete fibers; sorted-ID output and
length accumulation are unchanged. Measure reader fan-out before raising it:
SQLite/file-cache contention can make more readers slower, unlike compute workers.

Set `progress_interval_seconds` (positive seconds) to log structured phase
progress; `progress_file` optionally receives the same JSONL records. Each row
contains `stage_index`, `phase`, `completed`/`total`, elapsed time, units/second
and phase ETA. Unix builds also report process CPU seconds and peak RSS bytes.
The units are fibers for preparation/loading, samples for spatial queries,
initial candidates for `score_batches`, and accepted events for export. A phase
ETA is not an overall job ETA. `score_batches` includes encounter deduplication,
winding measurement and adaptive refinement; their final timings are also saved.

Spatial queries use read-only shared indexing and worker-local contiguous
candidate buffers. Canonical sample-pair ordering is restored before scoring.
Geometry/resampling is prepared independently per input trace, then assembled
in input order, with deterministic exception precedence. Local candidate keys
are unique: parallel chunk sorting/ordered merging preserves their exact total
ordering. Legacy epsilon reductions are not reordered. Whole-fiber finalization
validates existing owners directly rather than copying/remapping unchanged pieces;
actual subdivisions still use the common certified-event remapper.
`candidate_batch_size` defaults to 262144 initial candidates and is a scratch
ceiling target, not an observation limit. A persistent dynamically scheduled
worker pool consumes independent jobs targeting at most1024 initial candidates
(or the configured target if smaller). Complete fiber-pair groups stay together,
so a job can exceed that target; `maximum_batch_candidates` reports the maximum.
Workers run scoring, deduplication, measurement and refinement without a barrier
between unrelated jobs. Results and integer counters are assembled deterministically.
Orientation/winding substage seconds sum worker durations; extraction seconds
remain wall time and are the appropriate throughput measure.
All accepted events remain in the output. The legacy epsilon-based nearest-pair
reduction and arbitrary caller filters keep their serial invocation order.
Database row encoding is parallel and a bounded producer queue overlaps normal
sampling/encoding with the SQLite writer. Insertion preserves original event
order in one transaction; there is no per-row commit or filesystem sync.
Native whole-volume extraction with a database output now defaults to
`stream_export: true`: finalized whole-pair event batches feed export while
other pairs continue extracting. `export_queue_batches` defaults to128 completed
jobs, plus at most one priority batch for the next required ID; blocked worker
payloads are additionally bounded by the extraction worker count. A separate
two-batch encoding queue overlaps sampling/JSON preparation and insertion.
This bounds additional streaming buffers, not the retained extraction report.
Canonical ordering, encounter deduplication and evidence normalization are
preserved. Streaming accepts whole-fiber local events without extra cuts,
owner filtering or aligned-normal processing; incompatible modes fail explicitly.
The database is not published until extraction succeeds and indexing/validation
finish. Errors cancel blocked producers; a failed partial file is not published.
Use `stream_export: false` only for a sequential timing/control run. The same
export implementation handles both modes. Progress reports independent phase
timelines; the final event count is unknown until stream completion.
`tail_after_extraction_seconds` measures remaining export/index/check time after
extraction returns; streamed `export_seconds` includes overlapping source waits
and must not be added to extraction seconds. `processing_seconds` is end-to-end.
The unpublished partial database receives its full foreign-key audit and
integrity check before publication; redundant per-insert FK lookups are avoided.
Export settings preserve the existing format and defaults:
`sqlite_insert_batch_rows` defaults to32 (range1..32). Constraint rows use cached
multi-row prepared INSERTs, including an exact-size final partial batch. The
runtime SQLite variable limit can reduce the batch size. Row order, values,
schema and publication checks are unchanged;1 permits controlled timing comparisons.
`sqlite_page_size` is zero (SQLite default) or a power of two from512 to65536;
`sqlite_in_memory: true` builds/indexes/validates the entire database in RAM,
then saves it with SQLite's backup API. RAM mode needs enough memory for the
entire database plus extraction. Both modes count final persistence in
`export_seconds` and separately report `persist_seconds`. RAM temporary sorting
alone is not equivalent to building the entire database in RAM.
Bounded arc projection uses binary interval lookup in the shared geometry helper,
preserving closed interval endpoints and original segment order/calculations.
No sampling, radius, numerical precision or acceptance tolerances are changed.

`compare_database` verifies every extracted measurement and its order against
a saved database. It can be combined with `database` to validate and then export
a new result; `comparison_seconds` separates verification overhead from timing.
`wall_seconds` includes verification; `processing_seconds` excludes that optional
comparison. Export reports insertion, index/commit and validation times separately.

In an already configured VC build:

```sh
cmake --build volume-cartographer/build/your-build --target vc_fiber_constraints vc_fiber_bake_annotations
vc_fiber_constraints /data/fibers/constraints.json
```

The shared library includes normal-aligned winding integration, normal-sign
alignment, constraint geometry, binary/winding belief propagation, least-squares
and ordered-cut solvers. There is no new mandatory HiGHS dependency.

## Python stages

Under `vesuvius.neural_tracing.fiber_trace_3d`:

| Stage | Modules |
|---|---|
| Normal gauge and H/V | `fiber_normal_sign`, `fiber_normal_convention`, `fiber_hv_stage` |
| Direction and winding | `fiber_direction`, `fiber_label_layout`, `fiber_soft_winding`, `fiber_implicit_winding` |
| Physical U/V geometry | `fiber_uv_layout`, `fiber_shared_curve`, `fiber_intermediate_uv`, `fiber_bucket_uv` |
| Sparse/distributed execution | `fiber_geometry_cache`, `fiber_geometry_kernels`, `fiber_grouped_gather`, `fiber_pair_parallel`, `fiber_uv_distributed` |
| Read-only diagnostics | normal/sign, winding-history, density, U/V, reference-comparison and tiled-raster visualization modules |

These retain their existing configuration dictionaries and `prepare`/`solve`/
`run` APIs. The executable modules default to their existing managed launcher;
`--unmanaged` is for the authorized child/debug path. Supply the stage's required
input/output paths and `gpu_manager` settings. There is no host-specific runner.
Do not launch GPU optimization directly alongside unregistered workloads.

The distinction between a distribution center and physical geometry is strict:
see [the N/U coordinate contract](fiber_layout_coordinates.md). Integer-label
curves are not evaluated at a fractional distribution center, and old saved
geometry is not silently reinterpreted. Reference alignment is global, not a
per-fiber fit. Ordered evidence and family conventions are persisted.
An [initialization-width prior must not be reintroduced](fiber_layout_no_width_prior.md).


`fiber_gt_constraint_score.run(settings)` can build cross-reference event caches
as well as consume them. For construction, supply `configs`, `gt_root`,
`extract_template`, `orient_template` and `output_root`; optionally set
`native_binary` to the built `vc_fiber_constraints`. `events_only: true`
writes `cross_events.json`; `prepare_only: true` stops after native extraction.
The frozen GT normal sweep is sampled through `fiber_curve_bank`;
no surface reconstruction is required. With `frozen_event_cache`, the existing
scorer reuses those events. Input hashes prevent incompatible cache reuse.

`fiber_enclosure_width_init.fit_enclosure_widths` retains the constrained
least-squares width initialization from opposite-sign H/V enclosure crossings,
using measured phases/arclength and caller-supplied integer labels.

## Minimum flow and splitting

`fiber_min_flow.solve` accepts explicit node/edge arrays, targets, weights and
optional one-sided penalties. It returns integer winding labels and a primal/
dual certificate. `Seam` and `corrected_targets` transport fiber and connector
crossings in the same gauge. `fiber_min_flow_split` retains the existing
arc-based split proposal/solve support. The cached AFV adapter reuses shared
AFV and polyline geometry rather than copying loaders.

Validation (from `vesuvius/`):

```sh
python -m pytest -q tests/test_fiber_min_flow.py tests/test_fiber_min_flow_split.py tests/test_fiber_uv_layout.py tests/test_fiber_shared_curve.py
```

`test_constraint_extraction.py` is an opt-in real-data exporter check; its four
`VC_CONSTRAINT_TEST_*` environment paths are documented in the test. Source
datasets, references, checkpoints and native build dependencies are not bundled.
No cross-platform execution or new performance/convergence claim is implied by
this extraction.

For reproducible managed replays of prepared no-GT / forced-GT / selectable-GT
solves, see [the shareable layout benchmark](fiber_layout_benchmark.md).
