# Fiber constraints, winding/UV layout and minimum flow

This extracts the existing fiber-to-constraint and fiber-layout implementations.
Inputs and earlier stage solutions are immutable; preparation records source
identities, hashes and coordinate conventions. Preparation, search, experiment
matrices, diagnostics and tifxyz surface export are documented in
[fiber-layout workflows](fiber_workflows.md). Run-specific configs and data are
external; portable usage samples are included.

## Storage and identities

All internal fiber geometry, measured relations and derived graph stores use schema22 vcobj. A store locator may be a plain SQLite vcobj file, an unpacked collection manifest or a packed multi-database file. Native store readers resolve shards and readonly store references; workflow consumers do not open physical SQLite tables or implement placement-specific fallbacks.

Fibers and relations retain complete global UUID keys. Native UUID columns use binary values; Python fiber catalogs and solver checkpoint `ids`/`source_id` arrays use canonical `U36` strings. Dense integer indices identify positions in solver arrays only. Compact constraint rows do not retain source event IDs, source-fiber ordinals or reversal flags. Runtime constraint indices follow edge-UUID order; endpoint values use canonical A/B order.

Each measured constraint creates one core relation and one `ext_fiber_graph_solve_constraints` row. The core relation stores UUID endpoints, integer kind and commit sequence, with no XYZ anchors, dependency JSON or adjacency indexes for this immutable workflow. The eight-column measurement format4 stores the edge UUID, two separate float32 little-endian arc BLOBs in base voxels, two three-byte signed Lasagna normals, a three-byte signed unit connector from A to B, one winding-distance byte and nullable explicit extra JSON. Winding code zero is invalid; other codes decode as `(code-1)/16` turns. Distances above 15.875 turns are skipped before rounding, never clamped. Scores, evidence weights, endpoint XYZ and temporary normal-piece diagnostics are not persisted.

Access to stored endpoint XYZ fails explicitly; a unit connector is ordering evidence, not a physical connector length. Consumers requiring geometry evaluate the original selected source curves separately at their arcs. See [the measurement format contract](../../vc_object_store/docs/extensions.md#fiber-graph-solve-extension) for codecs, frames and orientation reuse.

Graph outputs reference immutable inputs rather than duplicating geometry. Normal orientation creates a derived measurement overlay, preserving unchanged geometry, relations and the original source identity. Shared native `resolveStorePosition` / Python `CollectionReader.resolve_store_position` resolves an exact source snapshot to its public store locator across plain, sharded, packed and referenced stores.

Measurement provisioning records the original `fiber_source` position (`store`, `generation`, `commit_sequence`), plus compact normal-source identity. Frozen geometry reads and axial features use that original scope with the extracted membership, not an overlay root relabeled with an earlier pin. Verification compares v3 complete-owning-reference fiber digests against visible graph geometry, rejecting local geometry overrides; the source-position guard also detects coordinate-registration changes. Missing identities require regenerated evidence. No per-fiber provenance list is added to provisioning.

Python geometry access is shared through `tracers.shared.annotations.fiber_store`; graph access uses `fiber_graph_store.FiberGraph` backed by the native compact reader. Immutable source/subset creation uses `write_initial_fibers` with measured line points, not invented controls. Mutable annotation APIs remain separate; AFV is an external import source, not an internal graph/result format. Cache identities include store positions or the shared v3 complete-reference fiber digest. Rebuild caches whose source/measurement encoding or identity is incompatible; do not rename them into compatibility.

## Native fixed evidence

`vc_fiber_constraints CONFIG.json` reads fibers through the composed store reader in the volume named by `coordinate_space` (`SAMPLE/VOLUME_ID`). It validates inference scan identity/base shape and computes local parallel/perpendicular evidence before compact storage quantization. Geometry search, acceptance tests and solver loss definitions are unchanged; float32 arcs, compact normals and sixteenth-turn stored distances are explicit format choices.

Required settings are `fibers`, `coordinate_space`, `normal_manifest`, `normal_provenance`, `remote_cache`, `output`, and either `full_volume: true` or `minimum_xyz`/`maximum_xyz` in base XYZ. Set `database` to the output vcobj locator and `repetitions: 1` to save a graph. Existing outputs are refused. `threads` controls CPU extraction; `load_threads` independently controls source-read fan-out (default 1). Source UUID ordering and deterministic assembly are retained; more readers need measurement rather than assuming linear scaling.

`import_workers` controls independent UUID-routed output shards (default 32); `packed_store` selects the packed container (default true). Extension rows follow their relation/object key into the same shard. Import uses the shared native initial-import API and references source objects directly, without cross-shard endpoint copies. Legacy `sqlite_serialize_memory`, `sqlite_in_memory`, `sqlite_page_size` and `sqlite_insert_batch_rows` options are rejected: they described the removed AFC writer.

`stream_export: true` is the default when writing a graph. Finalized whole-pair event batches flow through a bounded ordered queue while extraction continues; `export_queue_batches` defaults to 128. This bounds queued batches, not the retained extraction report. Native import handles sharded writes, completion, sealing and final persistence; an error does not publish a completed graph. `stream_export: false` uses the same graph exporter after extraction for timing/control comparisons.

Spatial queries, encounter deduplication, source-order scoring and measured geometry retain their existing implementations. `candidate_batch_size` defaults to 262144 initial candidates as a scratch target, not an observation limit. The persistent worker pool schedules smaller independent jobs while keeping complete fiber-pair groups together. Results are assembled deterministically; worker-stage seconds are not additive wall time. No source samples or accepted constraints are discarded to improve storage throughput.

Set `progress_interval_seconds` and optionally `progress_file` for structured per-phase counts, rates, elapsed time and ETA. Units are reported per phase; an extraction-phase ETA is not a full-job ETA. `tail_after_extraction_seconds` measures export work remaining after extraction. Streamed export time overlaps extraction and must not be added to it. `processing_seconds` is end-to-end processing excluding optional verification.

`orient_database` selects a second pass using the fiber-local minimal-turn normal sweep and writes aligned normal blobs to a new referenced graph. `normal_curves_output` and `normal_prior_output` retain their geometry/prior roles. Reuse requires the recorded orientation run, sampling spacing and complete original source position. Validation preserves endpoint validity and checks transported orientation while reporting quantization-angle discrepancies; it does not assert float64 byte equality after intentional arc/normal quantization. Spatial extraction can sample pieces independently, merging signs at identical shared arc samples before publication; no piece IDs or correction tables remain in the final graph.

`compare_database` is an explicit regression check against a saved measured graph. It verifies measurements and ordering without changing their interpretation; `comparison_seconds` is reported separately. Read-only diagnostic validation is not part of ordinary bulk-import timing. The previous AFC-specific SQL cloning, JSON-normal UPDATE batches and schema1/schema2 database checks are not used.

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
arc-based split proposal/solve support. The cached fiber adapter reuses the
shared fiber store reader and polyline geometry rather than copying loaders.

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
