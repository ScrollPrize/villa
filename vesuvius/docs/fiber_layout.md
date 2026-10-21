# Fiber constraints, winding/UV layout and minimum flow

This extracts the existing AFV-to-constraint and fiber-layout implementations.
Inputs and earlier stage solutions are immutable; preparation records source
identities, hashes and coordinate conventions. Experiment matrices, run-specific
configs/data and layout-to-Lasagna surface reconstruction are not included.

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
