# Startup Compact Geometry Acceleration Task Log

## Planning Notes

- Baseline from the previous task:
  - startup compact geometry build: `464` records, `14773` valid CPs,
    `184` skipped CPs, `63.5 MiB`, `8m53s`;
  - hot-path profile after startup: `compact_geometry=0.012 ms/patch`.
- Current bottleneck is loader construction, specifically serial Lasagna normal
  sampling and decoding over line points.
- This task should not add a new worker-count config key. Startup geometry uses
  the existing `loader_workers`; `loader_workers=1` remains the serial path.

## Implementation Log

- Added CP-window line range discovery before startup normal sampling.
  Geometry arrays remain full-record sized, but only required line ranges are
  sampled and marked valid.
- Added batched 2x2x2 channel interpolation for Lasagna `grad_mag`, `nx`, and
  `ny`, vectorized normal decoding through Lasagna `_decode_normals`, and a
  batched principal-axis solve matching the scalar ambiguity handling.
- Kept `_lasagna_normal_at_zyx()` as the scalar debug/reference path.
- Added record-level startup construction with `loader_workers`; the serial
  path is still selected by `loader_workers=1`, and the final store remains
  indexed by original record order.
- Updated specs, code-structure docs, local-development notes, and changelog.

## Validation

- Targeted geometry tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py::test_loader_tolerates_invalid_remote_line_endpoints_outside_cp_window vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py::test_loader_samples_only_cp_local_lasagna_normals vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py::test_vectorized_lasagna_normals_match_scalar_path vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py::test_parallel_startup_geometry_matches_serial`
  - Result: `4 passed in 2.35s`.
- Focused loader suite:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `266 passed in 7.46s`.
- Startup/load-only/profile benchmark:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`
  - Startup compact geometry: `464` records, `14773` valid CPs, `184`
    skipped CPs, `40.0 MiB`, `2m12s`.
  - Benchmark: `100` batches, `12800` patches, `106697.8 ms`,
    `119.96 patches/s`.
  - Profile summary highlights: `coord_gen=13.856 ms/patch`,
    `compact_geometry=0.012 ms/patch`, `source_geom=10.903 ms/patch`,
    `loading=7.536 ms/patch`.
  - Baseline comparison: startup improved from `8m53s` to `2m12s`, compact
    memory from `63.5 MiB` to `40.0 MiB`, and load-only throughput from
    `76.76` to `119.96 patches/s`.
