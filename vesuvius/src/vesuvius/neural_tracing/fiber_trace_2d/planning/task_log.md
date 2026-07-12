# Top-View Loader Performance Parity Task Log

## Implementation Notes

- Confirmed the regression source from the top-view commit: side-view loading
  used source caching plus batched line/coordinate/sampler paths, while
  top-view loading rebuilt a separate source grid per CP without cache and used
  top-specific profile keys that were hidden from the benchmark columns.
- Added view-specific top source-cache keys while preserving existing side-view
  cache identity exactly.
- Changed `build_top_strip_source` to use the same cache load/store payload as
  side source construction.
- Changed top-view preparation to use the side-style batched coordinate
  augmentation helper and grouped `CoordinateSampler.sample_coord_batch`.
- Changed top-view profiling to aggregate into the existing benchmark stages:
  `load_batch_wall`, `load_batch_worker`, `descriptor`, `strip_coord_cache`,
  `line_window`, `lasagna_normals`, `strip_coords`, `coord_augmentation`, and
  `volume_sample`.
- Removed duplicate top line/CP transformation by reusing the already
  transformed center side-strip sample line/CP pixel coordinates. For the same
  source/output patch frame and geometric augmentation, those coordinates are
  identical between side and top views.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: `201 passed in 5.61s`.

## Performance

Command used for all profile measurements:

`PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`

- Initial current-code probe before changes was interrupted after confirming
  the problem: top source/load work was hidden in `outside`, and top source
  construction did not use the side-view cache path.
- After adding top source cache and batched top coord path, first full run
  populated cold top cache entries: `83.09 patches/s`, with
  `source_geom=6.997 ms/patch`.
- Fully cached rerun after source cache warm-up: `123.87 patches/s`,
  `source_geom=0.083 ms/patch`, `line=5.628 ms/patch`.
- After reusing side line/CP coordinates for top patches: `135.03 patches/s`,
  `source_geom=0.079 ms/patch`, `line=2.775 ms/patch`,
  `loading=2.538 ms/patch`.

## Remaining Notes

- The remaining steady-state loader cost is dominated by per-CP source-cache
  reads (`coord_cache=9.651 ms/patch` in summed worker-time accounting) and
  side-view line-coordinate work. Top-view no longer performs its own source
  generation or line-coordinate lookup on the warm path.
