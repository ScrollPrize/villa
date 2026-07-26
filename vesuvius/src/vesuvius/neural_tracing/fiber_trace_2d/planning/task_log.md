# VC3D BBox Dependency Metadata For 3D Prefetch Task Log

## Implementation Notes

- Added `Volume.collect_bbox_dependencies(offset, shape, level=0)` to the
  VC3D Python binding. It reuses the existing selected-level ZYX
  `collectChunkKeys(...)` conversion and the same per-chunk metadata dict
  emitted by `collect_coords_dependencies(...)`.
- Added `CoordinateSampler.chunk_requests_for_bbox(start_zyx, end_zyx)` and
  implemented it for `Vc3dCoordinateSampler` by calling the new VC3D binding.
  Local `NumpyZarrCoordinateSampler` returns no remote chunk requests.
- Switched 3D prefetch to compute a clamped selected-level augmentation-envelope
  bbox and pass that bbox directly to the sampler. The previous chunk-center
  coordinate materialization path is gone from 3D prefetch. The prefetch bbox
  is no longer rounded to zarr chunk boundaries in Python; VC3D owns bbox-to-
  chunk conversion.
- Removed `prefetch_sampler_device` from 3D config parsing, tests, and the
  active `train_s1a_nml_all_64_sd2.json` config. The remaining producer
  concurrency knob is `prefetch_sampler_workers`.
- Updated specs, code-structure docs, local-development notes, and changelog to
  describe VC3D-owned bbox-to-chunk dependency metadata.

## Deviations / Deferred Items

- No planned runtime behavior was intentionally skipped.
- VC3D bbox smoke was limited to checking that the installed binding exposes
  `Volume.collect_bbox_dependencies`; no live remote-volume prefetch was run.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'prefetch or dependency'`
  passed: 9 passed, 101 deselected.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/sampling.py`
  passed.
- `python -m pip install -e volume-cartographer --no-deps --break-system-packages`
  rebuilt and reinstalled the editable VC3D package without dependency changes.
- `python -c "from vc.volume import Volume; print(hasattr(Volume, 'collect_bbox_dependencies'))"`
  printed `True`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: 110 passed.
