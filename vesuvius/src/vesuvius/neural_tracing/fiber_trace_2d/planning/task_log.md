# Process-Level Startup Fiber Geometry Preload Task Log

## Planning Notes

- User rejected the thread-local zarr/VC3D handle direction.
- Correct requirement is startup preload by splitting fibers/records across
  independent worker processes.
- Geometry must be built during loader construction, not lazily during the
  first training pass.
- Parent process must own one compact in-RAM geometry store after startup.
- Keep deterministic serial-vs-parallel output equivalence.

## Implementation Log

- Replaced the current planning files with the process-level startup preload
  task and plan before changing loader code.
- Removed the `_RecordOpenSpec` / worker-local record-opening path from
  `loader.py`.
- Restored cloning to share loaded record metadata and the compact geometry
  store while creating fresh samplers for the clone.
- Added `_GeometryBuildJob` and a module-level
  `_build_fiber_line_geometry_process(...)` worker entry point. Worker
  processes reopen the configured base volume and Lasagna channels, build the
  compact geometry for their assigned record, and return compact geometry plus
  memory accounting to the parent.
- Changed startup geometry initialization to use `ProcessPoolExecutor` when
  `loader_workers > 1`, preserving the direct serial path for
  `loader_workers=1`.
- Runtime `load_batch` and top-batch preparation now use normal descriptor
  tuples and the parent-owned compact store; they no longer have descriptor
  index or worker-local record switches.
- Updated specs, code-structure docs, changelog, and loader tests to reflect
  process-level startup preload rather than thread-local record handles.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed: 268 passed in 7.58s.
- The focused test environment denies Python 3.14 forkserver socket creation,
  so the serial-vs-parallel startup equality test monkeypatches the executor
  symbol to run submitted worker jobs without spawning OS processes. The real
  example benchmark below exercised the actual process startup path.
- Benchmark command:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile`
  completed. Startup geometry built 464 records with 32 workers in 12.7s,
  producing 14,773 valid CPs, 184 skipped CPs, and a 40.0 MiB compact store.
  The load-only benchmark processed 12,800 patches in 155,183 ms
  (82.48 patches/s).
