# Require Lasagna Normals For Native Trace2CP Task Log

## Implementation Notes

- Started from the existing Python/C++ native Trace2CP smoothness paths.
- Added `_native_trace_requires_normal_sampler()` in Python and changed
  `_native_trace_cfg_with_effective_smoothness()` to raise when active
  tangent/normal or cumulative tangent smoothness has no `normal_sampler`.
- Changed Python tensor smoothness helpers to raise if active normal-aware terms
  are called without sampled candidate normals.
- Made `vc_fiber_trace_metric` require explicit `--normal-manifest` during
  argument validation and removed the CLI fallback that attempted to build a
  normal sampler from the fiber prediction manifest.
- Added a C++ core guard so default normal-aware tracing throws when callers pass
  a null normal sampler.
- Updated focused Python and C++ tests to pass synthetic normal samplers where
  they intentionally exercise default normal-aware tracing.
- Added Python and C++ regression tests for missing-normal-sampler failures.
- Updated specs, code-structure docs, and changelog to describe explicit
  Lasagna-normal requirements.

## Deviations / Deferred Items

- Independent agent review of `task_plan.md` was skipped because this session
  is proceeding directly in default execution mode; the plan was checked
  locally against the current spec/docs before implementation.
- Candidate-level invalid normal samples still use the existing per-candidate
  fallback behavior. This task only makes missing Lasagna normal samplers
  inaccessible from normal-aware native Trace2CP entrypoints.
- The first parallel C++ build attempt raced on the shared `vc_fiber_tracer`
  object while building `test_fiber_trace3d` and `vc_fiber_trace_metric`
  simultaneously. Re-running the metric build after the shared target completed
  passed.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. python -m pytest vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp"`
  - passed: 58 selected tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. python -m pytest vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - failed in an unrelated existing progress-format assertion:
    `test_native_3d_whole_fiber_progress_reports_compact_error_units_when_known`
    expected one newline but current progress output emits an initial pending
    line and a final line.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  - passed.
- `cmake --build volume-cartographer/build --target test_fiber_trace3d`
  - passed.
- `cmake --build volume-cartographer/build --target vc_fiber_trace_metric`
  - passed after serial rerun.
- `volume-cartographer/build/bin/test_fiber_trace3d`
  - passed: 8 test cases.
- `volume-cartographer/build/bin/vc_fiber_trace_metric --help`
  - passed; usage shows required `--normal-manifest`.
- `volume-cartographer/build/bin/vc_fiber_trace_metric dummy.lasagna.json dummy.json`
  - failed before manifest I/O with the intended required `--normal-manifest`
    error.
- `git diff --check`
  - passed.
