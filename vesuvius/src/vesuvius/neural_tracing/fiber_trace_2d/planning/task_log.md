# Trace2CP Timing Rows Task Log

- Started from the side-strip DP Trace2CP implementation.
- Goal: print one compact timing table per Trace2CP command, aggregated by
  stage for whole-fiber runs.
- Added timing row collection for Trace2CP source construction/sampling,
  inference, reference tracing, optional TTA, combined DP/z-DP, z-debug
  reconstruction, top-strip/top-model debug work, overlay rendering, and file
  output.
- Single-pair Trace2CP prints `trace2cp timings`; whole-fiber Trace2CP prints
  `trace2cp fiber timings` aggregated across valid pair evaluations.
- Investigated the choppy side-DP traces. The side DP was using short
  `--line-trace-step` values as its horizontal transition length, so integer
  y-state transitions quantized the visual slopes. Changed side DP to use a
  fixed 32 px horizontal transition while keeping `--line-trace-step` for
  output resampling.
- The combined DP now applies the existing candidate-angle max as an angular
  excess penalty, so the global DP path better matches the baseline direct
  candidate tracer's local angle cone.
- Added opt-in DP progress output to the shared monotone helper. Trace2CP CLI
  side DP, z-side DP, and top-model DP call sites pass labels; direct helper
  tests/calls stay quiet by default. Progress rows include solved columns,
  elapsed seconds, and `eta_s`; start/done/failed rows provide lifecycle
  markers.
- Validation:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'trace2cp_timing or trace2cp_joint_dp or trace2cp_top_monotone or trace2cp_combined or trace2cp_z_search'`
    passed: 20 passed, 206 deselected.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed: 226 passed.
  - `git diff --check -- <touched files>` passed.
