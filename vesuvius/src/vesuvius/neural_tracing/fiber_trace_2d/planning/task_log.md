# Trace2CP DP Routing And Side/Top Experiment Exclusivity Task Log

- Corrected the Trace2CP routing contract: DP remains available, but only when
  `--trace2cp-dp` is explicitly passed.
- Restored default combined Trace2CP to the regular stepwise candidate-fan
  tracer.
- Restored default `--trace2cp-z-search` to the older stepwise z-search tracer
  rather than the side-z DP backend.
- Added `--trace2cp-dp` CLI validation and propagation for single-pair and
  whole-fiber Trace2CP.
- Made `--trace2cp-side-top-z-experiment` exclusive in single-pair mode: it now
  builds only the segment source, center side prediction, original top strip,
  side/top-z experiment outputs, and its summary/debug directories. It does not
  run `_evaluate_trace2cp_refinement_chain`, write `trace2cp_vis.jpg`, or write
  `trace2cp_summary.txt`.
- Added focused tests covering default stepwise combined routing, explicit DP
  combined routing, default stepwise z-search routing, explicit z-DP routing,
  and exclusive side/top-z experiment export behavior.
- Updated `planning/specs.md`, `docs/code_structure.md`, `planning/task.md`,
  and `planning/task_plan.md` for the corrected semantics.
- Validation:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'combined_defaults_to_stepwise or explicit_dp_uses_dp_backend or z_search_defaults_to_stepwise or z_search_explicit_dp or side_top_z_experiment_export_is_exclusive'`
    passed: 5 passed, 235 deselected.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed: 240 passed.
  - `git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md`
    passed.
