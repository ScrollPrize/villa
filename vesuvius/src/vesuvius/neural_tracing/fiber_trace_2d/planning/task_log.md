# Trace2CP DP Local-Angle Semantics Task Log

- Replaced the previous task docs with the current Trace2CP DP local-angle
  semantics task.
- Changed side Trace2CP DP horizontal transitions from 32 px to 4 px.
- Removed the conversion of `max_direction_angle_degrees` into a hard
  `dy_limit`; candidate angle now only affects the local direction-field
  angular excess penalty.
- Kept a separate broad compute search band at `4 * horizontal_step`, capped by
  strip height, so steep local directions above 45 degrees are representable
  without making the DP state unbounded.
- Reduced shared DP second-order smoothing penalties from `dy=0.05`,
  `dz=0.1` to `dy=0.005`, `dz=0.01`.
- Added a regression test for a steep locally aligned DP path with a small
  candidate-angle setting.
- Validation:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'trace2cp_joint_dp_candidate_angle_does_not_cap_global_slope or trace2cp_top_monotone_direction_path_z_torch_matches_numpy or trace2cp_top_monotone_direction_path_z_progress_prints_eta or trace2cp_joint_dp or trace2cp_combined or trace2cp_z_search'`
    passed: 14 passed, 214 deselected.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed: 228 passed.
  - `git diff --check -- <touched files>`
    passed.
