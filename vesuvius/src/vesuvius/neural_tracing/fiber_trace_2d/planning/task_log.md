# Trace2CP Side-DP Z Smoothness Task Log

- Confirmed side/joint Trace2CP DP previously passed `z_transition_penalty=0.1`
  and `dz_smooth_penalty=0.0` into the monotone DP helper.
- Added side-DP-specific constants:
  - `_TRACE2CP_SIDE_DP_Z_TRANSITION_PENALTY = 0.0`
  - `_TRACE2CP_SIDE_DP_DZ_SMOOTH_PENALTY = 0.05`
- Updated `_trace_score_trace2cp_joint_dp_bidirectional` to pass those
  constants, so side-DP no longer penalizes total z movement and only penalizes
  abrupt changes in z step.
- Left the lower generic DP helper defaults unchanged so top-model DP
  diagnostic behavior is not changed by this side-DP task.
- Added a wrapper-level regression test proving side/joint DP passes zero
  `z_transition_penalty` and nonzero `dz_smooth_penalty`.
- Updated `planning/specs.md`, `docs/code_structure.md`,
  `planning/task.md`, `planning/task_plan.md`, `planning/status.md`, and
  `planning/changelog.md`.
- Validation:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'joint_dp_uses_z_smoothness_without_z_step_penalty or explicit_dp_uses_dp_backend or z_search_explicit_dp_uses_dp_backend or top_monotone_direction_path_z_torch_matches_numpy'`
    passed: 4 passed, 240 deselected.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed: 244 passed.
