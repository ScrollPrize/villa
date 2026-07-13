# Torch-Vectorized Trace2CP DP Backend Task Log

- Started from the shared monotone Trace2CP DP helper used by side combined
  tracing, z-search tracing, and top-model diagnostic tracing.
- Goal: replace the slow Python loops inside each DP column with torch tensor
  work while preserving the sequential column recurrence and existing fallback
  behavior.
- Added `_trace2cp_top_monotone_direction_path_z_torch`, which keeps active DP
  costs on the torch device and CPU-stores backpointers. It vectorizes each
  DP column over move chunks, all z layers, all rows, all previous moves, and
  all sampled transition columns.
- CLI side combined DP, side z-search DP, and top-model diagnostic DP now pass
  their existing torch device into the shared helper. Direct helper calls with
  no device still use the NumPy/Python backend.
- Added effective pruning before the torch work:
  - side-z only infers/optimizes z layers reachable by a center-anchored path
    for the current horizontal transition count;
  - side DP derives a hard vertical move cap from the candidate-angle limit.
- Local synthetic CPU comparison command:
  `PYTHONPATH=vesuvius/src:. python -c '<synthetic 9-layer 96x321 DP benchmark>'`
  measured fallback NumPy/Python at `10.6454s`; torch CPU with chunk 16 at
  `0.8672s` in the chunk sweep, about `12.3x` faster on this small case.
- Validation:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'trace2cp_top_monotone_direction_path_z_torch_matches_numpy or trace2cp_top_monotone_direction_path_z_progress_prints_eta or trace2cp_joint_dp or trace2cp_combined or trace2cp_z_search'`
    passed: 13 passed, 214 deselected.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
    passed: 227 passed.
