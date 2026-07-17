# Native 3D Trace2CP Vectorized Beam Lookahead Log

## Implementation

- Added torch cone-candidate generation for both angle-step candidates and the
  legacy square-grid fallback. The public NumPy candidate helpers remain for
  compatibility and tests.
- Added batched current-point branch selection. Real native caches use
  `sample_point_choices_torch(...)` across all active states; legacy/fake
  caches without branch choices keep the previous `sample_point(...)` fallback
  behavior.
- Generalized candidate scoring to evaluate `[frontier, candidate, branch]`
  tensors in one call. The existing single-state scorer now wraps the batched
  scorer, so greedy tracing and focused tests keep the same API.
- Replaced the beam lookahead inner loop with tensor frontier expansion:
  candidate generation, candidate scoring, target-plane crossing, cumulative
  loss, and near-duplicate pruning operate on tensors. Only the selected
  pruned frontier nodes or target-reaching path are reconstructed as Python
  `NativeTraceStep` chains.

## Validation

- Ran:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Result: `85 passed in 7.18s`.

## Deviations / Deferred

- The lazy inferred-block cache is still CPU-resident by design. Beam expansion
  batches point lookup calls, but routing points to inferred blocks and
  constructing missing blocks still crosses through CPU/NumPy before sampled
  tensors return to `cache.device`.
- Final path reconstruction remains Python object work because only the chosen
  path or the small pruned live frontier is reconstructed.
- I did not add a wall-clock benchmark command/harness in this task; validation
  is focused regression coverage plus the existing 3D test suite.
