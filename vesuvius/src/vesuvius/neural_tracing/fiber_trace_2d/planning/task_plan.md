# Trace2CP Side-DP Z Smoothness Re-Enable Plan

## Implementation

1. Change the side-view Trace2CP joint-DP dz smoothness default back to `0.5`.
2. Keep `z_transition_penalty` at `0.0`.
3. Update the regression test that captures the DP penalty values.

## Spec Update

- Update side-DP specs to state the per-step z movement penalty is zero and
  second-order dz smoothness is enabled with coefficient `0.5`.

## Docs Updates

- Update `docs/code_structure.md` where it describes side DP z smoothing.
- Replace `task_log.md` with this task's implementation notes and validation.
- Add a changelog entry because this changes Trace2CP DP default scoring.

## Testing

- Run the focused fiber_trace_2d loader/runner regression suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```
