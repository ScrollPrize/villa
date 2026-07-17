# Native 3D Trace2CP Sampled CP Start Direction Log

## Implementation Notes

- Task started from `planning/todo.md` item:
  "for cp start: lets actually _not_ use the cp dir but the sampled dir that
  most closely aligns with the cp dir. The do apply smoothness and dir
  supervision directly as applicable".
- Added a native 3D Trace2CP start-specific branch sampler that chooses by
  pure angular agreement to the CP-local tangent and ignores branch presence.
- Greedy and beam search now initialize current/previous/history direction from
  that sampled CP direction.
- Removed the first-step relaxation/gate path from production candidate
  scoring, including zeroed first-step smoothness and all-pairs neutralization.
- Updated focused tests, specs, docs, changelog, and marked the todo complete.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py`
  passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
  passed: `103 passed in 3.40s`.

## Deviations

- No intentional simplifications or deferred requirements.
