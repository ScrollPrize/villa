# Fiber Presence Head And Z-Selected Embedding Supervision Task Log

## Result

- Added optional model presence output layout:
  direction channels first, optional one-channel sigmoid presence output next,
  embedding channels after presence.
- Added balanced sheet/fiber presence loss:
  transformed CP pixels are positive, reachable valid non-CP pixels are
  negative, and unreachable shift-margin edges are ignored.
- Changed contrastive positives to z-selected same-fiber matches:
  each anchor CP sample/strip-z offset trains only against the already
  most-similar other CP from the same fiber across loaded offsets.
- Threaded presence and output slicing through training, TensorBoard
  visualization, benchmarking, and runner checkpoint loading.
- Enabled `presence_enabled` and `presence_weight` in
  `configs/loader_example.json`.
- Updated specs, code-structure docs, changelog, and status.

## Deviations

- None.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/embedding.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `179 passed in 6.95s`.
