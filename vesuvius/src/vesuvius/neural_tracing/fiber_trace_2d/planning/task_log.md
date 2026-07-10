# Task Log: Restore Exhaustive Deterministic Random Sample Order

## Implementation Notes

- Replaced trainer/prefetch use of flat sequential sample mode with the default
  deterministic pseudo-random sample mode.
- Changed loader random sampling from random-with-replacement to deterministic
  random-without-replacement full-dataset passes.
- Each pass sorts all flat CP indices by seeded content-based random keys, so
  every configured CP is visited once before the stream repeats.
- Full-dataset prefetch (`--prefetch-steps 0`) now starts at sample position 0
  and covers exactly one complete deterministic-random pass for train and test
  loaders.
- Positive `--prefetch-steps N` still starts at the requested training-step
  offset and covers the same deterministic random prefix that training would
  consume from that step.
- Retained explicit `sample_mode="flat"` only as a debug/test mode; normal
  training and training prefetch do not use it.
- Updated specs, code docs, local development notes, and changelog.
- Added focused tests for exhaustive random pass coverage and updated prefetch
  mode/count expectations.

## Deviations

- The plan said repeated passes could be deterministic but not necessarily
  identical. The implementation uses pass-specific seeded keys, so repeated
  passes are deterministic and may have different random order.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  passed with `49 passed in 5.63s`.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  passed.
- `git diff --check -- <touched fiber_trace_2d files>` passed.
