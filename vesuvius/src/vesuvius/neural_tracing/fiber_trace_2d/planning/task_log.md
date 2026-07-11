# Task Log

Current task: exclude unreachable patch edges from contrastive embedding
negative supervision.

Plan review:
- The task is a local correction to the contrastive loss semantics.
- The planned mask preserves positive supervision and deterministic negative
  pairing while removing the false edge-negative signal.
- No config migration is needed; the mask is derived from existing
  `augment_shift_x/y` and patch shape.

Implementation:
- Added `contrastive_negative_reachable_mask(...)` and an optional
  `negative_candidate_mask` argument to `contrastive_embedding_loss(...)`.
- Training and benchmark contrastive calls now pass a per-run reachable mask
  derived from patch shape and enabled `augment_shift_x/y`; if augmentation is
  disabled, the reachable region collapses to the CP-neighborhood around the
  center.
- Updated specs, code docs, and changelog.
- Added a regression test that verifies high-similarity edge pixels do not
  contribute as negatives when outside the reachable CP-neighborhood region.

Validation:
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/embedding.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k 'contrastive_embedding'`
  - Result: 2 passed, 148 deselected in 2.77s.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 150 passed in 5.58s.
- `git diff --check` on touched files passed.
