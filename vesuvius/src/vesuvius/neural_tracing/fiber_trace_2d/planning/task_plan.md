# Default Training Without Embedding Loss Plan

## Scope

- Change the standard training path so default/example runs optimize direction
  plus presence only.
- Preserve explicit contrastive embedding support for experiments.
- Avoid changing Trace2CP embedding inspection flags or existing embedding loss
  implementation.

## Implementation

- Add an effective training helper that returns embedding channels only when
  `training.contrastive_enabled` is true.
- Use that helper when constructing training and benchmark models, so a disabled
  contrastive config cannot create an unused embedding head.
- Update `configs/loader_example.json` to remove the contrastive opt-in keys
  from the standard example run.
- Keep validation requiring `contrastive_embedding_channels > 0` when
  `contrastive_enabled` is true.
- Add focused tests for:
  - default training config has contrastive disabled and no embedding channels;
  - disabled contrastive with a stale positive channel count builds a
    direction+presence model only;
  - enabled contrastive still uses the configured embedding channel count.

## Spec Update

- State that standard training is direction plus presence; contrastive
  embedding is experimental explicit opt-in.
- Clarify that embedding channels are appended only for effective contrastive
  training, not merely because a stale channel count exists.

## Docs Updates

- Update `docs/code_structure.md` training/config sections.
- Update `planning/status.md`, `planning/task_log.md`, and changelog.

## Tests

- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
