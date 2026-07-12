# Fiber Presence Head And Z-Selected Embedding Supervision Task Log

## Planning Notes

- Read the local `fiber_trace_2d/AGENTS.md` workflow and reset this task log
  for the current task only.
- Read `planning/todo.md`; the relevant pending items are explicit
  sheet/fiber recognition and z-search embedding training that picks only the
  best same-fiber CP/z-offset positive.
- Inspected current implementation seams:
  - `model.py` currently outputs two direction channels followed by optional
    embedding channels.
  - `embedding.py` currently treats all same-fiber CP-local samples as
    positives and already has a reachable-mask helper for edge-ignored
    negatives.
  - `train.py` computes direction loss first and conditionally adds the
    contrastive embedding loss in both synchronous and prepared-batch paths.
  - loader samples already carry `fiber_path`, `control_point_index`,
    `strip_z_offset`, transformed CP coordinates, and valid masks, so the
    planned loss changes do not require a new loader path.

## Deviations

- None during planning.

## Validation

- Planning only; no tests run yet.
