# Trace2CP Last-Point Similarity Columns Plan

## Scope

- Change only the single-pair Trace2CP embedding-debug visualization.
- Keep Trace2CP tracing, refinement, public metrics, best-checkpoint selection,
  CP/global similarity panels, and training losses unchanged.

## Implementation

1. Add a trace-progress similarity map helper in `runner.py`.
   - Validate the normalized embedding field and valid mask.
   - Iterate trace points in placement order.
   - For each newly placed point, sample the previous accepted point's
     embedding.
   - Compute cosine similarity for the vertical column band around the newly
     placed point.
   - Paint only that band, leaving unvisited columns as `NaN`.

2. Use the configured Trace2CP step length.
   - Pass `step_px` from `_evaluate_trace2cp_pair` into
     `_trace2cp_similarity_debug`.
   - Use `ceil(step_px / 2)` as the column-band radius.

3. Preserve existing debug panels.
   - Start CP, target CP, and same-fiber/global CP-bank maps remain full-image
     fixed-scale cosine maps.
   - Forward and reverse trace panels keep their overlays, but their similarity
     image becomes the column-band trace-progress map.

## Spec Update

- Update Trace2CP visualization specs so forward/reverse last-similarity maps
  are described as trace-step column-band maps, not full-image maps against the
  final trace embedding.

## Docs Updates

- Update `docs/code_structure.md` with the same concise behavior description.
- Update `planning/changelog.md`, `status.md`, and `task_log.md`.

## Validation

- Extend the existing Trace2CP similarity-debug regression to prove that:
  - early trace columns are painted from early trace-point embeddings;
  - later trace columns are painted from later trace-point embeddings;
  - unvisited columns remain unpainted.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run `git diff --check` on touched files.
