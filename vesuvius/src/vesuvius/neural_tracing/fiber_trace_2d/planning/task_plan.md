# Trace2CP Metric Plan

## Implementation

- Add a lightweight Trace2CP metric dataclass and helper that consumes the two
  directional traces plus start/target CP strip coordinates.
- Define public Trace2CP metric error as:
  `closest_abs_y_gap / abs(target_x - start_x)`.
- Use the actual closest y-gap across overlapping trace x samples. Do not apply
  the center-focus penalty for metric selection.
- If no usable overlap exists, return the default maximum error:
  edge distance from the CP/centerline y to the nearest usable vertical edge,
  divided by segment width.
- Keep existing endpoint and center-biased refinement diagnostics for
  visualization rows, but rename/publicly separate them from the metric.
- Add a no-refinement bidirectional metric path for training/test evaluation.
- Add a loader/test evaluation loop that evaluates each selected held-out CP
  against the next CP segment and averages valid segment metric errors.
- Use the new averaged test Trace2CP metric for `best.pt` selection when
  `test_datasets` is configured. Keep current snapshots at test cadence.
- Log/print the metric in training, TensorBoard, single Trace2CP vis, and
  whole-fiber Trace2CP vis summaries.

## Spec Update

- Replace public center-biased Trace2CP score wording with vertical-error per
  horizontal-span metric wording.
- Document that center-biased closest approach remains a visualization/refine
  diagnostic only.
- Document test evaluation and best snapshot selection by averaged held-out
  Trace2CP metric.
- Document fallback max-error semantics for no-overlap/edge cases.

## Docs Updates

- Update `docs/code_structure.md` to mention the shared Trace2CP metric helper
  and training test integration.
- Update `planning/changelog.md`.
- Replace `planning/task_log.md` with current-task implementation notes and
  validation results only.

## Tests

- Add focused unit tests for unpenalized closest-gap metric selection.
- Add a fallback metric test for no-overlap/edge cases.
- Add a training test that confirms best checkpoint selection uses the
  Trace2CP metric when a test loader is present, using monkeypatched lightweight
  evaluation.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
