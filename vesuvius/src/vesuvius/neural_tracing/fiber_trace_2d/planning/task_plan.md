# Reachable-Area Contrastive Similarity-Mean Sparsity Loss Plan

## Scope

- Update contrastive embedding training only.
- Keep loader sampling, direction loss, Trace2CP tracing, and visualization
  semantics unchanged.
- Do not add a new config key; reuse the existing reachable mask derived from
  configured shift augmentation.

## Implementation

1. Update `contrastive_embedding_loss`.
   - Reuse the already validated `negative_candidate_mask` reachable rectangle.
   - Pass `valid & reachable` into the similarity-mean sparsity term when a
     reachable mask is configured.
   - Keep the previous valid-only behavior when no reachable mask is supplied.

2. Update tests.
   - Extend the existing edge-handling regression so the masked
     similarity-mean value ignores unreachable edges.

3. Update docs.
   - Clarify that the similarity-mean sparsity term averages over valid
     reachable CP positions, not the whole patch, when shift bounds are known.

## Spec Update

- Document reachable-mask gating for the similarity-mean sparsity term.

## Docs Updates

- Update `docs/code_structure.md` contrastive training description.
- Update `planning/changelog.md`, `status.md`, and `task_log.md`.

## Validation

- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run `git diff --check` on touched files.
