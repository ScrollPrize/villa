# Cross-Fiber Contrastive CP Negatives Plan

## Scope

- Update contrastive embedding training only.
- Keep direction loss, loader sampling, Trace2CP tracing, and visualization
  semantics unchanged.
- Do not add new config keys; use the existing
  `training.contrastive_negative_margin` and `training.contrastive_weight`.

## Implementation

1. Extend `contrastive_embedding_loss`.
   - Reuse the normalized CP embeddings already sampled for positive terms.
   - Build a cross-fiber mask from existing per-sample fiber IDs.
   - Penalize same-margin cosine similarity for all CP embedding pairs whose
     fiber IDs differ.
   - Average available negative components: existing valid non-CP pixel
     negatives and cross-fiber CP negatives. This gives each component equal
     weight when both exist while preserving old behavior when only one exists.

2. Update contrastive grouped batches.
   - Concatenate consecutive same-fiber CP groups to fill a training batch
     instead of repeating one group across the whole batch.
   - Advance contrastive training steps by the number of groups per batch so
     adjacent steps do not overlap groups.
   - Synchronize value/image augmentation within each same-fiber group.

3. Extend metrics.
   - Keep `negative_loss` as the aggregate negative branch used by the loss.
   - Add component metrics for pixel negatives and cross-fiber CP negatives.
   - Report the total negative comparison count while preserving existing
     single-fiber behavior.

4. Wire metrics through training logs.
   - Add TensorBoard scalars for the component losses/counts when contrastive
     training is enabled.
   - Keep existing scalar names intact for aggregate values.

5. Add tests.
   - Existing same-fiber tests should continue to show unchanged weighting.
   - Add a cross-fiber batch test proving the new CP-negative component is
     present and splits the negative branch weight with pixel negatives.
   - Update grouped-batch tests to verify multiple same-fiber groups are
     concatenated and value augmentation is synchronized per group.

## Spec Update

- Update `planning/specs.md` contrastive section to document cross-fiber CP
  embedding negatives and the averaged negative-branch weighting.

## Docs Updates

- Update `docs/code_structure.md` training/contrastive documentation with the
  new negative component and metrics.
- Update `planning/changelog.md`, `status.md`, and `task_log.md`.

## Validation

- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run `git diff --check` on touched files.
