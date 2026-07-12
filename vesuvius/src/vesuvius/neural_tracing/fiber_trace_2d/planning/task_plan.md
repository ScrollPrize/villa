# Contrastive Similarity-Mean Sparsity Loss Plan

## Scope

- Update contrastive embedding training only.
- Keep loader sampling, direction loss, Trace2CP tracing, and visualization
  semantics unchanged.
- Do not add a new config key; the requested target is fixed at `0.1`.

## Implementation

1. Extend `contrastive_embedding_loss`.
   - Reuse normalized embedding fields and CP embeddings.
   - For each supervised CP sample, compute the per-pixel cosine similarity to
     that CP embedding, mapped to normalized `0..1` space.
   - Average that normalized similarity over valid pixels in the same patch.
   - Add an MSE term against target `0.1`, averaged across supervised CPs.
   - Keep the existing positive/negative pair balance intact, then add this
     sparsity term under the same `contrastive_weight`.

2. Extend metrics and logging.
   - Add similarity-mean loss, observed mean value, target, and sample count to
     `ContrastiveEmbeddingMetrics`.
   - Thread these metrics through training and TensorBoard scalars.

3. Add/update tests.
   - Update existing contrastive loss tests for the additional sparsity term.
   - Add focused assertions for the observed normalized similarity mean and
     target loss.

## Spec Update

- Document the fixed `0.1` normalized similarity-image mean target and how it
  is combined with the contrastive pair loss.

## Docs Updates

- Update `docs/code_structure.md` contrastive training description.
- Update `planning/changelog.md`, `status.md`, and `task_log.md`.

## Validation

- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run `git diff --check` on touched files.
