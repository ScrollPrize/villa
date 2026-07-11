# Plan

## Implementation

- Add configurable training keys for contrastive embedding:
  `contrastive_enabled`, `contrastive_embedding_channels`,
  `contrastive_control_points_per_fiber`, `contrastive_weight`, and
  `contrastive_negative_margin`.
- Extend `FiberStripDirectionNet` with an optional embedding head. The forward
  output remains one tensor: first two channels are the existing direction code,
  remaining channels are raw embedding channels.
- Slice direction channels explicitly in training and runner code so existing
  direction behavior continues to work with embedding-enabled checkpoints.
- Add a grouped same-fiber training batch loader path:
  deterministic grouped random order over fibers/CP groups, `N` CPs from one
  fiber repeated to fill the batch, unique raw sample indices for independent
  geometric augmentations, and a shared value-augmentation seed per group.
- Add a cosine contrastive loss:
  positives are all CP-neighborhood supervision pixels within the same fiber;
  negatives pair each positive with one deterministic valid non-CP pixel from
  the batch; positive and negative terms are averaged separately then weighted
  equally before applying `contrastive_weight`.
- Log train/test contrastive scalars and include them in the printed training
  line.
- Add TensorBoard image visualization of embedding similarity for the same
  selected training/test patches as direction visualization.

## Spec Update

- Update `planning/specs.md` to state the model may emit optional embedding
  channels after the two direction channels.
- Document the contrastive grouped sampling mode, synchronized value
  augmentation, cosine embedding loss, and similarity visualization.

## Docs Updates

- Update `docs/code_structure.md` for model/train/loader behavior.
- Add a changelog entry for contrastive embedding training.

## Tests

- Unit test model shape with and without embedding channels.
- Unit test grouped same-fiber CP batch construction and repeated CPs.
- Unit test value augmentation sharing in grouped contrastive samples while
  geometric parameters remain independent.
- Unit test the contrastive loss balances positive and negative terms and
  handles optional embedding channels.
- Run the focused fiber_trace_2d test file.
