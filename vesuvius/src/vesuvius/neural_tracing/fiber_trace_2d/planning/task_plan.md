# Fiber Presence Head And Z-Selected Embedding Supervision Plan

## Scope

- Replace the implicit “embedding becomes sheet/fiber presence” pressure with
  a dedicated per-pixel presence output and loss.
- Refine contrastive embedding positives for multi-z training so each CP is
  positively supervised only against the best same-fiber CP/z-offset partner
  already present in the batch.
- Keep direction supervision unchanged: the model still starts with the two
  Lasagna ambiguous direction channels and direction loss still uses only
  CP-local direction targets.
- Keep existing same-fiber batch construction; no loader rewrite is planned.

## Implementation

- Extend the model config with an optional `presence_channels` or boolean
  `presence_enabled` setting, and append one sigmoid presence channel after the
  direction channels and before/alongside embedding channels.
- Add explicit output slicing helpers in `model.py` so callers do not rely on
  hard-coded channel offsets once direction, presence, and embedding outputs
  coexist.
- Add training config keys for presence loss:
  `presence_enabled`, `presence_weight`, and optionally
  `presence_negative_margin` only if needed after implementation review. Keep
  defaults disabled/neutral unless the example config is intentionally switched
  on.
- Build presence targets from the same transformed CP coordinates already used
  by direction supervision:
  - positive mask: all rounded transformed CP pixels from the current flattened
    patch batch, filtered by image validity and bounds;
  - negative mask: valid pixels inside the shift-reachable CP region, excluding
    positive CP pixels;
  - unreachable edge pixels stay ignored for the negative term.
- Compute presence loss as balanced BCE or MSE with equal aggregate positive
  and negative weight. Prefer BCE if the output is a sigmoid probability; use
  MSE only if it better matches the existing code style after review.
- Log presence scalars in training/TensorBoard: total presence loss,
  positive/negative loss components, positive/negative sample counts, and
  optional mean predicted positive/negative probabilities.
- Add TensorBoard visualization for presence maps on train/test image logging
  steps, using fixed `0..1` scale and invalid pixels black.
- Replace the current contrastive positive construction in `embedding.py`:
  - group CP-local embedding samples by fiber and by control point;
  - for each anchor CP, consider only other CPs from the same fiber group;
  - include all available z-offset patch samples for each candidate CP;
  - compute cosine similarity for all anchor-offset/candidate-offset options;
  - choose the highest-similarity already-present candidate pair and supervise
    only that pair toward cosine `1.0`;
  - do not add positive loss for anchors with no other same-fiber CP in the
    batch.
- Keep embedding negatives and similarity-image sparsity separate from
  presence:
  - pixel negatives still ignore unreachable edges;
  - cross-fiber CP negatives remain disabled;
  - the similarity-image mean term can either remain initially or be removed in
    a follow-up if the explicit presence head makes it redundant. This plan
    does not remove it unless implementation reveals a direct conflict.
- Preserve deterministic behavior: selection of “best” positive is based only
  on current model similarities for samples already loaded in the deterministic
  batch; no extra stochastic pair sampling is introduced.

## Spec Update

- Add model-output layout specs:
  direction channels first, optional presence channel next, optional embedding
  channels after that; all consumers must use slicing helpers.
- Add presence-head training specs:
  positive target at transformed CP pixels, negative target at reachable valid
  non-CP pixels, edge-unreachable pixels ignored, balanced pos/neg weighting,
  and fixed-scale visualization.
- Update contrastive embedding specs to replace “all same-fiber CP positives”
  with “one best same-fiber other-CP/z-offset positive per anchor CP” when
  multiple z offsets are available.
- Clarify that the explicit presence head is the sheet/fiber presence signal;
  embedding training should focus on useful CP matching rather than dense
  presence classification.

## Docs Updates

- Update `docs/code_structure.md`:
  - describe the new model output slices and helper functions;
  - document where presence loss and embedding loss live;
  - document the z-selected positive-pair rule.
- Update `planning/changelog.md` after implementation with a short entry.
- Replace `planning/task_log.md` during implementation with only this task's
  notes, deviations, validation commands, and results.

## Tests

- Model/output tests:
  - model can emit direction + presence without embedding;
  - model can emit direction + presence + embedding;
  - slicing helpers return the expected channel ranges.
- Presence loss tests:
  - CP pixels are positive;
  - valid reachable non-CP pixels are negative;
  - unreachable edge pixels are ignored;
  - positive and negative losses are balanced rather than dominated by pixel
    count.
- Embedding z-selected positive tests:
  - for same-fiber CPs with multiple z offsets, only the most similar
    other-CP/offset pair contributes as the positive for each anchor;
  - same-CP different offsets are not treated as the required positive target;
  - anchors without another same-fiber CP do not create a positive term;
  - negative/sparsity terms still use the reachable mask.
- Integration tests:
  - `_compute_batch_loss` and `_compute_prepared_batch_loss` include presence
    and embedding metrics consistently;
  - TensorBoard visualization helpers can render presence maps.
- Validation commands:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/embedding.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Review Notes

- The main design choice to confirm during implementation is whether the old
  embedding similarity-image mean term should stay once the presence head is
  active. This plan keeps it to avoid silently removing an existing regularizer.
- If presence is enabled in the example config, the model output checkpoint
  layout changes and old checkpoints without the presence head will not load
  unless resume/loading is made tolerant. The implementation should either keep
  the example default disabled or document the checkpoint incompatibility
  clearly when enabling it.
