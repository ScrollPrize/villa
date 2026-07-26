# 3D Direction-Conditioned Recurrent Decoder Plan

## Current State

- `fiber_trace_3d.model.FiberTrace3DNet` wraps `Vesuvius3dUnetModel` and emits
  `7 * direction_branch_count` channels directly.
- The current multi-direction training objective routes each positive sparse
  supervision point to one free branch using detached
  `abs(dot(decoded_pred, gt)) * predicted_presence`.
- That branch routing does not give either branch an angular identity, so the
  model can split by spatial/appearance modes instead of local fiber angle.
- Lasagna 3x2 direction encoding lives in `fiber_trace_3d.direction`. The
  all-zero six-channel vector is off the valid encoding manifold; it can be
  reserved as an unconditioned query token, but must not be decoded as a real
  direction.

## Model Design

- Replace the free multi-branch head path with an opt-in
  direction-conditioned recurrent mode, configured under `model_3d`.
- Keep a shared spatial 3D U-Net as the only component with 3D spatial
  convolution.
- In conditioned mode, build the U-Net with output channel count equal to
  `model_3d.conditioned_latent_channels`, default `64`.
- Keep `model_3d.unet_base_channels` / derived `features_per_stage` as the U-Net
  starting width; do not tie it to the latent width.
- Add a pointwise conditioned decoder head:
  - input channels: `conditioned_latent_channels + 6`;
  - output channels: `7`;
  - implementation: small stack of `1x1x1` convolutions or equivalent per-voxel
    MLP layers only;
  - final activation should match current output semantics unless explicitly
    changed: sigmoid direction/probability channels, BCE on probabilities.
- Provide model helpers that keep old call sites readable:
  - conditioned forward for a tensor of query encodings;
  - unconditioned helper using all-zero query;
  - recurrent helper that feeds an encoded previous prediction into the next
    query.
- Decide whether to preserve legacy `direction_outputs(...)` /
  `presence_outputs(...)` for old branch checkpoints or add separate
  conditioned-output helpers. Prefer preserving old helpers for grouped-output
  compatibility and adding new helper names for query outputs.

## Query And Target Construction

- Extend materialized target metadata, or add a new query-batch construction
  layer in the trainer, so positive sparse supervision points can create
  multiple query samples per point without materializing dense direction
  targets.
- Positive query samples:
  - zero/unconditioned query -> GT direction, positive presence;
  - sampled perpendicular query with jitter -> same GT direction, positive
    presence;
  - both positive query groups have equal configured weight.
- Perpendicular query generation:
  - operate in 3D tangent space using transformed GT tangent in XYZ/ZYX with
    explicit axis conversion;
  - sample one vector from the plane perpendicular to the GT direction;
  - rotate/jitter it by a configured angle range, default `45` degrees;
  - encode with `encode_lasagna_direction_3x2(...)`;
  - deterministic randomness must be keyed by `stream_index` and point/sample
    identity, not by loader order.
- Negative query samples:
  - zero query and random encoded direction query over all valid presence-loss
    locations, using the existing presence margin semantics;
  - include positive locations too by design; weighted BCE makes this a
    controlled softening of the positive target according to the explicit
    positive/negative weights.
- Keep direction supervision positive-only. Negative query samples supervise
  presence only.

## Loss Design

- Replace branch selection loss in conditioned mode with query-sample loss.
- Direction loss:
  - gather conditioned decoder direction predictions at positive query samples;
  - use the existing weighted MSE on six Lasagna 3x2 channels with
    `projection_magnitude_weights_3x2(...)`;
  - average/normalize so zero-query and perpendicular-query positives contribute
    equally.
- Presence loss:
  - use weighted BCE on sigmoid probabilities unless logits are introduced
    deliberately;
  - normalize from explicit sample weights so the intended ratio of positive and
    dense weak negative terms is mathematically preserved;
  - log the effective positive target implied when a positive voxel also
    receives weak negative terms, if useful for debugging.
- Keep existing non-conditioned branch loss available for old configs until the
  conditioned mode is stable.

## Training And Config

- Add config keys, with tentative names:
  - `model_3d.conditioned_decoder_enabled`, default `false` initially for
    checkpoint compatibility unless the active experiment config explicitly
    enables it;
  - `model_3d.conditioned_latent_channels`, default `64`;
  - `model_3d.conditioned_decoder_hidden_channels`, default `64`;
  - `model_3d.conditioned_decoder_layers`, default `3`;
  - `training.conditioned_perpendicular_jitter_degrees`, default `45.0`;
  - `training.conditioned_positive_query_weight`, default `1.0`;
  - `training.conditioned_negative_query_weight`, default chosen to preserve
    the current positive/global-negative balance.
- Update the active 3D training config to enable conditioned mode and remove
  `direction_branch_count: 2` / `output_channels: 14` from that path, replacing
  them with latent/head config.
- Resume compatibility:
  - old free-branch checkpoints are not architecturally compatible with the new
    conditioned head;
  - fail clearly if a conditioned config tries to load an incompatible branch
    checkpoint without an explicit partial-load path.

## Inference And Visualization

- Native 3D Trace2CP must use the conditioned model contract:
  - decode zero query for the initial strongest direction;
  - for recurrent/secondary inspection, encode the first decoded direction and
    decode again;
  - never decode the all-zero query as if it were a real previous direction.
- TensorBoard training/test visualization should show conditioned outputs
  rather than free branches:
  - zero-query presence/direction;
  - perpendicular/query-conditioned presence/direction for a deterministic
    query used in the sample visualization;
  - optional recurrent second-pass presence/direction using the zero-query
    decoded direction as the next query.
- Keep the existing principal and oblique slice rows. Update column titles and
  helper names so they no longer imply branch 0/branch 1 when conditioned mode
  is enabled.

## Spec Update

- Replace the current 3D multi-branch spec text with conditioned-decoder
  semantics for the new experiment mode.
- Explicitly state that the all-zero six-channel query is a reserved
  unconditioned token, not a decodable direction.
- State that geometric/augmentation randomness for query construction is
  deterministic by `stream_index`.
- State that the conditioned decoder head is pointwise only and must not use
  spatial 3D layers.
- Document loss normalization so weak dense negatives at positive pixels are an
  intentional weighted-BCE soft target, not an accidental contradiction.

## Docs Updates

- Update `docs/code_structure.md`:
  - model section: shared U-Net latent producer plus pointwise conditioned head;
  - train section: query construction and loss groups;
  - Trace2CP section: zero-query first pass and recurrent conditioned pass.
- Update `planning/changelog.md` once implemented.

## Tests

- Unit tests for encoding/query semantics:
  - all-zero query is accepted as the unconditioned token and not decoded for
    query interpretation;
  - sampled perpendicular query is approximately perpendicular before jitter
    and remains deterministic for fixed stream/point ids.
- Model tests:
  - conditioned model outputs shape `B,7,D,H,W` for one query volume;
  - latent width defaults to `64` independently of `unet_base_channels`;
  - conditioned decoder parameters are all pointwise `1x1x1` layers or linear
    equivalents, with no spatial 3D kernels.
- Loss tests:
  - zero-query and perpendicular-query positives contribute equal direction and
    positive-presence weight;
  - weak dense negatives at positive pixels produce the expected weighted-BCE
    soft-target optimum/equivalent target ratio;
  - negative query samples supervise presence but not direction.
- Integration tests:
  - a tiny synthetic batch runs forward/backward in conditioned mode and gives
    gradients to U-Net and conditioned head parameters;
  - legacy branch-mode tests still pass for old configs;
  - active conditioned config builds and performs one smoke training step on a
    small fixture.
- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/model.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/train.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/targets.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check`.

## Review Notes / Assumptions

- This plan intentionally does not implement direct paired crossing-fiber
  supervision because the available GT is single-direction at a supervised
  point.
- The conditioned decoder is expected to learn the secondary-choice behavior
  from the query contract and negative/positive query sampling, not from
  explicit overlapping multi-direction labels.
- The independent-agent review step from `AGENTS.md` requires explicit user
  authorization for delegation with the available tools. I will do local
  plan/spec consistency review unless delegation is requested.
