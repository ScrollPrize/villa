# Observation memory and causal survival confidence

## Information flow

1. The main encoder extracts stem appearance features and an axial deep lattice
   from CT, presence, optional direction fields, and visible observed history.
   Persistent memory does not enter this encoding.
2. Memory stores 32 pooled deep tokens and up to 16 detail tokens containing
   3x3 stem neighborhoods and deep features at the head and observed history.
   There is no other image encoder. Stored observations precede memory conditioning.
3. Queries retrieve incoming historical observations. The coarse retrieval is
   projected/interpolated onto the deep lattice; the fine-feature decoder combines
   this conditioned lattice with stem features once per crop.
4. The generator reads current spatial features, visible references, immutable
   seed observations, cached observations and persistent slots. Four decoder
   layers jointly predict the future lateral coordinates. By default, one more
   pass through the same decoder samples the proposal and applies a bounded
   one-voxel lateral correction. Generator self-attention remains bidirectional.
5. A separate scorer samples four ordered locations along each incoming proposed
   segment, including the endpoint, using the existing 3x3x3 fine-feature stencil
   and deep features. At production forward spacing this is one sample per quarter
   forward voxel, matching dense supervision. The fixed forward-plane representation
   determines resolution; arbitrary resampling of candidates is not supported.
6. Segment tokens include sampled evidence and start/end/displacement/length.
   Two scoring decoder layers use causal segment self-attention and unrestricted
   cross-attention to the entire current image/reference/memory token set. Their own K/V projections are reused across
   scored candidates within one decision. No generator hidden states enter scoring.
7. One linear readout predicts the conditional first-failure logit per segment.
   Prefix confidence is the product of conditional survival probabilities,
   accumulated in FP32 log space. Supplied candidates and generated paths use the
   same scorer. Existing diagnostic `confidence_logits` remain prefix logits;
   explicit `hazard_logits` are used for training.

Holding observations and a prefix fixed, replacing its suffix leaves earlier
scores unchanged. Truncating/extending it also preserves those scores, up to
floating-point kernel rounding. Image lookahead remains unrestricted: the causal
mask applies only to proposed segments, never to the observed scene.

## Supervision and decisions

For conditional failure probabilities `h[i]`, prefix survival is
`S[k] = product(1 - h[i], i <= k)`. If the first known failure is in interval `j`,
the negative log likelihood is `-sum(log(1-h[i]), i<j) - log(h[j])`.
A fully correct or right-censored path contributes survival terms only through its
last known correct prefix. Hazards after a first failure or an unknown prefix
receive no supervision. A failure observed after an annotation gap does not locate
the first-failure interval; training conservatively censors at the gap.

Both generated and supplied candidate paths use this likelihood. Intervals sum
within a path; paths average per state and states average per effective batch.
Commit-window weighting applies to geometry only. This changes confidence-loss
scale from the former prefix BCE. Raw `confidence_count` remains a count of known
prefix labels for correctness metrics, not a count of hazard training targets.

Dense labels retain their existing recovery contract: the first event judges the
first point and admissibility of the origin-to-first-point connection. A displaced
origin may recover within the configured limit; that bridge is not required to
match the original centerline. Later events judge all dense crossings since the
previous point. Known departures and physical endpoints are failures; unknown
ends and unobservable identity are censored. Foreign-fiber masks remain negatives.

Confidence detaches scored coordinates. Its loss trains its own scorer, the
shared encoder and memory, but not the generator decoder, coordinate head or
refinement head. Geometry supervises initial/final proposals with weights
25%/75%. Seed, history and observation-memory tokens are available through
cross-attention.

Inference commits the longest prefix clearing its confidence threshold. Confidence
is monotone by construction; it does not need post-hoc minimum repair. The shared
trace policy's existing cumulative minimum is harmless for these monotone values.

## Retention and gradients

Memory carries an immutable seed observation, the newest 64 observation grids,
and 16 persistent slots. Spatial tokens retain position, orientation, age and
validity and stay readable outside the current crop. Every observation is cached,
including wrong turns. Slots compress independent observations using a learned
update gate; retrieved historical interpretations are never written back.

Training carries memory through two-decision gradient chunks. Endpoint replay
recomputes writer transitions chronologically and re-encodes up to three selected
historical crops, stratified by age. Other historical features are detached and
potentially stale. This is not full-history encoder backpropagation. The immutable
seed survives cache eviction, and persistent slots can retain older evidence.

See [training and validation commands](README.md).
