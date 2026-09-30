# Observation memory and causal survival confidence

## Token-only patch model

`scripts/launch_patch4_memory.sh` selects `axial_patch4_tokens_fiber_memory_v9`.
Its only spatial feature lattice is the 4x4x4 patch grid: 20,280 width-128 tokens
for the default crop. It has no fine reconstruction, output-plane feature tokens
or fine stencils. References, initial path queries, refinement feedback, segment
samples and memory details sample this coarse lattice using its physical stride
and offset. The generator and scorer attend to the deep grid and history only.
Both reuse compacted CUDA K/V across attempts. Historical encoder checkpointing
remains on by default and can be disabled independently of axial-block
checkpointing. Losses, memory chronology and continuous path coordinates retain
the contract below. Older v9 model configurations keep their original dense paths.

## Dense-model information flow

1. The main encoder extracts stem appearance features and an axial deep lattice
   from CT, presence, optional direction fields, and visible observed history.
   Persistent memory does not enter this encoding.
2. Memory stores 32 pooled deep tokens and up to 16 detail tokens containing
   3x3 stem neighborhoods and deep features at the head and observed history.
   There is no other image encoder. Stored observations precede memory conditioning.
3. Queries retrieve incoming historical observations. The coarse retrieval is
   projected/interpolated onto the deep lattice; the fine-feature decoder combines
   this conditioned lattice with stem features once per crop.
4. The generator reads the complete deep lattice and full-resolution dense
   features on every output plane, plus visible references, immutable seed
   observations, cached observations and persistent slots. Plane features retain
   every lateral pixel, are projected to the decoder width and receive physical
   XYZ position embeddings. Every query can attend to all planes in every layer;
   there is no per-query plane mask. The production crop supplies 163,216 fine
   tokens alongside 39,015 deep tokens. Four decoder layers jointly predict the
   future lateral coordinates. Up to two further passes through the same decoder
   receive the previous coordinates, resampled evidence, decoder state and detached
   conditional failure probabilities/prefix confidence. The same coordinate head
   predicts a complete absolute replacement path on every attempt; there is no
   offset head or correction-distance setting. Crop bounds and the first-connection
   limit apply after each pass. Generator self-attention remains bidirectional.
5. A separate scorer samples four ordered locations along each incoming proposed
   segment, including the endpoint, using the existing 3x3x3 fine-feature stencil
   and deep features. At production forward spacing this is one sample per quarter
   forward voxel, matching dense supervision. The fixed forward-plane representation
   determines resolution; arbitrary resampling of candidates is not supported.
6. Segment tokens include sampled evidence and start/end/displacement/length.
   Two scoring decoder layers use causal segment self-attention and unrestricted
   cross-attention to the deep image/reference/memory tokens and every lateral
   pixel on all output planes. The generator and scorer share plane sampling;
   the scorer learns independent channel and physical XYZ projections. It retains
   segment-local fine samples as well. Its own K/V projections are reused across
   generated and supplied paths within one decision. No generator hidden states
   enter scoring. Architecture `axial_fiber_memory_v9` requires newly trained
   weights; older checkpoints have no migration or compatibility path.
7. One linear readout predicts the conditional first-failure logit per segment.
   Prefix confidence is the product of conditional survival probabilities,
   accumulated in FP32 log space. Supplied candidates and generated paths use the
   same scorer. Existing diagnostic `confidence_logits` remain prefix logits;
   explicit `hazard_logits` are used for training.
8. Every proposal is scored before deciding whether to retry. Acceptance across
   the entire horizon ends that row's attempts immediately, even when the commit
   limit is shorter. Only unaccepted rows enter further decoder/scorer calls;
   encoder features and projected K/V are reused. The step setting is a maximum
   number of additional attempts, so two means between one and three proposals.
   Inference skips inactive rows; compiled training computes fixed slots and
   masks inactive results. `refinement_mask` identifies actual attempts; unused
   slots receive no loss or selection weight.

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
within a path; generated paths average over actual attempts, while supplied
candidates average separately per state. States receive fixed per-stream weights
before division by the update's supervised decision count: .75 for the endpoint,
.25 shared by up to two uniformly sampled historical positions. All observations
still enter memory; unsampled positions receive no task prediction. With no
auxiliary decisions the endpoint receives weight one. The history share and
auxiliary count are configurable. Each endpoint is predicted once using
reconstructed memory, with no additional replay coefficient or endpoint loss.
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
shared encoder and memory, but not the generator plane projections, decoder,
shared coordinate head or feedback fusion. Geometry cannot update the scorer
through detached feedback or discrete selection. The last attempted proposal gets
75% of the geometry weight and earlier attempts share 25%; a sole proposal gets
100%. All attempts are trained, including ones rejected by selection. Seed,
history and observation-memory tokens are available through cross-attention.

Matched choices reserve supervision for both recoverable alternatives, requiring
different geometry under identical local observations but different histories.
Departure pairs instead contrast stopping on a foreign tail with following that
same tail under its own history. Four supplied candidates include both original
continuations and two smooth switches at varied forecast positions; dense labels
locate their first failure. Candidate geometry/order is shared between paired
observations. Synthetic transitions never serve as geometry targets.

Inference selects the proposal with the longest acceptable prefix within the
commit limit, then greatest confidence at its last accepted point, then earliest
attempt. If none qualifies, the best valid first-point confidence is selected and
the commit gate still stops. A full-path acceptance ends further retries. Low
initial confidence therefore consumes the retry budget rather than ending retries;
there is no feedback warm-up schedule. Mean attempts used is logged for monitoring.
Confidence
is monotone by construction; it does not need post-hoc minimum repair. The shared
trace policy's existing cumulative minimum is harmless for these monotone values.

## Retention and gradients

Memory carries an immutable seed observation, the newest 64 observation grids,
and 16 persistent slots. Spatial tokens retain position, orientation, age and
validity and stay readable outside the current crop. Every observation is cached,
including wrong turns. Slots compress independent observations using a learned
update gate; retrieved historical interpretations are never written back.

Training collects detached visual observations across loader chunks. At each
sampled decision, it reconstructs every preceding writer transition with gradients
at current weights and re-encodes up to three past images, stratified by age.
Other cached features are detached and potentially stale. A later task loss
trains both the memory writer and selected past encodings; the current observation
is encoded and written by that decision's sole prediction. This is not full-history
encoder backpropagation. Selected CPU images are released after their final use;
completed streams are evicted. No decision reads future observations. The immutable
seed survives cache eviction, and persistent slots can retain older evidence.

See [training and validation commands](README.md).
