# Plan

1. Commit prior state. Reuse frozen EMA99 update6500 SHA0a2651b7 and existing hard bidirectional fiber cache; baseline EMA512/.25=69 exact hash documented.
2. Shared native bounded representative anchors: retain sampled unit embeddings in existing arclength history; deterministically choose K medoids with greedy facility-location coverage (minimize sum nearest cosine distance over all retained observations). Central first medoid; remaining anchors maximize coverage improvement, stable earliest ties. This rewards diversity only when it covers actual observations, not isolated farthest points. Stop if no improvement. Not claim globally optimal K medoids.
3. Score nearest selected medoid, optionally convexly combine with EMA512. Score parent before updates, independent state per beam/reset. Recompute only on new history sample or expiry; bounded history<=64 and anchors<=8, no full-path allocation. Keep default modes unchanged. Share cosine/history/EMA helpers, no duplicated tracer.
4. Test math on imbalanced clusters, diversity, bounds, ties, expiry, per-beam copies, missing/invalid and CLI parameters; native and Python suites through approved runner. Recheck off/original/EMA hashes.
5. Sweep K2/4/8, short bank32 samples spacing128/window4096 and long bank64 spacing256/window16384; anchor-only and half EMA mixture. Same .25 weight, manager nonexclusive20GiB/16CPU and cached inference extension. Repeat best exactly. Compare against69, full tables/timings. Refine useful anchor mixture/weight if promising; otherwise inspect promising EMA half-life neighbors256/384/640/768/1024 as an additional approach. Continue sensible measured refinements until improvement stabilized or no supported next acceleration/quality idea remains.

## Spec update
Reviewed aggregation hypothesis: opt-in normalized-mean LRU. Ordinary mean
cost is `1-r*cos(query,mean_direction)`, r=norm(sum anchors)/K; disagreement
shrinks influence toward1. Normalized mean removes that shrinkage, without
changing admission/eviction. Cache derived unit sum only when bank changes,
reuse shared norm/cosine math, exact-zero sum costs1. Preserve default arithmetic.
Measure coherence r (including small-r frequency) before claiming this helps;
near-cancellation can amplify unstable directions. Start at existing64/68
settings, refine if useful. This differs from EMA and whole-path centroids.

Reviewed next replacement-policy A/B: opt-in LRU `hit` touch policy. Existing
`nearest` policy refreshes even an anchor outside the novelty threshold, then
evicts another anchor. `hit` refreshes only exact/familiar observations; a novel
observation evicts actual LRU without touching an old anchor. Same threshold,
cadence, capacity and scoring, no new state. Default nearest remains unchanged.
Test deterministic eviction, familiar touches, threshold equality and default
equivalence. Compare at the64-restart settings, then refine if promising.

Independent methodology review adds a fixed-seed-only control (not EMA+seed):
many sparse-anchor segments have no post-seed observation, so isolate whether
adaptation itself helps before attributing results to the memory policy.
Opt-in seed mode uses the initial embedding unchanged throughout each trace,
reset at each new seed; no history updates or expiration. Include in the delayed
EMA sweep and refine its weight. If results justify it, next investigate local
embedding-independent directional ambiguity as an admission gate; measure the
margin distribution before inventing thresholds. Do not claim contamination
from current single-fiber evidence.

Reviewed conditional follow-up after seed/delay: whole-path observed-reference
scoring instead of synthetic centroid scoring. Retain the identical sums,
supports, merge criterion and admission; keep one observed exemplar per cluster,
choose the nearer of the two prior exemplars to the merged sum direction, stable
ties/zero-sum retain earlier. This is a hierarchical observed representative, NOT
an exact medoid (discarded examples cannot be recovered). First inspect
centroid-to-exemplar cosine; if nearly1, a broad sweep is not justified. Otherwise
test as a score-only A/B then refine if promising. No claimed guaranteed benefit.

Another reviewed conditional mechanism: embedding-cost reliability weighting by
selected presence^q (q0 exact existing arithmetic; initial.5/1/2), leaving other
costs/reference updates unchanged. Off-fiber training supervision is presence-
weighted, but GT positives are not, so this is a hypothesis not a required fix.
It can accidentally make background cheaper: measure selected-path presence
alongside restarts before accepting it. Current single-branch semantics match
training max presence; do not silently generalize to multi-branch models.

Additional hypothesis queued after family refinements: opt-in delayed EMA
updates. Dense novelty-LRU admissions8/32 worsened71 to76/84, which is consistent
with (not proof of) appearance contamination after a wrong turn. Keep per-beam
pending observations and apply only after a configured arclength delay, seed
active immediately, score parent state. EMA decay uses applied observation
distance (stored directly to preserve arithmetic); zero delay must reproduce ordinary EMA. Bound pending memory at4096 observations per beam and
error on overflow, no silent dropping. Preflight uses clamped float step and
positive base scale with conservative half-step capacity estimate; retain runtime
check because float coordinate precision is not universally bounded. Default
delay0, maximum2048, initial trials32/64/128/256; use empty-vector storage to
avoid adding allocations in other modes. Review/test boundary, zero-delay,
contamination lag, copy/reset, capacity; same frozen benchmark then refine.

Algorithmic refinements after initial grids (independently reviewed):
- LRU novelty gate: touch closest at observation cadence, but insert only if its
  cosine distance is at least novelty threshold. Threshold0 preserves prior LRU.
- Whole-path support-softmin: normalized cluster support weights and temperature
  combine costs as `min(c)-T*log(sum p*exp(-(c-min(c))/T))`. This requires stronger
  support than hard nearest matching. Stable expm1/log1p at large temperatures.
  Retain hard-min default; test normalization, one-cluster and split invariance.
Sweep these mechanisms separately, then refine each family's own best settings.

After LRU: replace the restricted rolling-bank appearance model with whole-path
bounded representative/diverse anchors. No distance expiry. Retain accumulated
support when compressing observations; explicitly test that old supported
appearances survive long distances. Review the streaming approximation before
implementation, then benchmark against the same frozen controls.
Reviewed implementation design: keep up to K spherical cluster summaries,
each an FP64 vector sum plus integer support. Append each spaced observation;
when K+1, merge the pair minimizing norm(a)+norm(b)-norm(a+b), with stable
index ties. Retain true sums (never normalized between updates); normalized
centroids score candidates, previous representative handles exact cancellation.
This minimizes incremental spherical reconstruction error approximately,
preserving whole-path mass rather than expiring old observations. It is not
exact global clustering and may merge rare modes. Start with nearest centroid
cost, optionally EMA blend; no implicit support-weighted scoring. Tests cover
support conservation, old modes, cancellation, K1, bounds and beam copies.

Next after current refinement: add opt-in anchors-lru mode. Each accepted
observation at configured spacing marks nearest anchor used (stable ties), then
inserts current point and evicts least recently used if full. Fixed anchor
vectors; duplicates may be skipped when exactly identical. Seed initializes
state, all private per beam/reset. Closest or average cosine costs, optional EMA
mixture, same weight. Sweep capacity4/8, admission spacing128/512 and nearest,
average, nearest+halfEMA (12 combinations). Validate LRU eviction/use/admission
math and default trajectory controls, repeat best. If promising refine weights.

Opt-in anchors mode, deterministic greedy coverage selection, bounded bank, EMA mixing, per-beam updates. Original defaults unchanged.

## Docs updates
New multi-anchor report, embedding_tracing options, task log/status/changelog including failures and limits.

## Validation
Prepared normalized aggregation A/B in `embedding_normalized_mean_next.json`:
EMA69 control, ordinary vs normalized mean at confirmed64 and68 settings, plus
hit-only66 settings. Require old hashes and nonzero candidate-parent coherence
counts on optimized production path; sum histogram equals count. Summarize r
and fraction below.1 (unstable-direction risk), with exact-repeat winner. Build
only after active sweep ends. This short comparison precedes the prepared wider
anchor boundary sweep; then refine normalized aggregation if useful.

Prepared boundary/multi-start sweep in `embedding_anchor_boundaries_next.json`:
After normalized A/B scored75 at each of the three seeds (different traces),
include those three families for their own finite refinement too: weights
.05/.10/.25/.50 and .05 neighbors, spacing32/128/512/1024/2048, K1/2/4/8,
novelty0/.1/.2. A negative A/B at borrowed settings is not a tuned-method result.
all anchor policies start from their own completed-round winner and, where
different, K2/weight.10 transferred seed. Extend spacing through2048 for
no-expiry methods (unused window4096), through1024 for windowed anchors with
their actual window1024; include K1. Keep earlier records as exact references.
This config is now running as `embedding_ema99_anchor_boundaries_20260914_v1`
(handle72112), after completed seed/delay/hit and normalized-mean A/B rounds.
These remain separate from the anchor-policy comparisons.

## Completion audit

Independent review after normalized A/B confirms implemented whole-path
sum/support appearance summaries (no age expiry), LRU semantics, per-family
refinement and algorithmic variants cover the requested methods. Finish the
active finite round and its exact repeats, consolidate best parameters/bounds/
hashes, verify released resources and keep training stopped. Treat this fiber as
tuning data, not independent validation. Observed-exemplar and reliability
weighting remain conditional hypotheses without supporting diagnostics; do not
silently claim they were executed or manufacture endless additional grids.

Prepared next mechanism grid in `embedding_methods_next.json`; after current
sweep exits and native tests pass, use it as the existing experiment control's
`embedding_trace_sweep` section. Same frozen pilot and manager allocation.
Require EMA control/zero-delay hashes identical, replay current64 setting after
rebuild, then compare seed-only, delayedEMA and hit-only touch. Refine each new
method after this initial grid. File is a prepared configuration, not a claim
that those experiments ran.

After the single-start family round, test transfer of the winning anchor
settings into the other anchor policies (preserving each policy's semantics),
then refine newly improved seeds. The plainLRU K2/.10 gain appeared only on a
later coordinate cycle: a novel/mean/appearance family can miss the same
capacity×weight interaction if neither single-coordinate move initially helps.
Compare each transferred seed against that family's existing best; this is
multi-start refinement, not a claim that coordinate minima are global optima.
Also refine admission distance around/above512 (192,256,384,512,768,1024,
1536,2048) and include capacity1 for the winning LRU policies; initial axes
stopped at512 and K2. For LRU, set unused history window>=largest spacing to
pass shared validation, with an equivalent-scoring control. Source level1 and
tile256 mean512base is one full input tile width (not blend stride); this is
scale context, not evidence that tile boundaries cause the optimum. One-anchor
LRU isolates periodic reference replacement versus genuinely multi-anchor use.

Post-grid refinement requested: optimize each reference family's best seed,
including nonwinning families. Prioritize anchor admission8/32/64 after actual
replay analysis found61/74 segments had no post-seed observation at512 spacing
(median144). Use deterministic cached coordinate searches on
weight and applicable capacity/admission/EMA/history axes, cycle until no strict
improvement within tested axes, recheck neighboring weights and repeat winners.
Record searched ranges and distinguish tested local optimum from global claims.

Independent review. Approved experiment/test runners only. CPU native correctness tests then manager GPU extension. Exact baseline and best repeat. Same whole fiber both directions; report local tuning not generalization. No installs/dependencies.
