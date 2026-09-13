# Plan

1. Commit prior state. Reuse frozen EMA99 update6500 SHA0a2651b7 and existing hard bidirectional fiber cache; baseline EMA512/.25=69 exact hash documented.
2. Shared native bounded representative anchors: retain sampled unit embeddings in existing arclength history; deterministically choose K medoids with greedy facility-location coverage (minimize sum nearest cosine distance over all retained observations). Central first medoid; remaining anchors maximize coverage improvement, stable earliest ties. This rewards diversity only when it covers actual observations, not isolated farthest points. Stop if no improvement. Not claim globally optimal K medoids.
3. Score nearest selected medoid, optionally convexly combine with EMA512. Score parent before updates, independent state per beam/reset. Recompute only on new history sample or expiry; bounded history<=64 and anchors<=8, no full-path allocation. Keep default modes unchanged. Share cosine/history/EMA helpers, no duplicated tracer.
4. Test math on imbalanced clusters, diversity, bounds, ties, expiry, per-beam copies, missing/invalid and CLI parameters; native and Python suites through approved runner. Recheck off/original/EMA hashes.
5. Sweep K2/4/8, short bank32 samples spacing128/window4096 and long bank64 spacing256/window16384; anchor-only and half EMA mixture. Same .25 weight, manager nonexclusive20GiB/16CPU and cached inference extension. Repeat best exactly. Compare against69, full tables/timings. Refine useful anchor mixture/weight if promising; otherwise inspect promising EMA half-life neighbors256/384/640/768/1024 as an additional approach. Continue sensible measured refinements until improvement stabilized or no supported next acceleration/quality idea remains.

## Spec update
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
Post-grid refinement requested: optimize each reference family's best seed,
including nonwinning families. Use deterministic cached coordinate searches on
weight and applicable capacity/admission/EMA/history axes, cycle until no strict
improvement within tested axes, recheck neighboring weights and repeat winners.
Record searched ranges and distinguish tested local optimum from global claims.

Independent review. Approved experiment/test runners only. CPU native correctness tests then manager GPU extension. Exact baseline and best repeat. Same whole fiber both directions; report local tuning not generalization. No installs/dependencies.
