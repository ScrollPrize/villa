# Plan: Fiberlet crop lookahead graph halo

## Contract

- Keep `--bbox` as the only public crop: it defines canonical seed ownership,
  anisotropic coverage candidates, first-exit clipping, trace artifact
  metadata, CT faces, and all downstream output.
- Derive an internal graph search box by expanding every requested-crop face
  by exactly `lookaheadDistanceBaseVoxels`.
- Materialize graph anchors, complete routes, costs, and transitions against
  the expanded search box, but return seed anchors only from the requested
  half-open crop.
- Do not add maximum Fiberlet length to the padding. The materializer includes
  every complete Fiberlet incident to an anchor inside the search box, so the
  final horizon-crossing edge and its outside endpoint are already available.
- Preserve sparse-dataset semantics: missing tuples are empty and required
  partial tuples fail.
- Use the expanded search box for speculative lookahead route clipping and
  graph-exhaustion decisions. Keep committed `traceSide` clipping at the
  requested crop.
- Preserve arithmetic, deterministic ordering, output clipping, seed and
  coverage semantics. Near-boundary selected routes may intentionally change;
  routes whose lookahead does not reach the old crop boundary must not change.
- Do not enable staged filtering. Record it as a follow-up experiment.

## Implementation

1. Add a shared crop-search-box helper that validates the crop/lookahead and
   returns the exact lookahead-expanded base-XYZ box. Refactor route clipping
   to accept explicit bounds.
2. Extend stored graph bulk materialization to accept separate graph and seed
   boxes. Require a finite nonempty seed box contained in the graph box; keep
   the existing overload as equal graph/seed bounds for other callers.
3. Make speculative lookahead use the expanded search box, while committed
   tracing, artifact writing, and visualization retain the requested crop.
4. Make `vc_fiber_trace_chunk trace` materialize the expanded graph with the
   requested crop as its seed box.
5. Report requested and search bounds plus padding in graph-preparation output.

## Spec Update

Update anchor-seeded crop tracing so immutable graph materialization and
speculative rollout use a lookahead halo while committed seed/output semantics
remain tied to the public crop.

## Docs Updates

Document the distinct requested crop and internal search graph, explain why
one lookahead distance is sufficient, and state the graph-preparation cost.

## Testing

- Unit-test exact search-box expansion and invalid configurations.
- Extend stored graph tests to prove an expanded graph can expose continuation
  edges while returning only requested-crop seeds, and reject a seed box not
  contained by the graph box.
- Add a horizon-crossing edge longer than the remaining halo to prove exact
  lookahead padding is sufficient.
- Add an end-to-end boundary choice where evidence beyond the requested crop
  changes the winning first edge, while committed output remains clipped at
  the requested crop.
- Cover sparse halo behavior: absent tuples are empty, required partial tuples
  fail, unrelated partial owner-halo tuples remain ignored, and tuples outside
  the search box do not affect the crop.
- Retain crop-tracing tests proving first-exit clipping at the requested crop
  and unchanged results away from its boundary.
- Build `vc_fiber_trace_chunk`, `vc_fiberlets`, `test_fiberlet_storage`, and
  `test_fiberlet_crop_trace` with `-j32`; run both focused suites and
  `git diff --check`.

## Changelog

Record lookahead-expanded crop graph materialization.
