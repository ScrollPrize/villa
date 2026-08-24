# Task log: two-stage regional Fiberlet graph reduction

## Initial findings

- The current 256-base sample retains 5,651/9,021 incident Fiberlets (37.36%
  reduction) but only 562/3,757 internal Fiberlets (85.04% reduction).
- Boundary-crossing Fiberlets dominate the all-Fiberlet population, so the
  internal count is the meaningful pruning signal.
- Stage one must union retained physical IDs across adjacent boxes before
  writing a derived dataset; independently filtering owner chunks would remove
  boundary routes required by neighboring boxes.
- For a 512-base region and 256-base boxes, the half-offset stage-two grid has
  exactly one centered box.
- Independent plan review required regional aggregation by canonical unique
  physical IDs, exact stage-two owner coverage, complete scoring metadata, and
  explicit handling of incomplete prefix/route pairs. The implementation plan
  was corrected before storage or CLI implementation continued.
- Box-local first-exit optimality is not globally compositional. The derived
  cache remains a diagnostic experiment and is not a default replay graph.
- The initial plan incorrectly tied derived-dataset coverage and identity to a
  requested region. User review corrected this: the reduced cache is a global
  reusable per-chunk dataset, while a target region only selects intersecting
  chunks and drives missing-chunk generation.

## Deviations and validation

- The first end-to-end attempt exposed two integration defects that unit-only
  verification had missed: the CLI rejected explicit cache overrides, and the
  reduced 256-base Fiberlet grid was paired directly with the original
  128-base anchor grid. The parser now accepts both cache roots, and stage two
  uses an on-demand rechunked anchor view backed by the original cache.
- The first corrected handoff still allowed the centered graph's conservative
  halo to generate reduced owners outside the selected eight. That run was
  stopped. Stage two now uses a bounded, non-publishing view: selected owners
  read the global reduced cache and all other owners resolve to ephemeral empty
  chunks.
- Focused storage validation passes: `test_fiberlet_storage` reports 27 test
  cases passed, and `git diff --check` is clean.
- The final hot Paris4 command used the existing 128-base source caches and the
  global 256-base reduced cache. Source preparation completed in under one
  second, stage one reported `generated=0 reused=8`, and stage two completed in
  1.4 seconds. Unique populations were 13,750 original, 7,112 after stage one,
  and 4,168 after stage two (69.69% total reduction). Internal populations were
  5,730, 3,436, and 618 respectively (89.21% total reduction and 82.01%
  incremental stage-two reduction).
- An automated 2x2x2 CLI-level regional fixture remains to be added. The real
  Paris4 end-to-end run currently covers the full command integration.
