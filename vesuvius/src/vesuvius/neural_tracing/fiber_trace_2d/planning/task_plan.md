# Plan: Full orientation constraints in fiber BP

## Contract

- Permit BP with or without `--perpendicular-only`. With the flag, retain the
  current perpendicular-only experiment exactly; without it, use all selected
  finite complementary parallel/perpendicular measurements.
- Continue rejecting hard-continuity links because BP requires one unsplit
  piece per represented fiber.
- Preserve factor energies `E_same=sum(1-p)` and `E_different=sum(p)` for
  parallel score `p`, then subtract `min(E_same,E_different)` from both
  oriented energies before inference. The remaining evidence strength is
  `abs(E_same-E_different)`, so scores near 0.5 are weak and scores near either
  endpoint are decisive. This normalization is immaterial to binary BP but is
  required in ternary BP so an arbitrary common factor offset does not favor
  zero-energy Mixed. Opposing measurements may partially or fully cancel.
- Omit exactly neutral merged factors (`E_same==E_different`) from the effective
  graph, degree, component, and mismatch accounting. Report neutral merged
  factor and raw-measurement counts separately from effective counts. Near
  ties retain their exact sign and strength; use no epsilon.
- Make diagnostics relation-aware: a resolved factor is mismatched when the
  observed equal/different assignment disagrees with the factor's lower-energy
  relation. Map neighbor H/V support through that relation.
- Replace the perpendicular-specific `softSameLabelProxy` terminology with a
  `softMismatchProxy`. For binary inference it is the independent-endpoint
  probability of violating the preferred relation. For ternary inference it
  uses explicit V/Mixed/H marginals: only oriented endpoint pairs contribute
  violation probability, and every pair involving Mixed contributes zero.
- Map ternary neighbor support using explicit neighbor V/H probabilities and
  omit Mixed mass. Use `abs(P(H)-P(V))` for ternary neighbor certainty. Keep the
  established scalar formulas for binary/min-sum inference.
- Rename CLI cohort reporting from `perpendicular_constraints` to
  `selected_constraints`.

## Implementation

1. Remove the CLI requirement that BP use `--perpendicular-only`.
2. Relax graph validation from strictly perpendicular evidence to any finite
   complementary score in `[0,1]`.
3. Normalize merged factor energies to the preferred oriented relation, drop
   exact neutral factors, and report neutral factor/measurement counts.
4. Generalize hard mismatch, soft mismatch, and neighbor-support calculations
   to each merged factor's preferred relation, using explicit ternary
   probabilities for Mixed-state reports.
5. Rename public report/CSV fields and documentation for the generalized soft
   mismatch proxy; retain no compatibility alias because this experiment is
   unshipped.

## Spec Update

Specify full-score factor acceptance, implicit decisiveness weighting, merged
evidence cancellation, relation-aware mismatch, and optional
`--perpendicular-only` filtering.

## Docs Updates

Update `volume-cartographer/docs/fiber_chunk_tracing.md` with the full-graph
invocation and generalized diagnostics.

## Testing

- Compare binary and ternary BP against brute-force exact marginals on trees
  containing both parallel- and perpendicular-preferring factors.
- Verify equal labels satisfy parallel factors and different labels satisfy
  perpendicular factors in hard and soft diagnostics.
- Verify neighbor support is mapped correctly for both relations.
- Verify `p=0.5` and exactly opposing merged evidence are omitted as neutral,
  do not connect components, and cannot favor Mixed; verify near-ties remain.
- Verify non-complementary scores and hard continuity still fail.
- Verify `--perpendicular-only` remains a valid filtering option while BP no
  longer requires it.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with `-j32`, run
  the focused suite, run the 1024 full-constraint BP command, and run
  `git diff --check`.

## Changelog

Record full parallel/perpendicular BP support and relation-aware diagnostics.
