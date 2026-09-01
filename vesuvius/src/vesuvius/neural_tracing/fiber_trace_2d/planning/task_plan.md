# Plan: separate winding value and sign-hardness constraints

## Constraint semantics

1. Retain dominant perpendicular-versus-parallel selection and H/V orientation
   evidence, but materialize two independent winding evidence items from every
   admitted nonzero signed observation:
   - winding value: `abs(predicted_delta - canonical_signed_target)`;
   - sign: compatibility or finite loss from the signs of predicted delta and
     canonical target.
2. Keep zero-distance parallel observations magnitude-only. Preserve parallel
   distance cutoffs and same-source hard continuation behavior. A dominant
   parallel observation with unsigned distance but no reliable aligned sign
   still contributes the available unsigned winding target; only its extra
   sign-hardness item is absent.
3. Represent structural `magnitudePresent` and `signPresent` separately from
   effective solver coefficients. Presence follows extraction, dominance,
   target existence, cutoff, and sign-category configuration. Effective state
   additionally follows weights, confidence, finite cost, and hard promotion.
4. Add nonnegative perpendicular-sign and parallel-sign multipliers. A zero
   sign multiplier disables that sign, including alignment promotion to hard;
   positive weights scale finite sign loss independently of magnitude class
   multipliers. The exact finite coefficient is `finite_sign_cost *
   relation_sign_weight * decision_confidence * normal_confidence`. Hard signs
   remain exact after positive-weight enablement; their weight magnitude is
   otherwise irrelevant.
5. Sum signed winding-value and sign-hardness potentials inside the existing
   pairwise BP factor.
   Separate factor objects are unnecessary because both items use the same two
   variables, but their preparation, enablement, energy, diagnostics, and
   counts must remain distinct. A reversal infringes the ordinary winding-value
   term and also the separately weighted sign-hardness term.
6. Apply the signed winding-value rule in every solver path: initialization,
   alternating calibration updates, component-sign selection, adaptive and
   fixed joint-grid messages, decoded ranking, and projection. Degree-scaled
   Defect cost continues counting one physical measurement, not its two
   evidence items.

## Diagnostics and benchmark

1. Split final-solution agreement into orientation, magnitude class, and sign
   rows. Count each prepared item independently; a Defect endpoint neutralizes
   each applicable item independently.
   Active predicates are separately listed: orientation checks H/V,
   winding-value checks the signed canonical target, and sign checks ordering.
2. Split reference-to-BP accuracy into perpendicular magnitude,
   perpendicular sign, parallel-same magnitude, parallel-other magnitude, and
   parallel sign. Preserve per-reference and aggregate right/wrong totals.
3. Make reference correctness denominators independent of the candidate solver
   weights. Weight zero changes inference but not whether extracted reference
   evidence is checked. Structural cutoff and missing/zero signed targets still
   determine whether an item exists.
4. Use the same signed winding-value plus separate sign-hardness score for
   reference winding inference/calibration as BP. The signed target generates
   the BP-consistent candidate and the sign item independently records the
   additional finite/hard reversal judgment. Preserve deterministic half-step gauge
   calibration and the fixed reference-source denominator used for tuning.
   Correctness checks structural items even when their solver coefficient is
   zero; a structurally present item lacking a usable calibrated active result
   counts wrong rather than disappearing.
5. Extend CSV diagnostics so winding-value and sign weights/enabled state are
   explicit and cannot be confused with one combined coefficient.

## CLI and tuning

1. Keep `--winding-weights P05,PFAR,P0,P1,P2` for magnitude weights.
2. Add `--winding-sign-weights PERP,PARALLEL`, defaulting to `1,1`; document
   strict validation. Extend local search to seven reversible zero-aware
   coordinates, at most 21 neighbors, with cache/ranking identity and
   lexicographic ordering over magnitude then sign coordinates. Print and
   install both tuples unambiguously. Exhaustive search spans all seven
   coordinates and retains the existing 100,000-scenario bound.
3. Run the established 1024 local `/2`, `*2`, and zero search from the current
   defaults (`0,4,2,2,1` and `1,1`), first reporting the corrected baseline,
   then every accepted move. Promote selected defaults only after the corrected
   fixed-denominator benchmark. Do not compare the new doubled evidence-item
   aggregate directly to the former combined-factor percentage. State that
   positive hard-sign weights are equivalent and that positive weights tune
   only non-promoted signs under the default mixed finite/hard configuration.

## Tests

- Add focused energy tests proving a reversal infringes both the signed
  winding-value loss and the independently weighted sign-hardness loss.
- Test zero sign weight, finite weighted sign, hard-sign promotion, parallel
  cutoff, and zero-target behavior independently.
- Add agreement and reference benchmark tests proving one signed observation
  produces separate winding-value and sign counts and stable denominators when a
  solver weight is zero.
- Build Release `vc_fiber_trace_chunk` and the focused winding test binary; run
  focused tests, CTest, and `git diff --check`.
- Run one established 1024 tuning process and record command, inputs, build,
  accepted moves, timing, and selected result.
- Cover endpoint reversal, unsigned parallel fallback without sign,
  independent winding/sign listing, deterministic component sign without sign
  evidence, scale-before-separation, Defect incidence, fixed/adaptive and
  alternating paths, and seven-coordinate CLI/search validation.

## Spec update

Update `planning/specs.md` with signed winding-value semantics, separately
weighted sign evidence, factor equivalence, independent diagnostics, and
weight-independent benchmark denominators.

## Docs update

Update `volume-cartographer/docs/fiber_chunk_tracing.md` with the seven tuning
coordinates, CLI syntax, output rows, and measured 1024 result.

## Changelog

Record the correction that makes ordinary signed winding value and additional
sign hardness independently tunable and measurable.
