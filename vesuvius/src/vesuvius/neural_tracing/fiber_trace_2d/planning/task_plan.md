# Plan: hard split continuity and aligned winding signs

## Implementation

1. Extend the shared H/V-aware winding configuration with:
   - a hard split-continuity switch, enabled by default;
   - an optional minimum absolute Lasagna-normal alignment for promoting an
     enabled signed factor to hard, defaulting to `cos(30 degrees)`.
2. Apply edge-local hard split continuity in the H/V/Mixed orientation prepass
   and in both joint-grid and alternating winding solvers. Two active endpoints
   on one continuation edge must have the same H/V and winding state. Either
   endpoint may instead be Defect, which neutralizes that edge and splits the
   source into independent active runs. Enforce active segments in pair
   potentials and deterministic final decoding so independent node MAP values
   cannot publish an active-active mismatch. Preserve existing finite pair
   behavior when disabled.
3. Centralize sign-mode selection. An enabled nonzero dominant sign is hard
   when global sign cost is `hard`, or when its measured absolute normal
   raw absolute alignment reaches the configured threshold (inclusive), even
   when transformed decision confidence or finite sign cost is zero. Otherwise it retains the
   existing finite confidence-weighted sign penalty. Use this identical rule
   in solver preparation, factor diagnostics, and reference inference. Global
   `hard` overrides the gate; missing alignment cannot be promoted; parallel
   cutoff and dominant/nonzero/enabled admission still apply first.
4. Add `--split-continuity hard|finite` and
   `--winding-hard-sign-angle DEG|off`. Validate degrees in `[0,90]` and convert
   to `cos(DEG)`. Defaults implement hard split continuity and 30 degrees;
   `finite` plus `off` exactly recover the previous finite-only behavior.
5. Add a shared final-solution constraint-agreement summary over prepared
   dominant factor diagnostics after fixed-orientation removal, parallel
   cutoff, and confidence/sign admission. Classify each measurement as
   continuity, perpendicular 0.5/far, or parallel 0/1/far; report prepared,
   active/evaluated, Defect-neutralized, infringed, and
   `infringed/evaluated` percent (`NA` for zero evaluated). Count a measurement
   once as infringed when any of these fail: expected H/V relation; enabled
   sign (`target*predicted <= 0`); or canonical target bin. Perpendicular bins
   use `delta/measurement_scale` and half-integer nearest-bin boundaries;
   parallel bins use unscaled latent delta and integer nearest-bin boundaries.
   Continuity requires identical full state only when both endpoints are
   active; any Defect endpoint neutralizes the edge.
6. Print the compact table for every final winding solution before reference
   benchmark output.

## Tests

- Unit-test hard split continuation for allowed active/Defect and Defect/Defect
  pairs, forbidden active-active H/V and winding mismatches, and valid identical
  active states.
- Unit-test a three-piece active/Defect/active chain whose two active runs have
  different H/V and winding states, plus deterministic final enforcement of
  each uninterrupted active segment after inconsistent nodewise MAP states.
- Unit-test finite compatibility mode and existing piece-break cost.
- Unit-test perpendicular and parallel sign promotion above/equal/below the
  30-degree alignment threshold, missing alignment, disabled threshold, and
  globally hard signs.
- Unit-test zero decision confidence/cost promotion, zero targets/weights,
  cutoff-suppressed parallel signs, and endpoint reversal.
- Unit-test reference observations use the same promoted-hard rule.
- Unit-test constraint infringement grouping, exclusions, repeated prepared
  measurements, Defect neutralization, H/V mismatch, sign
  mismatch, target-bin boundaries, aggregate percentages, and zero denominator.
- Build the optimized winding test and CLI targets; run focused winding tests,
  relevant crop-constraint tests, and `git diff --check`.
- Run the approved 1024 and 2048 direction-ablation workloads as a four-way
  attribution matrix on each artifact using the optimized build:
  `finite/off`, `hard/off`, `finite/30 degrees`, and `hard/30 degrees`.
  Compare prepared/excluded
  factor totals, active/Defect pieces, infringements, reference metrics,
  convergence/residual, runtime, and generated visualization artifacts.

## Spec Update

Specify exact hard split-continuity state compatibility, the finite fallback,
alignment-gated hard sign semantics and defaults, and final-solution
constraint-infringement accounting.

## Docs Update

Document the two CLI controls, interaction with finite sign and piece-break
costs, alignment convention, and the new solution-agreement table.

## Changelog

Record hard split continuity, alignment-gated hard signs, and final constraint
infringement diagnostics.
