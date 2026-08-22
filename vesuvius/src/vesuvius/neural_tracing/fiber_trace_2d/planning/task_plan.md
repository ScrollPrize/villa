# Plan: compact float-position fiberlet default

1. Add shared constructors for the accepted default evaluation profile and the
   exact-float oracle. Avoid relying on ambiguous aggregate defaults in
   correctness comparisons.
2. Make a default-constructed evaluation profile select exact float positions,
   compact fitted directions, fixed sqrt-density `uint16` edge costs, and a
   density ceiling of 256. Keep the persistent `CompactQuantized` codec and its
   integer position-quantum schema unchanged.
3. Apply the accepted profile to normal cache-backed `fiberlet-replay`: use its
   compact-direction geometry identity during preprocessing, pass the same
   logical cost view to graph replay, and continue sharing canonical float
   anchors. Leave the explicit eager diagnostic path exact-float.
4. Change the quantization benchmark's default selected scenario to the
   accepted profile. Run its baseline through the explicit exact-float oracle.
5. Add regression coverage for both named profiles, default construction,
   geometry-cache selection, and normal replay wiring. Retain the completed
   fractional-position tests and matrix coverage.
6. Build with `-j32`; run focused storage, cached replay, and path tests. Record
   the completed full-fiber radius-768 comparisons for both q1/8 and the new
   default profile.
7. Commit only the intended tracked source, tests, planning, specification,
   changelog, and documentation changes.

## Spec Update

- Define the default cache-backed replay profile and distinguish it from the
  explicit exact-float correctness oracle.
- Retain the fractional endpoint-quantization experiment and its cache rules.

## Docs Updates

- Update `volume-cartographer/docs/fiberlets.md` with the production default,
  exact oracle, eager-path exception, and measured Paris4 result.

## Changelog Update

- Record adoption of the compact float-position profile as the default and the
  q1/8 evaluation result.
