# Plan

1. Add a reusable population benchmark that maps final active piece latent
   coordinates through the final reference calibration and counts the inclusive
   annotated half-winding interval.
2. Report all removed pieces against final retained in-range pieces as both a
   percentage of their combined population and a removed/retained percentage.
3. Extend shared constraint-agreement evaluation with unique per-constraint
   state, then classify removed-incident, final-Defect, infringed, and fulfilled
   constraints without double-counting multi-term constraints. Preserve the
   exact original constraint indices for arbitrary retained subsets; rebuilt
   oracle graphs use their compact local indices instead.
4. Add focused unit coverage for calibration, interval validation, unique
   constraint classification, non-contiguous retained indices, conditioned
   prefixes, and benchmark arithmetic.
5. Build and run the focused winding-BP tests, then rerun the 1024 diagnostic
   and report its measured population retention.

## Spec Update

Document the piece and unique-constraint benchmark definitions.

## Docs Update

Add the diagnostic table and interpretation to
`volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog

Record the new reference-winding population diagnostic in the task changelog.
