# Plan: iterative ordered-winding offender removal

## Semantics

1. Operate on the admitted dominant sign factors used by the ordered Ceres fit.
2. Evaluate a sign factor against the fitted interleaved coordinate
   `offset + H/V phase`; zero signed separation is infringed.
3. Aggregate incident and infringed factors by source trace. An edge within one
   trace contributes once; a cross-trace edge contributes once to each endpoint.
4. Select the trace with maximum exact rational infringement percentage. Break
   ties by larger infringed count, then lower source trace index.
5. Exclude every piece of the selected trace through an explicit active-piece
   mask. Preserve its original H/V orientation for downstream diagnostics.
   Rebuild and re-solve the ordering so all subsequent scores are current.
6. Stop when the fitted graph has zero infringed sign factors. Isolated fibers
   with no incident sign factor are not offenders. The process terminates after
   at most the number of represented source traces.
7. Run the ordinary ordered-cut scan once on the final retained cohort. Removed
   fibers remain inactive/Defect in downstream diagnostics and artifacts.

## Implementation

- Add core report structures and a reusable function that evaluates continuous
  sign infringements by source trace and deterministically selects the worst.
- Add an iterative core driver around the existing ordered fit/cut solver; do
  not duplicate residual construction.
- Expose `--ordered-prune-offenders` for `--winding-solver ordered-cuts` only.
- Print one aligned row per removal containing iteration, source trace,
  original trace, pieces, local infringed/incident percentage, old full-graph
  infringements, surviving-edge infringements before re-solving, those same
  surviving-edge infringements after re-solving, remaining fibers, and solve
  time. This separates the effect of edge deletion from re-optimization.
- Require every nonempty Ceres result to be usable with finite active offsets.
  Treat a fully exhausted graph as a valid empty result without invoking Ceres.
- Keep reference-oracle split selection after offender pruning.

## Tests

- Synthetic contradictory ordering where the expected high-percentage trace is
  removed and the re-solved graph reaches zero infringements.
- Verify all pieces of a source trace are excluded together.
- Verify exact overflow-safe percentage comparison (`1/1 > 9/10` and
  `1/2 == 2/4`), deterministic tie breaking, internal factors counted once,
  cross factors counted at both endpoints, and zero-degree exclusion.
- Prove the second selection uses a genuinely re-solved ordering, cover an
  irreducible within-trace violation, safe full exhaustion, and the removal
  count bound.
- Verify the absent option preserves the existing result and reject the option
  for non-ordered solvers.
- GCC and Clang focused winding tests.
- Release run on the 1024 crop with `piece-length=384`; inspect removal progress,
  final infringement count, runtime, and reference agreement.

## Spec update

Document whole-trace removal, exact percentage ranking, deterministic ties,
re-solving, termination, and opt-in behavior in `planning/specs.md`.

## Documentation updates

Document the CLI option, table columns, computational cost, and diagnostic-only
status in `volume-cartographer/docs/fiber_chunk_tracing.md`.

## Changelog

Add a concise entry to both relevant changelogs after validation.

## Status and task log

Keep `planning/status.md` current and record deviations plus exact validation
commands/results in the task-local `planning/task_log.md`.
