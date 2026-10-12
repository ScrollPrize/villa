# Expanded reviewed-fiber GT and constraint-vote evaluation

## Task
Rerun the expanded bbox_gt JSON set as intact, non-rejectable fibers. Then use
its inferred relative winding layout as a new GT evaluation format for saved
solutions, including disabled pieces, without optimizing the tested solutions.

## Plan
1. Preserve the previous four-fiber artifacts; run existing shared pipeline in
   bbox_gt_solve_v2 with the seven currently supplied fibers.
2. Audit connectivity, retained fibers and sign violations before treating the
   inferred layout as GT. Do not claim geometry alone independently labels windings.
3. Confirm the two saved comparison artifacts and exact disabled-piece scoring
   contract. Extract cross constraints using existing native extraction, never
   a duplicate geometric implementation. Map event positions into existing pieces.
4. Each cross constraint induces a GT winding from the fixed solution; apply one
   global integer correction, then report exact matches / all eligible constraints
   as a percentage, plus support counts, families and disabled-piece breakdown.
   Never silently remove disabled pieces or choose an offset per GT fiber.
5. Validate offset invariance, incorrect winding votes, disabled pieces, piece
   boundary mapping and unchanged saved solutions. Persist provenance/results.

## Spec update
Reviewed fibers remain whole and non-rejectable. This adds a constraint-based
evaluation, not another optimizer or reference-based correction of tested solves.
Exact vote rounding/sign conventions must be documented before implementation.

## Docs and validation
Update task/status/log and a durable bbox_gt evaluation report. Use approved
runner only for managed experiments; existing adapter/solver tests plus metric
regressions. No dependency installs, no baseline replacement before validation.
