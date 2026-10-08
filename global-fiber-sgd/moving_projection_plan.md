# Moving-window perpendicular refinement

User task: recenter each perpendicular alternating-projection search window on
the current point, so the initial seed window does not block refinement.

1. Change shared native local-event scorer: B searches around current B, then A
   searches around current A using the updated B. Keep parallel behavior fixed.
   Add explicit legacy fixed-window option for reproducible old artifacts.
   Preserve physical endpoint bounds, distance/support/orientation/winding checks,
   and iteration cap. Do not relax tolerances or claim global closest points.
2. Track reciprocal failures touching the last projection window (not obsolete
   seed bounds). Record the mode in benchmark metadata.
3. Add a synthetic perpendicular regression that crosses initial window limits;
   verify more iterations converge with moving windows, parallel unchanged, and
   existing native suite/legacy real-data pilot pass.
4. Approved build and runner: new output directory for full-volume 4/16/64 sweep;
   no overwritten baseline or database. Compare attempt/rejection/event counts
   and time against saved fixed-window sweep. Retain 4 as default unless evidence
   warrants proposing a different cap. No inference or winding solve.

## Spec and documentation updates

Update bp_constraints.md for moving-window semantics and compatibility option;
write moving_projection_results.md, current task_log/status and changelog. Report
candidate-vs-unique denominators, adaptive candidate changes, window association
not proof, and any remaining finite-iteration/local-minimum limitations.

Independent plan review required by the fiber tracing AGENTS workflow.

Measured follow-up:16and64have zero reciprocal rejections and equal accepted
perpendicular counts;4has46rejections. Select16as the practical new default,
explicitly not claiming16/64event equality or global convergence. Pin historical
configs to4and fixed windows; rebuild/revalidate native suite and legacy pilot.
