# Remove redundant pairwise wrapped-phase penalty

User requests removing only pairwise wrapped-phase penalty. Squared is the
selected working candidate; this task does not rerun solves or implement moments.

## Spec update
Remove phase_error² from shared pair_geometry metric for every objective/backend.
Keep winding-coordinate difference (including observed phase_delta and family
offset), sign loss, individual point/angular priors and offset bounds unchanged.
Keep the clamped vertical metric and its weight/scale unchanged. No new toggle
retaining the redundant term; earlier benchmark numbers are historical and
predate this change, so reproduction requires their old source revision.

## Tests and docs
Regression: pair metric has no dependence on q/observed phase but vertical value,
gradient and clamp are unchanged; squared distance still responds to q changes.
Run focused UV objective/geometry/regression tests. Independently review plan
and implementation. Archive preceding task log; update current task/status/log,
changelog and mark earlier comparison as pre-removal. No GPU run needed for
removing arithmetic from the shared eager/compiled function.
