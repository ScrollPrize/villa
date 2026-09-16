# Plan

1. Independent review against task/specs/plan.
2. Change only refinement.direction_weight 1.0 -> 0.1; preserve LR0.01,
   model loss weights, smoothing0.1/10, original11800 resume and geometry rules.
3. Run CPU refinement regressions via approved cooperative test launcher.
4. Start via approved experiment production-start (manager allocation); verify
   launch identity, source checkpoint and startup health.

## Spec update
None: existing configurable weights cover this experiment.

## Docs updates
Update current example weight, task log and status with validation and launch.

## Changelog
No feature entry needed for this weight-only experiment; prior trial is committed.
