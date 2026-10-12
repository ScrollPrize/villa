# Enabled GT constraint L1 parameter search

## Task
Commit existing corrected solves and detailed scoring results, then iteratively
tune loss weights/parameters to minimize final enabled-event L1 against the
frozen seven-fiber GT solution. Prior work committed as 016cdf185.

## Implementation
- Reuse the 6735 extracted cross events, their signed measurements, fixed GT
  solution and shared-phase coarse-coordinate evaluator. No new extraction and
  no GT training constraints. Only final enabled test pieces contribute.
- Score mean absolute continuous winding residual, not rounded correctness.
  Remove one global integer winding gauge: minimize L1 using the full median
  interval, deterministic ties by abs offset then signed offset. This differs
  intentionally from old modal accuracy offset.
  Reject empty enabled cohorts; report event coverage, type-specific L1, retained
  length, old spiral L1, solver loss and width factor for every trial.
- Extract a reusable evaluator from existing scorer rather than duplicating
  phase/assignment logic. Bind cached events to tensor/GT/input/code hashes.
- Extend existing resumable coordinate search with configurable metric, keeping
  historical spiral-L1 searches unchanged. Fresh init, seed0, current corrected
  baseline, same 10080 updates / global batch24576 / six managed exclusive GPUs.
- Search most influential first: distance, sign, rejection, perpendicular
  relative weight, position distance, priors, LR. Start finite factors .5/2,
  widen/refine as indicated by accepted improvements; revisit coordinates after
  each acceptance. Persist full per-trial metrics and acceptances atomically.
- Verify baseline and first trials; keep the resumable controller running through
  refinement and report accepted improvements. Do not claim grid convergence
  before it actually finishes.

## Spec update
New explicitly enabled-only, equal-event GT continuous L1 tuning objective.
No changes to extraction, optimizer losses, coarse phase, or GT assignments.
Disabled cohort changes can improve metric artificially; disclose coverage, no
unrequested coverage penalty. Seven reviewed fibers are tuning data, not holdout.

## Docs updates
Record metric equation, exact commands/config, baseline and accepted candidates
in results/task_log/status; preserve previous results and default until measured.

## Validation
Independent plan review. Tests for L1 median integer gauge, empty cohort,
disabled exclusion, phase shifts and unchanged old metric/search behavior.
Re-evaluate frozen fresh/ref states, compare direct saved-event residuals.
Run via approved runner only; verify GPU manager allocation, checkpointed trials,
objective selection and first completed trial before handoff.
