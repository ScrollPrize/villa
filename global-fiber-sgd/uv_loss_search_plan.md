# Adaptive loss-weight search

User requests final-reference-L1 coordinate tuning, accepting improvements and
revisiting all other weights after each acceptance. No intermediate 1k reports.
Starting state is committed at 102d25a55 (only unrelated test temp untracked).

Use fresh seed0, original initialization, 10080 updates (~10k sampled passes),
same subset, six-GPU batch24576 and stateLR .07348469228349533 / widthLR2.
Start from position-weight .5 baseline. Earlier sign/HV stages remain frozen.
All joint U/V/broken variables remain optimized from the first update.

Implement a resumable coordinate-search controller through the existing approved
crop benchmark runner and existing managed distributed solver, not a new solver.
For each configurable weight, try 1/4 and 4 times its ORIGINAL baseline value,
then refine on the finite log2 grid [-2,-1,0,1,2] relative to the baseline.
Immediately accept final L1 improvements >1e-4 windings, restart the coordinate
list, cache exact tried configurations. Coarse first, then all fine-grid values.
Finish only after a complete pass tests every alternative grid value of every
coordinate around the final incumbent with no improvement. This is a bounded
coordinate optimum, not a claim of global optimum. Preserve nonzero sign costs.

Order: position_distance_weight, perpendicular_relative_weight, rejection_cost,
distance_weight, sign_charge_multiplier, sign_weight, prior_weight, metric_weight,
entropy_weight, v_prior_weight, offset_soft_weight, offset_limit_weight,
continuation_weight, width_prior, width_smooth. Optimizer LR is not changed.
Final metric alone determines acceptance, but every result records retained
length, within2, sign errors, width ratio and objective. Objective magnitudes
across different weights are not directly comparable. Reference is now tuning
data, not held-out evaluation. No automatic global production-default change.

Checkpoint trial configurations and compact results atomically after each solve;
on resume reuse only completed matching configurations. Failed/incomplete trials
must not count as completed search. Report accepted improvements only; no 1k
diagnostics, saved solve history contains start/end. All candidates are fresh.

## Spec update
No mathematical changes. Add opt-in experiment controller, bounded search and
strict final-metric selection. Missing root spec.md/plan.md noted; current
mathematical contract remains uv_bucket_split_plan.md and preceding plans.

## Validation
Independent plan/code review; CPU fake-solver tests for acceptance, restart,
finite-grid termination and resume; existing bucket math tests; production
six-rank compiled gradient preflight and finite-state validation per trial.

## Docs updates
Persist all trials, acceptance chain, final config and termination certificate;
update task/status/log, README/changelog and results Markdown. Commit successes.
