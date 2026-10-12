# Balanced perpendicular-dominant constraint losses

## LR follow-up
User stopped original suite, requests10x and0.1xLR. Fresh already finished10k,
reference stopped with2k last saved diagnostic, no final snapshot. Preserve both.
Run4 original-initialized10k trials (fresh/ref times high/lowstateLR); widthLR2
unchanged, explicitly communicated. High0.07348469228349534,low0.0007348469228349534.
Same6GPU globalbatch and objective. Independent config review then approved
run_uv_balanced_lr_suite.json. Reportall1kcheckpoints inclwidthfactor and ranges,
classsigns,retention,referenceL1 andwithin2,loss. No code changes for this follow-up.

## Task
Enable the existing fixed sign violation charge. Balance parallel/perpendicular
constraints by full-dataset class counts, then give perpendicular total weight
10 versus parallel 1, for every event-associated term.

## Implementation
Add opt-in constraint_class_balance and perpendicular_relative_weight (10).
For E events with P parallel and Q perpendicular: parallel weight E/(11P),
perpendicular weight 10E/(11Q). Mean event weight remains one. Compute once
from the entire graph, not each minibatch/rank. Multiply sampling importance
weights by these class weights before pair loss and endpoint phase priors.
This includes distance, vertical equality, sign hinge and fixed charge.
Uniform unary rejection/geometry priors and continuations are not classed.
Missing classes normalize over present classes; reject invalid relative weights.
Unconfigured runs preserve existing behavior. Reject unsupported categorical
cost-target mode with balancing rather than silently ignoring the option.

Enable sign_charge_multiplier=32, retain sign_weight10
and rejection_cost10. For this subset parallel weight~0.117 and perpendicular
weight~4.058; multiplier32 also retains rejection incentive for isolated parallel
violations with the other endpoint at least50% retained. No guarantee of global
feasibility in finite stochastic optimization. Keep hard-readout convention and
all other settings unchanged. Original-init fresh/ref,10k passes each,6GPUs,
same approved runner and new output roots; report each1k checkpoint.

## Validation
Independent plan review. Tests for exact10:1 class sums, mean1, absent classes,
invalid settings, default identity, all pair components and gradients weighted,
fixed-charge tiny violation and rejection gradient after class weighting.
Existing pair/distance/sign tests and production six-GPU preflights. Compare
unweighted sign diagnostics per class, retention and reference metrics with10k
baseline. Do not compare scalar losses as if objectives were identical.

## Spec update
Document opt-in class-balanced objective and activated fixed charge for this
experiment; no geometry/extraction changes or implicit changes to old configs.

## Docs updates
Record exact weights, commands, metrics in a new results document; update
task/status/log, README and changelog. Log numerical/protocol deviations.
