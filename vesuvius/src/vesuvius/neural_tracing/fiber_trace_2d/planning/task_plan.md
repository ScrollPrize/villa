# Plan: Evaluate node-unary Mixed BP on the 1024 crop

## Contract

- Reuse the existing 500-fiber 1024 trace dataset, 0.9 direction-dominance
  reference, no-split fibers, and perpendicular-only constraint graph.
- Fix `T=2.5` and sweep unary costs `2,4,6,8,10,12,16`; do not alter the solver
  during this experiment. Also run binary sum-product on the same graph and
  temperature as an H/V-only baseline.
- Treat Direction1/Direction2 confusion as a diagnostic only because the
  initial geometric split contains known errors. Prefer settings which produce
  a stable definite H/V partition with low resolved perpendicular-constraint
  mismatch and few exact ties; use agreement with the initial grouping and
  Mixed recall as secondary diagnostics.
- Define a confidently oriented fiber as `max(P(V),P(H)) >= 0.75` with V or H
  the unique top state. For the existing 0.25/0.75 resolved orientation bands,
  report resolved factor-count and factor-strength coverage together with hard
  mismatch rates; never interpret unresolved factors as successful matches.
- Report argmax label churn between neighboring unary costs. Prefer the Pareto
  frontier, then maximize confident H/V coverage subject to low weighted hard
  mismatch, using lower churn and fewer non-gauge ties as tie breakers.
- Report connected components and isolates. Treat exact V/H ties outside the
  seeded component as gauge ambiguity rather than instability.
- Write the selected result to the existing main 1024 basename so its short
  H/V/Mixed/tie OBJ layers are immediately inspectable.

## Spec Update

None; this is evaluation of the committed formulation.

## Docs Updates

Record the command, sweep, selected setting, confusion counts, AUROC, and output
location in the current task log. No user documentation change is required.

## Testing

- Confirm every run uses 500 fibers and 4,941 perpendicular factors.
- Record convergence and the full Direction1/Direction2/Mixed argmax table.
- Record confident H/V counts, resolved constraint coverage and mismatch,
  neighboring-cost label churn, components/isolates, and the same-temperature
  binary baseline.
- Confirm the selected run overwrites the main 1024 OBJ and CSV artifacts.
- Confirm the four short state OBJs partition exactly all 500 fibers.

## Changelog

No changelog entry unless the selected setting changes a committed default.
