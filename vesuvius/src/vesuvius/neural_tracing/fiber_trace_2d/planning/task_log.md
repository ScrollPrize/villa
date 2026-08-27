# Task log: Joint adaptive-grid winding BP

## Decisions

- Aligned-normal sign resolution remains an independent preprocessing result.
  It is not the H/V/Mixed orientation inference discussed by the joint solver.
- H/V/Mixed, integer winding, phase, scale, and the necessary ladder-order
  gauges belong to one joint inference model.
- Calibration uses an explicit crop-global variable on an absolute sliding
  grid over log inverse-scale and canonical phase. Retained cells keep their
  physical identity; support moves only under boundary pressure rather than
  remapping existing message indices.
- Each constraint component has a separate binary ladder-order gauge, while
  every component contributes to and receives the same global calibration
  posterior.
- The independent plan review required the complete orientation+winding factor
  equation, finite continuity-factor wording, conservative integer support,
  explicit message transport/window-growth rules, and brute-force tiny-tree
  validation. These corrections are incorporated in `task_plan.md`.
- The new `joint-grid` path becomes the default. The existing alternating
  multi-start/calibration implementation remains explicitly selectable for
  comparison and must not be silently used as a fallback.

## Deviations

- Implementation and validation are deferred at the user's request so an
  intervening fix can be handled first.

## Validation

- Independent plan review completed and its correctness findings were resolved
  in the plan. No implementation validation has been run for this deferred
  task.
