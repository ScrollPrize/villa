# Task: restore aggregate fiberlet replay as the default

The desired two-failure radius-768 baseline is the original fast
whole-fiberlet evaluator, not the newer stepped cost-profile evaluator with
constant weights or averaged subsegment density.

- Default replay to aggregate stored fiberlet and join costs.
- Score from the segment seed through the common horizon and prorate only its
  horizon-crossing fiberlet. The checkpoint is only a commitment boundary.
- Do not read or integrate route cost profiles in the default mode.
- Keep stepped cost-profile evaluation available as an explicit supported
  mode for later experiments.
- Update CLI, tests, specification, and documentation accordingly.
