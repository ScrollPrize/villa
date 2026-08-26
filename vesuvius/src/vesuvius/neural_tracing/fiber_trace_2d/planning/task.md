# Task: parallel constraint scoring and OBJ diagnostics

Accelerate stored crop-trace constraint scoring without changing its numerical
definition, then write three connector-line OBJ diagnostics:

- perpendicular constraints with normalized perpendicular score greater than
  `0.5` and normal-modulated winding distance greater than `0.3`;
- parallel constraints with normalized parallel score greater than `0.5` and
  winding distance less than `0.5` (same winding);
- parallel constraints with normalized parallel score greater than `0.5` and
  winding distance greater than or equal to `0.5` (separate winding).

Only measured inter-trace links belong in these views. Hard same-trace
continuity links have no spatial connector and must be excluded.
