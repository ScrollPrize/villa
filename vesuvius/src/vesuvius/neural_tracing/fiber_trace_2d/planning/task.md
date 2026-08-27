# Task: Fixed pre-pass orientation for winding BP

Add an option shared by both winding solver variants which first solves the
existing H/V/Mixed orientation problem, converts that result to one fixed class
per piece, and then solves only winding while keeping those classes unchanged.

- Support both `joint-grid` and `alternating` winding solvers.
- Preserve each solver's existing winding, calibration, component-sign, and
  adaptive integer-support behavior.
- Make the mode explicit in CLI output and persisted diagnostics.
- Keep current joint/interleaved orientation inference as the default.
