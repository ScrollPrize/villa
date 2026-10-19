# Contract
- Fixed measured constraints, H/V families and normal orientation; integer winding variables; fixed geometric seam transport.
- H/V membership changes integer target interpretation, never adds fractional winding variables.
- Optimize sum w_e |k_b-k_a-d_e| globally through the dual circulation. Report a primal/dual optimality certificate.
- Keep source worktrees and datasets unchanged. Record source commit, input paths/hashes, configuration, all exclusions and all seven benchmark results.
- Ground truth is evaluation only except when a benchmark explicitly includes GT fibers as ordinary input.
- No fitted geometry, width optimization, learned uncertainty or fiber disabling. Distinguish unavailable continuous-layout metrics from integer winding metrics.
- Ambiguous unsigned constraints cannot silently acquire an arbitrary sign. Report their handling and coverage explicitly.

## Resolved evidence policy
Retain unsigned observations only where both signed target interpretations coincide. Explicitly exclude and count ambiguous target sets (20 per large case here); the optimality certificate covers the retained model. For sign-only benchmark modes use one-sided integer ordering penalties. Angle interpolation is not physical seam transport: use raw XYZ intersections consistently for solving and reference labels. All missing source-audit inputs must be recorded; verify saved SGD cache hashes before solving.
