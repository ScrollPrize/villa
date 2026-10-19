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

## Fractional measurement mode
Default `target_mode: fractional` preserves signed calibrated fractional centers, clamped in outward coordinates to the existing same/next order margin. Add exact integer seam transport, then represent t=n+alpha by targets n and n+1 with weights w(1-alpha), w*alpha. This equals w|integer_difference-t| exactly on integer labels. Retain baseline-unique unsigned integer targets; exclude the same ambiguous observations. Sign-only cases keep their original integer hinges. Expanded flow arcs carry an original-event index; event coverage is not an arc count. Rounded targets remain available through explicit `target_mode: rounded`.
