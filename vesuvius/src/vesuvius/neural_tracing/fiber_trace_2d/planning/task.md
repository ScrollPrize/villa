# Preserve Native Diagnostics And Truncate Extrapolation At Invalid Data

## User Report

- Persisted native meeting-error labels appear after loading a fiber but vanish
  after pressing Reoptimize.
- Native extrapolation that reaches the volume edge, represented by invalid
  prediction directions, must stop at its last valid point instead of restoring
  the Lasagna fallback tail.

## Required Behavior

- Generated branch-overlay refreshes must restore current CP-owned native
  meeting/failure diagnostics rather than clearing them after reoptimization.
- Pressing Reoptimize in Lasagna mode must continue to protect existing accepted
  native spans; only an explicit mode change to Lasagna or per-span revert may
  clear them.
- One-way extrapolation must retain its last valid partial path when the next
  candidate generation has no valid directions.
- VC3D must treat `no_valid_candidates` extrapolation with a nontrivial retained
  path as a valid truncated native tail.
- Other incomplete reasons, including step-budget exhaustion, continue to use
  the existing Lasagna fallback.
