# Changelog

## 2026-10-13

- Added H-only ruled/extrapolated initialization and fixed-UV positional fitting
  through Lasagna's existing scale-space model and optimizer.
- Added positive weak-to-strong data ramp, immutable curve targets, support and
  geometry diagnostics, tifxyz output and managed GPU dispatch.
- Shared AFV polyline interpolation with the earlier surface workflow.
- Validated CPU pilot and regressions; documented remaining local folds and
  unmeasured GPU/whole-scroll behavior explicitly.
