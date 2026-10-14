# Changelog

## 2026-10-13

- Added H-only ruled/extrapolated initialization and fixed-UV positional fitting
  through Lasagna's existing scale-space model and optimizer.
- Added positive weak-to-strong data ramp, immutable curve targets, support and
  geometry diagnostics, tifxyz output and managed GPU dispatch.
- Shared AFV polyline interpolation with the earlier surface workflow.
- Validated CPU pilot and regressions; documented remaining local folds and
  unmeasured GPU/whole-scroll behavior explicitly.
# 2026-10-14

Added opt-in jointly sliding per-attachment UV fitting with detached-gradient
normal/tangential losses, implicit/deformed layout support, curved UV initialization,
shared-GPU runner configurations and both measured regional results. Existing
fixed-UV mode preserved. See joint_uv_results.md for limitations and commands.

Tested3×/10×bending and smoothness on both layouts.3×smoothness lowers local folds
without material mean-error degradation; residual folds persist. Added shared
post-hoc support audit and full tables in smoothing_results.md. Defaults unchanged.
# 2026-10-14 — patch repair experiments

- Added opt-in boundary-frozen sparse patch reconstruction, fit/seam acceptance,
  saved-UV joint continuation and soft corner-area guard. Tested both inputs,
  small/large-first, slope-preserving+damped, and stronger-guard variants.
- Immediate best folds183/184 vs225/233; not fold-free and continuation can
  undo repairs. No default promotion. Results/commands:repair_results.md.

# 2026-10-14: matched surface CT/fiber render

Added read-only native CT sampling and fitted-H attachment overlays, compressed
TIFFs with identical vertex-aligned mapping, source/piece metadata and provenance.
No GPU, solver or surface changes. See render_plan.md and task_log.md.
