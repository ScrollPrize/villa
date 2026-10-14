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
# 2026-10-14 — connected H/V surface attachment mode

- Add weaker signed-offset V targets from saved crossings, constant-end/global
  fallback; shared component UV translations with exact crossing locks and free
  intermediate attachments. Preserve H-only workflows and restore full motion state.
- Real three-step smoke completed; 1518 crossing pairs remain exactly coincident.
  This is implementation validation, not a converged or fold-repaired surface.
- Follow-up800-step deformed solve completed (22.79s). Central CT/HV overlay and
  inspection-only tifxyz exported. Central normal RMS:H1.0124,V6.0967basevx;
  735central folded quads remain. Not promoted as repaired/best surface.

# 2026-10-14: component selection

Connected-HV surface inputs now default to mixed-family connected components of
at least three distinct source fibers. Whole-component filtering precedes mesh
initialization; explicit zero threshold preserves unfiltered replay.

## Stronger UV regularization trial

Matched smooth10/100 and anchor0.1 candidates reduce local offset variation;
smooth100 gives81%lower local-offset p95 and88%lower slope-change RMS, with worse
normal fit and local folds. Added read-only matched-run UV audit; no solver
formula/global default change. See task_log.md and uv_regularization_plan.md.

## Tangential geometry fitting

Added optional geometry-only tangential error at detached currentUV,checkpoint
persistence and per-family full/tangent RMS. Matched smooth100 trials substantially
improve alignment while increasing folds;0.1 and1 rendered for comparison.

## Future render dimensions

Halved future image dimensions and fiber-line widths through defaults/presets
(8baseUV/pixel,4px). Existing outputs unchanged; about quarter the pixel count.
