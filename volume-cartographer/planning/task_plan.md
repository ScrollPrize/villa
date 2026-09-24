# Plan

1. Remove LineRibbonSurface and the added virtual sampling API; use ordinary QuadSurface.
2. Restore support-grid construction, transported/roll-smoothed frames, geometric depth normals and indexed projection.
3. Apply only manual angular corrections to baseline frames at construction; restore windowed cross-view tangents and linear centers.
4. Preserve CP displacement and provenance. Add baseline-equivalence and ordinary-surface regression tests; build VC3D and run focused tests.

## Spec Update

Prohibit custom runtime ribbon evaluators and cubic strip upsampling. Document construction-only frame corrections.

## Docs Updates

Update line_annotation_fibers.md, spec.md, status and task log.

## Review

Independent agent tool unavailable; local diff and regression review instead.

## Changelog

Record removal of the runtime ribbon regression.
