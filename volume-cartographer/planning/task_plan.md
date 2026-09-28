# Plan

1. Replace context-menu callback with an indexed CP callback at the menu bottom.
2. Expose whole-fiber reset through the annotation window menu.
3. Share controller reset/save logic with optional CP scope. Mirror matching
   peer-pane CPs and dirty adjacent spans only when a direction is removed.
4. Test reset scope and metadata preservation; build VC3D and run focused tests.

## Spec Update

Distinguish CP-local context reset from whole-fiber annotation-menu reset.

## Docs Updates

Update line_annotation_fibers.md with labels, scope and reoptimization behavior.

## Review

Local review; independent reviewer unavailable.

## Changelog

Record correction-reset menu scope change.
