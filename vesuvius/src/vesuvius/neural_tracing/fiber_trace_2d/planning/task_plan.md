# Plan: Merge Main Line-Annotation GUI Changes

## Merge Resolution

1. Resolve `LineAnnotationDialog` toolbar conflicts by keeping `main`'s
   Annotation popup, action-backed auto-reoptimization, tag pills, length
   label, and status presentation.
2. Retain the fiber branch's global Lasagna/fiber-model selector,
   extrapolation-distance control, accessors, setter, and signals.
3. Accept `main`'s intentional removal of the seed-direction and
   shift-scroll-mode selectors. Keep current-cut Shift+scroll along-line and
   seed placement in the main-defined direction.
4. Preserve all controller-side fiber mode, extrapolation, descriptor, trace,
   fallback, metric, and persistence behavior that merged automatically.

## Generated View Layout

1. Keep `main`'s fixed-height schematic overview map as a separate full-width
   widget immediately below the toolbar.
2. Keep the current-cut and side-cut viewers in the horizontal cut splitter.
3. Restore the branch's two-entry rendered-strip construction inside the
   vertical strip splitter, in this order:
   - `lineSurface` (top-view rendered strip);
   - `lineSideSlice` (side-view rendered strip).
4. Treat both rendered strips identically as ordinary interactive viewers:
   retain mouse-follow, click placement, Ctrl-right-click menus, pan, zoom,
   scrolling, camera persistence, overlays, and span labels.
5. Do not use or restore the older fixed rendered-strip implementation,
   fit-to-width camera policy, interaction swallowing, hidden scalebar/stats,
   or fixed rendered-strip height.

## In-Place Refresh And Overlay Consistency

1. Extend the `main` in-place update eligibility check from one rendered strip
   to the exact two-strip topology and validate both surface names.
2. Replace the single rendered-strip pending flag with per-strip pending state
   so each strip retains the old overlay until that viewer displays a frame at
   its current surface-geometry epoch.
3. Route static overlays, current-position markers, and span labels through
   the held pre-update data independently for each strip.
4. Keep `main`'s current-cut and side-cut frame/overlay synchronization,
   placement-focus handoff, and surface epoch checks unchanged.
5. Keep the schematic overview immediate: it may update before rendered panes,
   matching `main`'s explicit click-feedback behavior.
6. Preserve the teardown crash fix by clearing every pane/viewer reference and
   pending state before deleting generated containers.

## Toolbar And State Details

1. Keep action-backed `reoptimizationMode()` and action enablement for Show as
   mesh and Reinit reoptimization.
2. Keep `fiberOptimizationMode()` and `setFiberOptimizationMode()` backed by a
   `QComboBox`; restore the complete Qt include and event-filter installation
   required by that one retained combo.
3. Keep extrapolation-distance persistence and automatic reoptimization signal
   behavior.
4. Ensure busy state disables the retained fiber selector and extrapolation
   control without regressing main's action and tag enablement.

## Tests

1. Run `git diff --check` and verify there are no conflict markers or unmerged
   index entries.
2. Build `VC3D` and the focused line-annotation test target with `-j32`.
3. Run `test_line_annotation_generated_views`.
4. Build the production fiber-facing CLI targets touched by the merged branch
   when needed to catch shared-header regressions.
5. Inspect the final merge diff for accidental loss of fiber controls,
   two-strip construction, span labels, or main's lifecycle fixes.

## Spec Update

- Add the VC3D generated-view contract: a separate schematic full-width map
  plus two interactive rendered strip viewers, both carrying persisted
  per-span status labels. Explicitly exclude the obsolete fixed,
  fit-to-width rendered-strip behavior.

## Docs Updates

- Update `volume-cartographer/docs/line_annotation_fibers.md` with the merged
  toolbar and generated-view layout, including the interaction distinction
  between the schematic map and the two rendered strips.
- Update `docs/code_structure.md` with the dialog ownership and independent
  per-pane overlay/frame synchronization behavior.

## Changelog

- Record the merge of the main line-annotation GUI rework while retaining both
  interactive fiber strip views and their span diagnostics.
