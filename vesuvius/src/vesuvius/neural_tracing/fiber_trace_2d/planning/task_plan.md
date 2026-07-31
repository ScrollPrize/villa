# Plan: Per-Segment Interpolation Goals And Cubic-Spline Fallback

## Data Model And Persistence

1. Replace the native-trace-specific optional segment state with a general
   CP-owned segment descriptor. Control point `i` continues to own the segment
   to control point `i+1`; the final control point owns no segment.
2. Add strict enums and JSON conversions for:
   - `interp_goal`: `global`, `cspline`, `lasagna`, `trace`.
   - `interp_mode`: `cspline`, `lasagna`, `trace`.
3. Store both fields on every new non-final control-point object. Treat
   `interp_goal` as the persistent request and `interp_mode` as the actual
   producer of the currently stored dense polyline. Never infer the actual
   producer from the fiber-wide mode after loading.
4. Store two common display/diagnostic fields in the segment descriptor:
   - Optional numeric `metric`, interpreted only by `interp_mode`: final maximum
     normal-alignment error in degrees for `lasagna`, minimum meeting-plane
     error in base voxels for `trace`, and absent for `cspline`.
   - String `msg`, containing compact persisted debug/status information. A
     direct successful result records a compact mode result; a fallback result
     records the rejected attempt reason or reason chain.
5. Keep detailed mode-specific diagnostics in the same segment descriptor
   without duplicating derivable geometry:
   - Trace configuration, accepted meeting diagnostics, and trace failure
     diagnostics remain trace-specific.
   - Add a compact Lasagna failure code/detail when its attempt falls through
     to `cspline`.
   - Clear stale metric, message, and mode-specific failure fields whenever a
     higher-priority attempt succeeds on a later reoptimization.
6. Write a new `vc3d_fiber` schema version. Continue loading versions 1 and 2:
   - A segment without metadata becomes `interp_goal: global` and
     `interp_mode: lasagna`.
   - An accepted v2 native segment becomes actual `trace`; its goal becomes
     `global` when the old fiber-wide mode was trace and explicit `trace` when
     the old fiber-wide mode was Lasagna.
   - A v2 trace-fallback record becomes goal `global`, actual `lasagna`, and
     retains its trace failure diagnostic.
   - Promote an accepted v2 meeting error into the new trace `metric`. Derive a
     compact `msg` from legacy trace diagnostics. Legacy Lasagna geometry has no
     persisted metric, so calculate its maximum normal error when the fiber is
     opened and persist it on the next save.
   New saves always emit the new schema and explicit fields.
7. Update every strict reader/writer that consumes VC3D fibers: VC3D, the
   shared native fiber JSON reader, the Python fiber parser, and `fiber_merge`.
   Preserve descriptors through scaling, snapshots, import/export, merges,
   branch updates, and asynchronous saves.
8. Define CP mutation rules explicitly:
   - Splitting a segment by inserting a CP copies the original goal to both new
     segments and invalidates both actual results for regeneration.
   - Moving a CP preserves goals and invalidates both adjacent actual results.
   - Deleting an interior CP leaves the surviving left owner goal on the newly
     merged segment and removes the deleted owner descriptor.
   - Reordering or an invalid final-CP descriptor fails validation loudly.

## Goal Resolution And Fallback Pipeline

1. Keep the fiber-wide selector as the existing Lasagna/trace choice. Resolve
   `interp_goal: global` to that mode; explicit goals ignore later global mode
   changes.
2. Measure the automatic-short rule in base coordinates before any trace-scale
   conversion, using straight-line Euclidean distance between its CP endpoints.
   For a global Lasagna/trace segment with distance `< 100` voxels, select
   `cspline` directly and record actual mode `cspline`; exactly 100 voxels still
   attempts the resolved global mode. Do not record a fake trace or Lasagna
   failure. Explicit `lasagna` and `trace` goals always attempt their requested
   mode regardless of distance.
3. Build one deterministic optimization coordinator over ordered CP spans:
   - Resolve initial requested modes and the dirty dependency region.
   - Attempt eligible trace spans independently using the existing native
     tracer and accepted/failure semantics.
   - Initialize every span that now requires Lasagna with the existing
     per-span rollout/reinitialization behavior. Decide success or fallback for
     that span from its own usable-candidate result; one failed span must not
     demote successful neighboring Lasagna spans.
   - Combine adjacent spans that now require `cspline` into maximal runs and
     solve each run jointly after all per-span trace and Lasagna outcomes are
     known.
   - Stitch all successful initial spans, then jointly Ceres-refine connected
     Lasagna geometry while protecting actual trace and `cspline` spans and
     applying their endpoint directions as hard constraints.
   - Stitch the resulting dense spans once, restore exact CP samples, remap CP
     line indices, and atomically publish all descriptor changes with the line.
4. Extract the existing private Lasagna span initializer/rollout into one shared
   API used by both the current line reinitializer and the new coordinator; do
   not copy its candidate logic into VC3D. Its per-span `failed` and
   `failureReason` determine `lasagna -> cspline`.
5. Treat the subsequent joint Ceres refinement as refinement of already usable
   Lasagna geometry, not as the per-span fallback decision. A non-converged but
   finite refinement retains the initialized Lasagna geometry and actual mode;
   invalid output or structural exceptions fail the task rather than silently
   changing segment modes.
6. Preserve compact reasons through the cascade:
   - A trace gap rejected under the 10-base-voxel-dominated threshold records a
     message such as `trace gap 14.2 vx exceeds 10 vx`.
   - A trace gap rejected under the 10%-of-length-dominated threshold records a
     message such as `trace gap 12.4% exceeds 10%`.
   - Other trace failures use their stable trace reason, and a failed Lasagna
     initialization appends its usable-candidate failure reason.
   The visible `msg` stays compact; existing detailed fields remain available
   for the tooltip and debugging.
7. Re-run from `interp_goal` on every relevant reoptimization rather than
   treating a fallback `interp_mode` as sticky. A previously failed Lasagna or
   trace attempt can therefore become the actual mode after CP or neighboring
   geometry changes.
8. Compute the dirty region as follows:
   - CP insertion, deletion, or movement dirties its adjacent spans.
   - A segment-goal change dirties that segment.
   - A global-mode change dirties every `global` segment but leaves explicit
     segments fixed initially.
   - Expand through each connected Lasagna or `cspline` run that must be solved
     jointly and through a neighboring run when its consumed boundary tangent
     changed. Unaffected explicit segments remain protected and provide hard
     boundary geometry.
9. Preserve extrapolation behavior under the fiber-wide Lasagna/trace mode;
   per-segment goals govern only CP-to-CP spans.

## Shared Cubic-Spline Interpolator

1. Add one reusable core helper for ordered 3D CP interpolation; VC3D must call
   this helper rather than embedding a private spline implementation in the
   controller.
2. Solve each maximal `cspline` run as one chord-length-parameterized,
   piecewise cubic Hermite spline:
   - CP positions are exact interpolation constraints.
   - A neighboring non-`cspline` stored span supplies a hard unit boundary
     tangent from the first distinct dense point next to the boundary CP.
   - With no external boundary geometry, use the natural spline boundary
     condition; the two-CP/no-direction case reduces exactly to a straight
     line.
   - Internal CP derivatives are solved once for the complete run and shared
     by both incident spans, giving continuous direction rather than separate
     per-span estimates.
3. Use a deterministic minimum-bending solve with a fixed, documented tension
   toward the CP chord polyline. The tension discourages avoidable length and
   waviness while the bending term supplies smooth tangent transitions.
4. Limit derivative/Bezier-handle magnitudes from adjacent chord lengths and
   validate finite output, exact endpoints, forward chord progress, and a
   bounded local deviation. Reduce handle magnitudes deterministically when a
   candidate violates the shape checks; invalid or duplicate CP input remains
   a hard error.
5. Resample the curve at the existing base-coordinate annotation spacing,
   retaining every CP exactly and avoiding duplicate stitched endpoints.
6. Do not consult predictions or normals to determine spline geometry. Normal
   samples may be attached opportunistically for existing generated-view frame
   and display machinery; unavailable/invalid normals use the existing
   synthetic line-model path and do not make `cspline` fail.

## GUI And Editing Behavior

1. Replace the current binary native-trace/revert Ctrl-right-click actions with
   a checkable `Interpolation goal` submenu for the containing CP-to-CP span:
   `Global`, `Cubic spline`, `Lasagna`, and `Fiber trace`.
2. Carry segment owner/index, current goal, and actual mode into generated
   control markers. Emit one goal-change signal and route all four actions
   through the same controller path.
3. On selection, preserve a rollback snapshot, update only `interp_goal`, run
   the coordinator over the resulting dependency region, then refresh views,
   metrics, branch metadata, and saved fiber state. Restore both geometry and
   descriptors if the task fails structurally.
4. Route full reoptimization, auto-reoptimization after CP edits, explicit
   segment changes, and global-mode changes through the same coordinator. Remove
   the separate one-span native trace/revert implementation once equivalent
   behavior is covered.
5. Populate strip status directly from the persisted segment descriptor rather
   than asynchronously recomputing the primary value:
   - `trace`: `metric` is the minimum meeting-plane error in base voxels.
   - `lasagna`: `metric` is the final stored span's maximum normal-alignment
     error in degrees, calculated after joint refinement.
   - `cspline`: `metric` is absent.
   Show `msg` for every mode, including the fallback reason chain. The menu
   check state always reflects `interp_goal`, not the actual fallback.
6. Replace the midpoint-only label placement with viewport-space layout:
   - Project both CP endpoints into the strip viewport and intersect that span
     interval with the visible horizontal interval.
   - Show the label whenever this intersection is non-empty, even when the true
     span center is off-screen.
   - Prefer the projected span center, clamp the anchor into the visible part of
     the span, and clamp the complete label rectangle into the viewport. A label
     may extend beyond a very short visible span; requiring the full rectangle
     to fit inside that span would make short spans impossible to label.
   - Pack visible labels in fiber order with left-to-right and right-to-left
     sweeps, pushing the next label as needed while preserving order and keeping
     every rectangle inside the viewport. Use an additional vertical row
     deterministically only when the labels cannot fit without overlap in one
     row.
   - Render the metric and compact `msg` as a two-line label when both exist;
     render whichever one exists alone. Keep detailed trace/Lasagna diagnostics
     in the tooltip.
   Perform sizing and collision checks in viewport pixels because the existing
   label items ignore scene transforms.

## Tests

1. Core spline tests:
   - Two CPs produce the exact straight segment.
   - Several unevenly spaced CPs produce deterministic finite output, exact CP
     samples, shared internal tangents, and no backward/wavy span.
   - One- and two-sided hard boundary directions are honored.
   - Adjacent `cspline` spans are solved jointly and are independent of normal
     and trace samplers.
2. Coordinator tests with fake trace and Lasagna solvers:
   - `trace -> lasagna -> cspline` and `lasagna -> cspline` cascades.
   - Actual mode changes after a later retry succeeds.
   - The 100-base-voxel shortcut applies only to `global` goals and not manual
     goals.
   - Lasagna initialization failure demotes only that span; adjacent successful
     Lasagna spans remain Lasagna and receive the later joint refinement.
   - Adjacent `cspline` fallbacks coalesce into one spline run.
   - Global-mode changes touch all and only global spans unless boundary
     dependency expansion requires an explicit neighboring run.
   - Expected optimizer failures fall back; structural errors remain errors.
3. Persistence tests across C++, Python, and merge tooling:
   - New schema round-trip for every goal/actual combination and diagnostics.
   - Versions 1 and 2 load with the documented inferred values.
   - CP insert/delete/move and fiber scaling preserve goals and update actual
     modes correctly.
   - Mode-dependent `metric` units/absence and `msg` survive save/reload, while
     stale metrics are cleared when actual mode changes.
   - Strict validation rejects unknown enum values and a final-CP descriptor.
4. Generated-view/controller tests:
   - Ctrl-right-click identifies the containing span and exposes all goals with
     the correct checked action.
   - A goal change schedules the correct connected run and rolls back on hard
     failure.
   - Display metrics/messages follow actual mode and survive
     save/reload/reoptimization.
   - A partially visible span retains its label, off-screen centers clamp into
     view, and adjacent labels are packed without overlap at multiple zooms.
5. Run focused C++ and Python suites, then build `VC3D` with `-j32`. Use the
   existing local test trees and do not install or bootstrap dependencies.

## Spec Update

- Replace the fiber-wide-only span policy with the `interp_goal`/`interp_mode`
  contract, exact fallback order, retry semantics, and 100-base-voxel global
  shortcut.
- Specify per-span Lasagna initialization fallback followed by joint refinement,
  connected spline runs, dirty dependency expansion, persistence migration,
  and CP mutation rules.
- Specify persisted mode-dependent `metric`/`msg` semantics and viewport-aware
  multi-label layout.
- Specify the joint chord-length cubic spline, boundary directions, tension,
  shape checks, resampling, and normal-independent geometry contract.
- Clarify that fiber-wide mode continues to select extrapolation behavior and
  resolves only `global` CP-to-CP goals.

## Docs Updates

- Update `docs/code_structure.md` for the general segment coordinator and
  shared spline helper.
- Update `volume-cartographer/docs/line_annotation_fibers.md` with the new JSON
  schema, goal/actual distinction, fallback matrix, grouped solves, threshold,
  metrics/messages, menu and label behavior, and legacy loading.
- Document any new core spline API beside the line optimizer/fiber tracer APIs.

## Changelog

- Record per-segment interpolation goals/actual modes and metrics/messages,
  grouped cubic-spline fallback, the 100-voxel global shortcut, the
  Ctrl-right-click selector, and visible-span label packing.

## Plan Review

- Review schema behavior against existing v1/v2 fiber loading and every strict
  reader before implementation.
- Review fallback grouping so one failed Lasagna initializer cannot overwrite
  successful neighboring Lasagna spans, successful trace spans, or unrelated
  explicit goals.
- Review CP edit and global-mode dependency expansion for stale boundary
  directions.
- Review spline shape constraints against the requirement to remain close to
  the shortest CP path without consulting normals.
- Review label packing in viewport rather than scene coordinates at multiple
  zoom levels and with long fallback messages.

## Confirmed Decisions

1. The 100-voxel rule uses straight-line Euclidean CP distance in base
   coordinates; `< 100` selects `cspline` and exactly 100 attempts the global
   Lasagna/trace mode.
2. Lasagna fallback is decided per span during initialization/rollout. Connected
   successful Lasagna spans are jointly refined only after failed spans have
   been replaced by `cspline` geometry.
3. The fiber-wide selector remains Lasagna/trace. All three concrete modes are
   available as explicit per-segment goals; `cspline` is not a global mode.
