# Plan: Iterative H/V consensus growing

1. Add a separate `consensus` command to `vc_fiber_trace_chunk`. Reuse the
   existing trace-artifact loading, Lasagna compatibility validation, batched
   constraint extraction, winding cutoff, distance, resampling, and thread
   controls. Do not invoke or alter the HiGHS labeling modes.
2. Operate on original stored fibers, not extracted pieces. Map every piece
   constraint to its source trace, discard same-trace links, and count every
   retained piece-pair constraint once as evidence even when a source-trace
   pair has several constraints. Sort each trace adjacency by neighbor and
   stable piece IDs before accumulation so extraction thread count cannot alter
   floating summation or ties. Missing graph links are ordinary absence;
   cross-trace hard links, if presented, count like any other evidence.
3. For every stored line with at least two distinct points, compute arc length,
   endpoint chord, straightness `chord/arc`, and the exact point-to-polyline
   distance from the crop center. Define the nominal crop side as the smallest
   extent of the stored crop. The primary seed must have arc length strictly
   greater than half that side; rank eligible fibers by greatest straightness,
   smallest center distance, greatest arc length, then lowest trace index, and
   assign the winner H. Reject an input with no eligible primary seed rather
   than silently weakening this rule. Degenerate lines are immediately broken
   and are neither seeded nor counted as growth steps. When no unassigned fiber
   has active assigned evidence, start the next disconnected component with
   the same ranking but without the primary length cutoff, so every valid
   disconnected fiber remains labelable.
4. For each unassigned fiber, collect constraints to already assigned H/V
   fibers. Its growth priority is `constraint_count / mean_closest_distance`
   (in base voxels), with infinite priority for zero mean distance. Select the
   maximum priority, then greater constraint count, smaller mean distance, and
   lower trace index. All distances must be finite and nonnegative. Zero mean
   distance has infinite priority, with the remaining ties unchanged.
   Constraints to broken fibers provide no orientation evidence and do not
   enter priority. This deliberately measures spatial/count connectivity, not
   parallel confidence, perpendicular confidence, or winding.
5. Evaluate the selected fiber as H and V using the existing orientation term:
   equal labels cost `1-parallel_score`, differing labels cost
   `parallel_score`, summed once per retained piece-pair constraint to active
   assigned fibers. Broken costs
   `broken_cost_per_link * current_active_evidence_count`. Choose minimum total
   cost with deterministic tie preference H, then V, then broken. This is an
   irreversible incremental objective: constraints to an already broken fiber
   stay disabled and are never charged later, and the report totals the costs
   selected at each step rather than rescoring the final graph. Winding values
   remain extraction diagnostics only. A valid fiber adjacent only to broken
   assignments is treated as a new active-evidence component and reseeded H.
6. Write final `<base>_h.obj`, `<base>_v.obj`, and `<base>_broken.obj` files
   containing complete original trace polylines. Write snapshots after 10, 20,
   ..., 100 additions and then 200, 300, ... additions as matching
   `<base>_step_N_{h,v,broken}.obj` triplets. Always write final outputs,
   including when the final count is also a snapshot. `N` counts all valid
   assignments, including the first seed, later component seeds, and broken
   choices; degenerate lines do not count. Normalize the basename by removing
   its extension, name OBJ elements `trace_N`, and write a valid empty file for
   any absent class. Degenerate input lines are not assignments and therefore
   are not included in the broken snapshots or final broken layer.
7. Report seed/component events including seed straightness, center distance,
   and arc length, detailed choice rows for the first 100 assignments, and
   final H/V/broken counts and objective components at the end of output,
   and each snapshot path. Add deterministic unit tests for source-trace
   aggregation, priority, assignment, disconnected reseeding, tie ordering,
   and exact snapshot milestones. Cover the primary length cutoff, center
   distance tie-break, multi-piece evidence multiplicity,
   broken-before-unassigned behavior, degenerate/no-evidence inputs, stable
   H/V/broken output contents, and identical results after a deterministic permutation of
   the input constraint vector. CLI validation must reject HiGHS-only flags in
   consensus mode.
8. Update specs, documentation, changelog, status, and task log. Build with
   `-j32`, run `test_fiberlet_crop_trace`, then exercise the centered-384 input
   with the default exclusive `1.5` winding cutoff.

## Spec update

Document the separate deterministic consensus command, trace-level aggregation,
priority and assignment formulas, disconnected-component behavior, and all
three output-class filenames/milestones.

## Documentation update

Add a runnable command and describe inputs, default cutoff, reported statistics,
final H/V/broken layers, and matching snapshots.

## Testing

Use small synthetic trace/constraint graphs with known selection and labels;
verify deterministic snapshots and unchanged existing solver tests.

## Changelog

Record the iterative trace-level H/V consensus diagnostic.
