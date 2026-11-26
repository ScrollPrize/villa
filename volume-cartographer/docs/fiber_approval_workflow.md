# Fiber approval workflow

Store format: `vc_object_store/docs/fiber-approval.md`. Solver: [fiber_map_scaffold_solve.md](fiber_map_scaffold_solve.md).

## Principle

- Approval is a layer on top of a fiber's stored geometry and normals; the fiber itself is never changed.
- Per aspect (geometry, normals) a stretch of the line is undefined, approved or broken.
- Stretch ends are arbitrary positions on the line (indices into the stored line points, with a fraction), not
  control points.
- Editing the fiber makes the layer stale where the line points changed; the rest stays valid.
- "Enable editing" in the line annotation asks first when the fiber carries approvals; cancel keeps it read only.

## Fiber approval window

- The line annotation window in approval mode. It is opened from the fiber map: a double-click on a row of the
  approval list, or "Open in fiber approval" in a fiber's context menu. Both need a selected solve.
- The baseline normals are the selected solve's normals for that fiber. Without them the window does not open.
- Normal points are normal-only control points of the window:
  - added by placing a control point, or by a Shift-drag in the cross view (which also sets the normal);
  - deleted like control points;
  - stored in the approval layer, never on the fiber.
- At most one normal point between two line points: setting one there replaces the one already in that segment.
- Shown normal = the solve's normal + correction. The correction is interpolated between normal points only;
  the fiber's own control points do not end it.
- A stretch is selected by its two ends; the two combo boxes set its geometry and normals state. The overview
  bar underlines approved geometry green and broken stretches amber and dims everything outside the selection.
- Every change is stored at once as a new generation of the layer, together with the normals shown.

## Neighbour state of a crossing

- Every crossing of the open fiber with another fiber of the selected solve has a dropdown on the bar between
  the two strips, at the crossing's position: undefined, direct (the two fibers lie on neighbouring sheets) or
  distant (farther apart).
- Beside it the measured separation, unsigned, in windings: 0.5 is a direct neighbour, 1.5 one sheet over,
  2.5 two. This is a display convention; it does not say which fiber lies outside.
- The state is about the distance only. Which fiber lies outside is never part of it: that comes from the
  normals. "sign ?" marks a crossing whose normals do not say it.
- A change is stored at once (`ext_fiber_constraint_approval`), whatever the geometry's state.
- Where two dropdowns would overlap, the earlier one is shown; zooming in separates them.

## Solver

- Approved normals: the stretch is used with the normals that were shown, as axes; which way they face is solved.
- A normal point outside an approved stretch fixes the axis at its one position.
- Broken (either aspect): the stretch is removed from all relations.
- Broken geometry also splits the fiber there: each part takes its own winding. The per-fiber result columns
  hold the longest part.
- Stale stretches and normal points are not used; the run counts them.
- Manual normals on the fiber's own control points are held as axes outside approved stretches.
- A crossing stated direct: its separation is that of a direct neighbour, whatever was measured. Without a
  known side it states nothing.
- A crossing stated distant: a measurement that also says "not direct" is kept. Otherwise the separation is
  dropped (how far is not known). In both cases the side's bound moves one sheet out.
- A reviewed separation is heavy (weight 10), never exact: nothing is held rigid.

## Before the solve

- The stretches approved for normals are faced first. Crossings between approved stretches carry the facing, so
  connected stretches are one object; its sign is the one that agrees best with the full volume's signed normals.
- Approved normals that contradict each other around a loop fail the solve with a message naming the crossing.
- The fiber normals are traced out from the approved stretches. Between two stretches that face opposite ways
  the half turn is spread evenly over the gap.
- A fiber may break: continuation links (same winding, weight 10) lie at the ends of every approved stretch and
  about every 2,000 voxels between them, between constraints. An approved stretch never breaks.
- Sign constraints weigh 3, and 30 where an anchor lies on a stretch approved for normals.

## Dock

- One solver profile, "default scaffold".
- The solve list is sorted newest on top. Selecting a solve shows it and updates the map.
- A solve that finishes is selected.
- Closing the approval window with changes solves again; while a solve runs, once it has finished.
- The approval list holds stretches only. A stretch is not part of a solve: the list is derived from what the
  selected solve left violated and from what is approved now, so an approval changes it at once and a solve only
  changes the costs.
- A stretch ties constraints together (an approval at one constraint alone says nothing): from a violated
  constraint to at least one other constraint of the fiber, at least 500 voxels, 100 voxels past its outer
  constraints; close constraints are covered by one stretch. Its cost is every violated constraint on it.
- An approved stretch is fixed: it is listed exactly as approved and never cut or shifted. New stretches lie
  between approved ones and reach at most 100 voxels into one.
- Columns: geometry and normals approved, approved w. (crossings with an approved winding distance, of all), violated winding constraints
  of all, violated sign constraints of all, broken continuation links of all (each as X/Y), where, cost, cost per 1000 voxels, length, fiber.
- A stretch opens with its exact ends (double-click, or "Open this stretch" in the window's context menu).
