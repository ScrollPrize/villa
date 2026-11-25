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
- Shown normal = the solve's normal + correction. The correction is interpolated between normal points only;
  the fiber's own control points do not end it.
- A stretch is selected by its two ends; the two combo boxes set its geometry and normals state. The overview
  bar underlines approved geometry green and broken stretches amber and dims everything outside the selection.
- Every change is stored at once as a new generation of the layer, together with the normals shown.

## Solver

- Approved normals: the stretch is used with the normals that were shown, as axes; which way they face is solved.
- A normal point outside an approved stretch fixes the axis at its one position.
- Broken (either aspect): the stretch is removed from all relations.
- Broken geometry also splits the fiber there: each part takes its own winding. The per-fiber result columns
  hold the longest part.
- Stale stretches and normal points are not used; the run counts them.
- Manual normals on the fiber's own control points are held as axes outside approved stretches.

## Dock

- One solver profile, "default scaffold".
- The solve list is sorted newest on top. Selecting a solve shows it and updates the map.
- A solve that finishes is selected.
- Closing the approval window with changes solves again; while a solve runs, once it has finished.
