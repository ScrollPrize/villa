# Task plan
Reuse the existing fiber_uv_layout.split_bounds eligible-midpoint splitter: length1000 voxels and minimum4 perpendicular observations, matching recorded villa2 split settings. No continuation penalty (recorded setting0). Independent integer variable per piece; all pieces retained. Same measurement weights, fractional calibration and exact full-fiber/connector seam corrections as accepted baseline.

Cases: free, joint with GT sources protected from splitting, and third joint variant with GT allowed to split (the flow analogue of selectable GT; no disabling). Clarification sent to user; proceed on these stated assumptions absent correction. Use third case's original saved config/tensors after checking equality/provenance. GT identity derived from AFV metadata, never labels. Keep original full-fiber data/Paths immutable and remap only graph endpoint indices to pieces; each piece label uses the original fiber-start chart, so existing transport applies without a new artificial seam jump.

Reuse shared scorer by adding optional piece lookup by source and original arc. Reference remains unchanged and unsplit; use the predicted piece containing the GT cross-event endpoint. Report coverage, components/gauges, single global gauge diagnostics and 100% retained length. No GT-derived splitting or exclusions. Save piece bounds, original parents, endpoint mapping, graph/certificate, manifest hashes and all three results separately; preserve accepted baseline. Report both original measurement sign errors and requested GT signs clearly separately.

Test splitting eligibility/protection, reversed arc directions, exact-cut ownership, endpoint mapping, fractional graph with seam transport, and piece-aware reference lookup; regression and same-data unsplit equivalence. Run three cases, compare old free/joint values, inspect remaining violations and any newly disconnected pieces. Update docs and planning.

## spec update
Allow opt-in independent pieces with original-source/arc identity; no disabling. Piece lookup required for evaluation. Default whole-fiber baseline unchanged.

## docs updates
Add split experiment report, commands, settings, three-case semantics, count tables, gauge limitations, timings and provenance. Update status/task_log/changelog.
