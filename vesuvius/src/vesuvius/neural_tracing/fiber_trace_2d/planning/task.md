# Task: durable Fiberlet crop trace artifacts

Make crop tracing produce a durable, reusable Fiberlet Zarr artifact before
any visualization or downstream processing.

- Store accepted traced paths and their global metric cost in the existing
  Fiberlet sparse-Zarr envelope rather than treating OBJ as authoritative.
- Preserve traced geometry exactly in base-volume XYZ; do not force complete
  crop traces through the short Fiberlet transverse-lattice route codec.
- Generate all line, seed-anchor, direction-group, and quality OBJ artifacts
  by reopening the stored trace dataset.
- Report a cost-density histogram and write each rank decile as a separate OBJ
  so low- and high-quality trace populations can be inspected independently.
- Provide an explicit visualization-only CLI path for regenerating OBJ output
  from an existing trace dataset.
