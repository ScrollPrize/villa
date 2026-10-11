# Joint full-pass optimization and reference metrics

Replace random event sampling with seeded permutations traversed exactly once
per pass. Joint Adam updates remain internally batched for memory: logits,
anchors, offsets and widths all train. Partition a node permutation over the
same updates (each node once); evaluate all continuation links each update.
Report completed full passes, not sampled-pass equivalents. Reject active
oversampling in full-pass mode. Keep the legacy sampled mode explicit for replay.

Reference evaluation is read-only, using the existing validated projection cache.
Aggregate retained pieces by original source fiber, with length-weighted median
fitted winding; compare with its existing median reference winding. One global
median difference minimizes fiber-weighted L1. Report L1 and inclusive ±2 percent,
matched/retained/total counts, offset and exclusions. Never feed this into losses.

## Spec update
Full passes are the default joint traversal. Internal batches are optimizer
updates, not reported iterations. No upstream sign/HV rerun or geometry freezing.

## Docs updates
Document traversal and metric denominators, validation and limitations in task
log and changelog. Add regression tests for coverage and offset-invariant metrics.

## Tests
Exact event/node coverage including tails, reproducibility, known offset/outlier
metrics, rejected/unmatched fibers, and existing solver/reference tests.
