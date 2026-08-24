# Task: lossless post-stage-two Fiberlet graph simplification

After the centered stage-two route reduction, simplify the retained graph
without changing its valid entry-to-first-exit routes or objective values.

- Remove directed states proven unable to lie on an entry-to-exit route by
  conservative forward/reverse reachability, while retaining uncertain states.
- Remove every anchor unused by the remaining graph.
- Detect zero, one, and multiple admissible continuations using the regular
  directed join constraints.
- Merge maximal consecutive physical Fiberlets across interior anchors where
  the continuation is unambiguous in both directions.
- Precompute deterministic directed continuations where only one successor is
  available, including cases that cannot be represented as one undirected
  physical merge.
- Preserve stage-two boundary semantics through explicit boundary portals.
- Validate that exact same-endpoint duplicates are absent. Do not remove
  distinct higher-cost routes merely as dominated because that would change
  the valid route set.
- Report before/after graph populations and contraction distributions for the
  centered stage-two crop.

Macro-Fiberlets must reference their original directed Fiberlet sequence,
retain an explicit live-direction mask, and preserve the ordered edge costs,
join costs, lengths, geometry identity, and visited anchors. Applying a macro
must atomically validate every hidden anchor. Do not approximate a concatenated
route by writing it as an ordinary single-Fiberlet lattice route.
