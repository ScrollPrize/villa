# Root integration review

Frozen production snapshot: `69ba26311` (implementation commit `878525dd1`). This pass changes review records only. The preceding fixes were committed before review; no new production fix is silently folded into the reviewed snapshot.

## Traces and dispositions

- Rechecked the latest shared manifest decoder, full/incremental native fit-input comparison, Python selected-value facades and surface wrappers, and the position/reference field codec. Manifest references are validated before filtering; explicit numbered selection remains a boundary API rather than another internal handle. The shell refresh omission reported by the consumer lane remains a genuine sibling-path gap despite correct full reads.
- Rechecked `SurfaceGrid.cpp` and `ToolObjects.cpp` against their callers: transformed grids compare selected identity as well as payload; tool coordinate inference compares every selected source frame; artifact publication uses loaded references when available. Explicit new-output selection and deliberate unrepinned loaded tool state are not additional defects.
- Rechecked `PointMap.cpp`, public CMake linkage and the importer/GUI delegation: numeric aliases, full-consumption ID parsing and finite XYZ validation share one implementation. The file importer's whole-document collision policy remains appropriately separate from the point-map decoder.
- Rechecked native/Python reference constructors and extension siblings. The relationship extension ingress is the one confirmed decoder bypass reported by the core lane; ordinary native aggregate construction of already-selected references is not itself a schema bypass.
- Screened the Python location facade, Lasagna stored-surface/artifact/provenance users and Spiral change-feed classification. Client-local output and explicit current-object discovery are legitimate boundaries. Change-feed entries describe commits of the polled store; constructing references from that feed's store/lineage is not an inherited-owner substitution. No additional concrete defect established in these traces.

These are invariant/caller traces, not proof of every numerical algorithm, asynchronous interleaving or filesystem fault. Independent lane inventories and Astra adjudication are alongside this file. No production schema, UUID representation, history policy or placement behavior changed in this review.
