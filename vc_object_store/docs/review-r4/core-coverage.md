# R4 core inspection coverage

Baseline: `2b5d1e01fd26f27b3410d1b024c3f10a73533287`. Inspection only; no production fixes, installs, builds or runtime probes in this lane. Six individually logged candidates exist, not six validated bugs. C-001 has an explicit unresolved reachability objection and must not be counted as confirmed.

Scope: logical core C++ APIs, mutation/history/extension/query control flow, selected geometry/coordinate handling, session/draft behavior and shared internal wrappers. Physical storage/provisioning/collection copy, artifact storage, transport, converters, source formats, imports, detectors/linking and their APIs belong to the root's storage lane. Native/Python consumers belong to their independent lanes. Header entries are interface/inline-contract inspections; kernel entries were checked for storage/routing responsibilities and caller contracts, not exhaustively numerically proven. The ledger records architectural control-flow coverage, not a claim that every numerical algorithm or all possible interleavings were formally verified.

Existing regression scenarios were inspected as supporting evidence; the previous green suite is not proof of these newly reported scenarios. All candidate finding files state their evidence level and required reproduction.

## Dispositions

| File | Disposition |
| --- | --- |
| `vc_object_store/include/vc_object_store/geometry/ArrayBytes.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/geometry/BoundsIndex.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/geometry/Geometry.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/geometry/SegmentClip.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/geometry/SurfaceGrid.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/geometry/SurfaceIndex.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/geometry/TriangleDistance.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/geometry/VertexIndex.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/include/vc_object_store/objects/AttachedDocument.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/CatalogStore.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/Extensions.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/Geometry.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/GeometryCoordinates.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/GeometryEdits.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/GeometryRegion.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/ObjectStore.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/ObjectStoreSession.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/ResidentSurface.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/Selections.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/StoreReader.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/SurfaceCreate.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/SurfaceGrid.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/SurfaceTileCache.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/ToolObjects.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/TransactionJson.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/Transactions.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/VolumeTransforms.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/objects/VolumeUmbilicus.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/include/vc_object_store/store/GeometryViews.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/AttachedDocument.cpp` | C-001 (reachability unproven; not confirmed) |
| `vc_object_store/src/Commands.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/DraftCheckpoints.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/DraftCheckpoints.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/ExtensionHistory.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Extensions.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/GeometryRegion.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/History.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Lifecycle.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Metadata.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/ObjectStoreSession.cpp` | C-003, C-004 |
| `vc_object_store/src/Queries.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Records.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Selections.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Snapshots.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Spatial.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/StoreReader.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/SurfaceCreate.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/SurfaceDrafts.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/SurfaceGrid.cpp` | C-005 (selected-state overload) |
| `vc_object_store/src/SurfaceProximity.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/SurfaceTileCache.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/ToolObjects.cpp` | C-003; native lane owns discarded mapping guard sibling |
| `vc_object_store/src/Transactions.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/Versions.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/VolumeTransforms.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/VolumeUmbilicus.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/catalog/Catalog.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/catalog/CatalogStore.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/ArrayLayout.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/Dependencies.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/DependencyIndex.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/DependencyIndex.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/Geometry.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/GeometryDelta.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/GeometryDelta.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/GeometryEdits.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/GeometryInternal.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/GeometryReader.cpp` | C-006 nested materialization queries |
| `vc_object_store/src/geometry/GeometryReader.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/GeometryReference.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/GeometryReference.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/GeometrySnapshot.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/GeometryToken.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/GeometryViews.cpp` | C-005 |
| `vc_object_store/src/geometry/SequenceIdentity.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/SequenceMetadata.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/SequenceMetadata.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/SharedGeometry.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/SharedGeometry.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/SourceMetadata.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/geometry/SourceMetadata.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/geometry/StructuralGeometry.cpp` | C-002 |
| `vc_object_store/src/geometry/TypedGeometry.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/internal/Anchors.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/internal/Anchors.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/CoordinateFilter.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/EdgeSql.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/internal/EdgeSql.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/ExtensionHistory.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/Extensions.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/History.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/JsonFields.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/internal/JsonFields.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/Publication.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/ReadRequest.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/RecordFields.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/RecordJson.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/internal/RecordJson.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/RecordOverrides.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/internal/RecordOverrides.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/RecordResolver.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/RecordSql.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/internal/RecordSql.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/ReferenceResolver.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/Spatial.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/SpatialCache.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/SqlValues.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/Sqlite.cpp` | C-006 |
| `vc_object_store/src/internal/Sqlite.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/SurfaceDrafts.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/TransformedLength.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/Versions.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/VisibleRows.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/internal/VolumeFrames.hpp` | Public/internal interface and inline contract checked against implementation |
| `vc_object_store/src/relationships/EdgeGeometry.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/relationships/Edges.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/relationships/Traversal.cpp` | Logical store control flow inspected; no additional concrete finding |
| `vc_object_store/src/spatial/SurfaceGrid.cpp` | Format helper boundary; detailed source-I/O audit delegated to root |
| `vc_object_store/src/spatial/SurfaceIndex.cpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/src/spatial/SurfaceTiles.cpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/src/spatial/SurfaceTiles.hpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |
| `vc_object_store/src/spatial/VertexIndex.cpp` | Geometry/index kernel boundary; no physical store lookup or history-policy branch |

Count: **122 assigned production files dispositioned**. Delegated paths are not silently counted as clean core files.

## Deliberate non-findings and limitations

- Draft scale normalization to `{1,1}` is explicit in `SurfaceDrafts.hpp`; inspected recovery restores editable points into the source surface rather than claiming that scratch draft metadata is the original complete surface record. No new scale-loss finding without a caller that incorrectly adopts that normalized scale.
- Current-only selected versions cannot promise retained history after replacement. Immutable referenced stores cannot simply be edited in place; this is why C-001's initial trigger remains unproven.
- Raw byte APIs intentionally preserve non-finite bytes; typed geometry enforces a different validation contract. Do not re-report raw preservation as a validation failure.
- Deleted identities can be explicitly revived by supported record operations, continuing their version. Ingestion's separate rejection of tombstones is not proof that all revival is forbidden.
- Complete selected reads and write eligibility are separate concerns; C-005 requests keeping their identities distinct rather than permitting historical writes without current-state checks.
- No new schema, benchmark, full-volume, GUI runtime or macOS result is claimed. Independent-model review must adjudicate these candidates before acceptance.
