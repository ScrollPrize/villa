# Post-closure native core review

Frozen source: `de3afb065`; regression comparison: `70da58935`. No production changes, installs or runtime reproductions were made by this lane. The previous relationship-extension codec bypass is fixed by `b1f242597`; this pass deliberately inspected additional metadata/lifecycle, query/selection, structural edit, native export and discovery paths rather than treating the old search as exhaustive.

## PC-CORE-01 — binary selection position decoder bypasses strict shared commit-sequence validation

Priority: low boundary inconsistency; source `src/Selections.cpp:91` (`decodeSelectionRecord`). Astra independently confirmed the concrete selection-artifact failure at runtime: `1.5` decodes as `1`. The native selection-artifact decoder reads `header.commit_sequence` using `.get<int64_t>()`, while the shared `commitSequenceFrom` rejects booleans, fractions, strings, negative values and overflow. A syntactically valid VCSEL-v2 artifact whose JSON header has `commit_sequence: -1` decodes into a SelectionRecord with a negative snapshot position. The object-version binary fields remain strict unsigned integers; this finding does not allege those are broken.

Concrete source-level reproduction: create a SelectionRecord with valid store/generation UUIDs, query `{}`, empty objects/transforms and `commitSequence=-1`; `encodeSelectionRecord` writes the number unvalidated and `decodeSelectionRecord` accepts it. The public header documents that malformed records throw. The saved position sibling `store_position_from_json` rejects the same negative commit sequence. This is a native boundary policy mismatch, not a Python reimplementation.

The same decoder/encoder logic exists at `70da58935`, so this is an earlier omission, not a regression in the latest three closure fixes. No evidence here that normal `recordSelection` generates an invalid position or that invalid selections silently read wrong geometry: normal producers use native status, and later strict consumers can reject malformed positions. Native `encodeSelectionRecord`/`decodeSelectionRecord` and Python `selectionRecordFromPy`/`encode_selection_record`/`decode_selection_record` are the sibling ingress/serialization scope; the root binding lane and Astra own their detailed runtime evidence. A minimal fix would reuse the existing strict position codec, not invent another validation policy or storage format.

Additional source-traced sibling for adjudication (snapshot metadata/publication counters, not an alternate complete-reference representation): `src/internal/RecordJson.cpp::commonFromJson` reads snapshot `commit_sequence` and `version` with generic integer getters before range checks; it rejects negative positions but still coerces fractional values. `presentCommonFromJson` does generic numeric field adaptation for projected wire records. These should be considered under one scalar-ingress invariant rather than counted as a separate schema issue. No new runtime evidence for these RecordJson paths is claimed by this lane.

## Excluded candidates and legitimate special cases

- `ObjectCopy.cpp::copied_from` is an intentionally documented external provenance field. Earlier `reference-review/native.md` explicitly excluded this from internal reference transport; no runtime caller reconstructing an operational reference from it was found. Do not reopen it merely for naming/style or create another metadata migration.
- Geometry edit/repair numeric counters are low-level transaction-local decision projections. Command construction first establishes ordinary full record guards; their presence alone is not proof of a surviving consumer reference format.
- Local versus composed query plans in Queries/Records/Traversal belong to the storage layer. They are not forbidden consumer fallbacks; their output passes shared visibility/identity resolution.
- `SurfaceDrafts.hpp` explicitly documents draft-grid scale as `{1,1}`; this pass does not recast that old declared draft limitation as a new store bug without a failing promised consumer behavior.
- Shared payload hash names such as `@metadata`/`@annotations` are instrumentation buckets, not hash domains (`geometryPayloadDigest` hashes bytes only). Metadata closure reuse across those categories is not a hash-verification mismatch.

## Per-file disposition ledger

255 production files enumerated across native sources/public headers and bindings. SOURCE-TRACED means the named invariants and call chain were inspected, not a formal proof of all arithmetic or concurrency. DELEGATED entries link to the named sibling lane's evidence. Initially unassigned native entries have explicit complementary dispositions in root.md, formats.md and spatial-linking.md rather than being silently called clean. Pure numeric kernels and format-specific algorithms are not part of a schema/reference redesign.

| File | Disposition | Evidence / scope |
| --- | --- | --- |
| `include/vc_object_store/CtSheetNormal.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/MemMap.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/PackedTrackStore.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/Progress.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/TerminalProgress.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/Tiff.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/TrackCrossings.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/Uuid.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/client/HostedStore.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `include/vc_object_store/geometry/ArrayBytes.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/BoundsIndex.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/Geometry.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/PointMap.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/SegmentClip.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/SurfaceGrid.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/SurfaceIndex.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/TriangleDistance.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/geometry/VertexIndex.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/migration/ConvertStore.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/object_links/LinkObjects.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/object_links/ObjectLinks.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/AfvSource.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/Artifacts.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/objects/AttachedDocument.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/CatalogStore.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/Extensions.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/FiberGraphSolveExtension.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/FiberImportExtension.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/FiberJson.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/Geometry.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/GeometryCoordinates.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/GeometryEdits.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/GeometryExport.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/GeometryIngest.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/GeometryRegion.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/InitialImport.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/objects/ManifestReferences.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/ObjectCollection.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/objects/ObjectCopy.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/ObjectStore.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/ObjectStoreSession.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/PackedStore.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/objects/Project.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/objects/RelationshipInputs.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/ResidentSurface.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/Selections.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/StoreConversion.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/objects/StoreReader.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/SurfaceCreate.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/SurfaceGrid.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/SurfaceTileCache.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/ToolObjects.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/TransactionJson.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/Transactions.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/UmbilicusFile.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/VolumeTransforms.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/VolumeUmbilicus.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `include/vc_object_store/objects/imports/ParallelImportStream.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/objects/imports/ShardQueue.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `include/vc_object_store/server/Server.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `include/vc_object_store/store/GeometryViews.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `python/AfvBindings.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/CMakeLists.txt` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/ExtensionBindings.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/GeometryRegionBindings.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/ObjectReadBindings.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/SessionBindings.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/StorageBindings.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/TypedGeometryBindings.hpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/objects.cpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/vc_object_store/__init__.py` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/vc_object_store/derivation.py` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/vc_object_store/documents.py` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/vc_object_store/locations.py` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/vc_object_store/selected.py` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `python/vc_object_store/surfaces.py` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `src/Artifacts.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/AttachedDocument.cpp` | SOURCE-TRACED | attachedCollections/replaceAttachedCollections/writeCollection/writeVolumeCollections: loaded owner/membership/mapping guards and receipt references retained. |
| `src/CMakeLists.txt` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [root ledger](root.md); no fresh numerical/codec correctness proof claimed. |
| `src/Commands.cpp` | SOURCE-TRACED | expectObject/expectRelationship/expectRead/command edge edits: shared full references; geometry decision numeric counters are transaction-local projections after full guards. |
| `src/CtSheetNormal.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [root ledger](root.md); no fresh numerical/codec correctness proof claimed. |
| `src/Database.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/DraftCheckpoints.cpp` | DELEGATED | Astra storage/transport/checkpoint lane; no independent concurrency/closure proof claimed here. |
| `src/DraftCheckpoints.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/ExtensionHistory.cpp` | SOURCE-TRACED | owner-versioned replace/read snapshots: native schema+payload routing, full object/relationship lookup; no alternate Python policy. |
| `src/Extensions.cpp` | SOURCE-TRACED | create/append/scan definitions+rows: physical table layout and composed compatibility native; current owner snapshots use shared object resolver. |
| `src/FetchQueue.cpp` | DELEGATED | Astra storage/transport/checkpoint lane; no independent concurrency/closure proof claimed here. |
| `src/FetchQueue.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/GeometryRegion.cpp` | SOURCE-TRACED | RequestScope budgets all composed sources; VisibleObjectRows shadows before object limits; emitted current references and geometry read under one native read transaction. Raw clipping arithmetic not independently re-proved. |
| `src/History.cpp` | SOURCE-TRACED | resolveObjectVersion/currentObjectRevision/objectsAtVersions/relationshipsAtSelected/relationshipAtVersion/keepRecordedRelationship: current/exact and archive/current-only paths traced; typed complete identities preserved. |
| `src/HostedSession.cpp` | DELEGATED | Astra owns hosted cache/transport. Earlier selected/exact cache boundary traces are not claimed as a fresh full concurrency audit here. |
| `src/HostedSession.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/InitialImport.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/Lifecycle.cpp` | SOURCE-TRACED | retireObject/restoreObject/lifecycleBinding: source owner history and geometry descriptor restoration shared; checked publication/savepoints retained. |
| `src/MemMap.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [root ledger](root.md); no fresh numerical/codec correctness proof claimed. |
| `src/Metadata.cpp` | SOURCE-TRACED | mutateMeta/setApproval/bindGeometryCoordinates: compare before override, shared current/owner and extension-reserved-field policy; no defect found. |
| `src/ObjectCollection.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/ObjectCopy.cpp` | SOURCE-TRACED | readObjectCopy/writeObjectCopy: selected record+geometry and per-operation savepoint; copied_from is accepted external provenance, not an internal reference transport. |
| `src/ObjectStoreSession.cpp` | SOURCE-TRACED | selectObjects, current/exact readStore, draft installation/contents and artifact materialization boundaries traced; async/fetch/checkpoint internal races delegated to Astra, not independently re-proved. |
| `src/PackedCollection.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/PackedStore.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/PackedTrackStore.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/Project.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/Provisioning.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/Queries.cpp` | SOURCE-TRACED | selectObjects/spatialRegions/measuredSelection: composed visibility before global limits; selected geometry supplies length; physical-versus-composed query planning is legitimate. |
| `src/Records.cpp` | SOURCE-TRACED | upsert/delete/list/tag/count: visible resolver for composition; physical SQL remains inside storage owner; deletion keeps same-key marker. |
| `src/Selections.cpp` | SOURCE-TRACED | recordSelection/encodeSelectionRecord/decodeSelectionRecord: PC-CORE-01 boundary consistency candidate below. |
| `src/Snapshots.cpp` | SOURCE-TRACED / PC-A01 | exportJson/importJson: metadata-only restrictions and whole-operation savepoint retained, but Astra's independent tombstone counterexample demonstrates that live-only admission bypasses the shared reserved-identity policy. See [independent evidence](astra.md). |
| `src/SourcePaths.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/Spatial.cpp` | SOURCE-TRACED | inheritSpatialBounds/refreshGeometryBounds/indexGeometryBounds and region schema boundaries inspected: derived bounds materialized by storage layer, shared geometry identity used. Full geometric numeric/index correctness not independently re-proved. |
| `src/StoreConversion.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/StoreReader.cpp` | SOURCE-TRACED | selectedObject/decodeObjectReference/selectedInputs/expectations/transforms: complete references carried; explicit selectors are legitimate ingress. |
| `src/SurfaceCreate.cpp` | SOURCE-TRACED | surfaceRecord/createSurfaceOperations/createSurface: shared native construction; grouped mutation savepoint. |
| `src/SurfaceDrafts.cpp` | SOURCE-TRACED | makeSurfaceDraft/draftGrid/editSurfaceDraft/placeDraft: no store-owner reconstruction; existing explicit draft scale={1,1} limitation is documented, not newly changed. |
| `src/SurfaceGrid.cpp` | SOURCE-TRACED | readSurfaceGrid overloads: exact selected identity+payload matches transformed view; current selector delegates. |
| `src/SurfaceProximity.cpp` | SOURCE-TRACED | nearestSurfaces: VisibleObjectRows selects winners; shared geometry revision and frame mapping; no consumer placement branch. |
| `src/SurfaceTileCache.cpp` | SOURCE-TRACED | revision/tile immutable payload cache, not a metadata/frame cache; keys are geometry identity, not object-version substitutes. |
| `src/Tiff.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [root ledger](root.md); no fresh numerical/codec correctness proof claimed. |
| `src/ToolObjects.cpp` | SOURCE-TRACED | coordinatesFor/writable/publishArtifacts: loaded refs/mapping guards preserved; all source frames compared. |
| `src/TrackCrossings.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [root ledger](root.md); no fresh numerical/codec correctness proof claimed. |
| `src/Transactions.cpp` | SOURCE-TRACED | write/configureWrite/release: one checked write context, receipt before transaction release; owner routing setup shared. |
| `src/UmbilicusFile.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [root ledger](root.md); no fresh numerical/codec correctness proof claimed. |
| `src/Uuid.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [root ledger](root.md); no fresh numerical/codec correctness proof claimed. |
| `src/Versions.cpp` | SOURCE-TRACED | reference constructors/relationshipsAt/prepare checked writes/receipts: duplicate object selection rejected; tombstone/reference semantics shared. |
| `src/VolumeTransforms.cpp` | SOURCE-TRACED | checkVolumeMapping/volumeTransforms/chooseVolumeTransformRequest: full selected dependencies and mapping guards; path semantics belong to native catalog integration. |
| `src/VolumeUmbilicus.cpp` | SOURCE-TRACED | volumeUmbilicus/setVolumeUmbilicusRequest: selected full ref and snapshot commit guard before membership query. |
| `src/Wire.cpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `src/catalog/Catalog.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/catalog/CatalogStore.cpp` | DELEGATED | Astra storage/transport/checkpoint lane; no independent concurrency/closure proof claimed here. |
| `src/client/HostedStore.cpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `src/formats/FiberInput.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/FiberInput.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/formats/FiberJson.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/FiberMetadata.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/formats/GeometryInput.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/GeometryInput.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/formats/GeometryOutput.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/GeometryOutput.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/formats/PointCollectionInput.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/RemoteSource.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/RemoteSource.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/formats/SurfaceInput.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/SurfaceInput.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/formats/TrackInput.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/formats/TrackInput.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/formats/UmbilicusInput.cpp` | COMPLEMENT-REVIEWED | Store-boundary source disposition in [formats ledger](formats.md); no fresh numerical/codec correctness proof claimed. |
| `src/geometry/ArrayLayout.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/Dependencies.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/DependencyIndex.cpp` | SOURCE-TRACED | structuralEdgeCandidates: shared incident-edge resolution; native dependency subset semantics, not a per-object directory. |
| `src/geometry/DependencyIndex.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/Geometry.cpp` | SOURCE-TRACED | put/edit/prepareEdit/affectedEdges/validateEdgeEdits/publishStructuralSnapshot: shared selected geometry routing, operation savepoints, native endpoint-decision validation; source-traced, no new defect. |
| `src/geometry/GeometryDelta.cpp` | SOURCE-TRACED | load/decode delta: shared revision scope resolves immutable payload chain; not an alternate consumer resolver. |
| `src/geometry/GeometryDelta.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/GeometryEdits.cpp` | SOURCE-TRACED | updateGeometryArrays: one outer operation savepoint; stable IDs and annotation migration through shared structural edit path. |
| `src/geometry/GeometryExport.cpp` | SOURCE-TRACED | exportGeometry reader/local overloads: exact selected payload and owner topology, shared output protection, staged publication. |
| `src/geometry/GeometryIngest.cpp` | SOURCE-TRACED | registered/local source ingestion expectedAt/ingestBatch: source file preparation boundary remains separate from checked atomic native commit. |
| `src/geometry/GeometryIngestBatch.cpp` | SOURCE-TRACED | ingestGeometryBatch/addIngestOperation: canonical incoming UUID checks plus whole-batch savepoint, tombstone reservation, native metadata normalization. |
| `src/geometry/GeometryIngestBatch.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/GeometryInternal.hpp` | SOURCE-TRACED | payload hash name is instrumentation only; verified manifest hash shared; no domain mismatch in metadata copy. |
| `src/geometry/GeometryReader.cpp` | SOURCE-TRACED | source-screened/caller traced through geometry reference/payload reader; raw decode/range arithmetic not fully re-proved in this pass. |
| `src/geometry/GeometryReader.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/GeometryReference.cpp` | SOURCE-TRACED | resolveGeometryRecords/decodeGeometryRecord: archives/current descriptors resolved once in pinned scope; metadata consumers use ordinary geometry reader. |
| `src/geometry/GeometryReference.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/GeometrySnapshot.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/GeometryToken.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/GeometryViews.cpp` | SOURCE-TRACED | snapshot typed factories/selected views/transformed writes: full selected identity and retained mapping checked; geometry-only selectors confer no write authority. |
| `src/geometry/InitialGeometry.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/PointMap.cpp` | SOURCE-TRACED | pointIdFromString/decodePointMap: canonical strict finite numeric decode; shared importer/GUI caller previously migrated. |
| `src/geometry/SequenceIdentity.hpp` | SOURCE-TRACED | stable-ID array policy is shared by structural/edit/ingest consumers. |
| `src/geometry/SequenceMetadata.cpp` | SOURCE-TRACED | mergeRetainedAllocations/initialSequences/MetadataSnapshot: source-scope lookup at shared payload layer; no consumer row ordinal as stable ID. |
| `src/geometry/SequenceMetadata.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/SharedGeometry.cpp` | SOURCE-TRACED | payloadSource/readSharedPayload/storeSharedPayload/visitGeometryPayloads: common reference resolution and checksum verification. Hash name is metrics category, not hashing domain. |
| `src/geometry/SharedGeometry.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/SourceMetadata.cpp` | SOURCE-TRACED | nativeMetadata: export topology selected through references; destination native file IDs are format adaptation. |
| `src/geometry/SourceMetadata.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/geometry/StructuralGeometry.cpp` | SOURCE-TRACED | editGeometrySequence/resize/nextGeometryPointId: retained allocation namespace used; foreign source IDs cannot claim target provenance. |
| `src/geometry/TypedGeometry.cpp` | SOURCE-TRACED | encode/decode/validate owned types: no metadata-owner inference; explicit format conversion validates stable IDs and auxiliary-array preservation. |
| `src/imports/AfvRecords.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/imports/AfvSource.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/imports/FiberGraphSolveExtension.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/imports/FiberImportExtension.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/imports/SourceIdentityIndex.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/imports/SourceIdentityIndex.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/imports/SourceRelationships.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/imports/SourceRelationships.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/Anchors.cpp` | SOURCE-TRACED | canonicalizeEdge/anchorFromJson/dependenciesFromJson: one canonical endpoint and strict int64 point dependency codec. |
| `src/internal/Anchors.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/Artifacts.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/CollectionReads.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/CoordinateFilter.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/Database.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/DatabaseCopy.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/Digest.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/EdgeSql.cpp` | SOURCE-TRACED | read/write/normalize/selection: shared UUID, lifecycle, opposite-kind identity checks and tag insertion; no new defect. |
| `src/internal/EdgeSql.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/ExtensionHistory.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/Extensions.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/History.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/ImmutableImport.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/InitialEdges.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/InitialEdges.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/InitialObjects.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/InitialObjects.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/JsonFields.cpp` | SOURCE-TRACED | Shared bbox/scale adaptation; native geometry publication performs final shape/coordinate validation. No new concrete owner/reference defect. |
| `src/internal/JsonFields.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/ObjectVersionJson.hpp` | SOURCE-TRACED | reference/position/version number codecs: one strict ingress definition; used as PC-CORE-01 sibling. |
| `src/internal/Ownership.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/Ownership.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/Publication.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/ReadRequest.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/RecordFields.hpp` | SOURCE-TRACED | shared field visitor defines record projections; not an alternate reference representation. |
| `src/internal/RecordJson.cpp` | SOURCE-TRACED | shared object/edge wire and snapshot field interpretation; commonFromJson generic integer conversion is a PC-CORE-01 sibling candidate, not a new schema proposal. |
| `src/internal/RecordJson.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/RecordOverrides.cpp` | SOURCE-TRACED | createRecordOverride/prepareOverrideReferences: same-key before-state copy only for mutation; immutable geometry refs retained without payload copying; relocation shared. |
| `src/internal/RecordOverrides.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/RecordResolver.hpp` | SOURCE-TRACED | object/edge/query/incident/version bytes: all placement resolution native, tombstone precedence explicit. |
| `src/internal/RecordSql.cpp` | SOURCE-TRACED | readObjectRow/projectedObjectColumns/ExplicitRecordId plus metadata snapshot mutation boundaries checked; table/index SQL is appropriate physical storage ownership. Full ID allocator arithmetic not independently re-proved. |
| `src/internal/RecordSql.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/ReferenceResolver.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/RelationshipOrder.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/Schema.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/Schema.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/Spatial.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/SpatialCache.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/SqlValues.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/Sqlite.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/Sqlite.hpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/internal/SurfaceDrafts.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/TransformedLength.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/Versions.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/internal/VisibleRows.hpp` | SOURCE-TRACED | VisibleObjectRows: composed shadowing done before admitted cursor output. |
| `src/internal/VolumeFrames.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/linking/Detectors.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/linking/FiberSurface.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/linking/LineSurface.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/linking/LineSurface.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/linking/LinkCandidates.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/linking/LinkObjects.cpp` | SOURCE-TRACED | native detector config decode; no store placement policy imported into consumers. |
| `src/linking/PointCollectionSurface.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/linking/StoredGeometry.cpp` | SOURCE-TRACED | selected discovery stores full expectations; named geometry loader accepts selected record with named payload; pinned overlap caller passes selected metadata. |
| `src/linking/StoredGeometry.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/linking/SurfaceOverlap.cpp` | SOURCE-TRACED | materializePinned resolves full a/b references once and passes selected records+geometry to cached loader; current materialization deliberately uses live state. |
| `src/linking/TrackSurface.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/linking/TrackTrack.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/migration/ConvertStore.cpp` | DELEGATED | Astra storage/import/history-policy lane; boundary calls traced where listed above, full implementation not independently re-audited here. |
| `src/relationships/EdgeGeometry.cpp` | SOURCE-TRACED | repairEdgeGeometry/edgeGeometryState: shared normal edge update and recorded binding resolution; numeric repair counter is low-level transaction check. |
| `src/relationships/Edges.cpp` | SOURCE-TRACED | add/update/remove/select: common owner/override/endpoint identity and savepoints; no-op revalidation retained. |
| `src/relationships/RelationshipInputs.cpp` | SOURCE-TRACED | fittingRelationshipInput: one native family/order/eligibility policy consumed by Python and GUI. |
| `src/relationships/Traversal.cpp` | SOURCE-TRACED | current/exact/around graph and BFS: native storage query plans differ but share visibility and endpoint validation; full selected owner closure preserved. |
| `src/server/Server.cpp` | DELEGATED | Root binding/wire lane or Astra transport lane; no independent clean claim here. |
| `src/spatial/SurfaceGrid.cpp` | SOURCE-TRACED | readSurfaceGrid overloads: exact selected identity+payload matches transformed view; current selector delegates. |
| `src/spatial/SurfaceIndex.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/spatial/SurfaceTiles.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |
| `src/spatial/SurfaceTiles.hpp` | DECLARATION/HELPER | Enumerated; related core implementation traced where named. No claim of a fresh exhaustive helper/header proof. |
| `src/spatial/VertexIndex.cpp` | SOURCE-TRACED | Full store-boundary disposition and symbol/caller evidence in [spatial-linking supplement](spatial-linking.md); no numerical-algorithm proof claimed. |

## Verification and closure limits

No new tests run by this lane. The parent owns fixed approved-runner validation. Current/exact and history-on/off storage branches were source-traced in the shared object/relationship/geometry APIs; referenced/sharded/packed placement was followed to the shared resolution boundary, not independently runtime-exercised for each case. Concurrent hosted/cache timing and low-level SQLite/packed-I/O algorithms are delegated. There is no claim that an unchecked entry is bug-free or that the complete review is closed merely because a test suite passes.

