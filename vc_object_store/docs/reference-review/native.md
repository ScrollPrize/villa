# Native single-reference review

Baseline: `38c72d97222b3bc70a1a7477b6550c2a8eaec2aa` (checkout HEAD confirmed). Review-only; no production code or tests changed. All evidence below is source-traced, not runtime-verified. Scope is schema 21 native store, bindings/wire, external C++ GUI/CLI consumers; no model/solver audit. Candidates require independent adjudication and are not claims of reproduced incorrect results.

Owning representation: `ObjectVersion` in `objects/Transactions.hpp`; selected records/revisions expose `reference()`. Owning exact resolution: `resolveObjectVersion`, `ObjectStore::objectsAtVersions`, and `StoreReader::selectedObject`. Public language/wire references must use the shared native codec. Read and write counterparts were followed where listed below.

## Candidates

- [N-001](findings/N-001.md): session pins split/rebuild selected identity throughout tools, geometry edits and GUI state.
- [N-002](findings/N-002.md): native graph/history APIs convert complete references into internal JSON-array strings and re-resolve selected state.
- [N-003](findings/N-003.md): GUI history rows split/rebuild reference fields for both View and Restore.
- [N-004](findings/N-004.md): persisted/hosted receipts expose array-form references distinct from standard reference dictionaries.
- [N-005](findings/N-005.md): independently implemented reference codecs at native/persisted/Python boundaries.

## Caller and boundary coverage

| Files / symbols examined | Classification and result |
| --- | --- |
| `include/.../Transactions.hpp`, `ObjectStore.hpp`; `src/Versions.cpp`, `History.cpp`, `internal/RecordResolver.hpp`, `internal/History.hpp`, `internal/Versions.hpp` | Owners, complete-reference resolution and current/exact read/write guards. N-002/N-004. Native object records/revisions include selected metadata and publication fields; this review does not propose blindly replacing storage/archive IDs with public refs. |
| `src/Commands.cpp`, `Transactions.cpp` | Checked request/receipt serializers, restore source, expectObject/checkTransactionContext/history preconditions. Shared exact checking is present; codecs and internal history transport are N-002/N-004/N-005. |
| `src/Wire.cpp`, `include/.../TransactionJson.hpp` | Actual hosted boundary. Standard object reference dictionaries, relationship endpoint references and region object_reference delegate here. N-005 covers codec reuse, not existence of wire serialization. |
| `src/StoreReader.cpp`, `relationships/Traversal.cpp`, `client/HostedStore.cpp`, `HostedSession.cpp/.hpp`, `ObjectCollection.cpp` | Local/hosted/collection exact read and graph delegation use ObjectVersion vectors. Hosted cache textual keys are cache identity encodings, not alternate values reconstructed into exact references; no finding on key formatting. Lower graph conversion is N-002. |
| `src/ObjectStoreSession.cpp`, `ToolObjects.cpp`, `include/.../ObjectStoreSession.hpp` | Pin production, artifact publish/current guard, tool read retention/checking and draft state. N-001; draft metadata already has optional ObjectVersion beside its application metadata. |
| `src/DraftCheckpoints.cpp`, `Selections.cpp`, `include/.../Selections.hpp` | External persistence boundaries. Draft refs use the standard dictionary shape but duplicate codec; selection v1/v2 compact binary entries/origin dictionary are legitimate format adaptation. N-005, not a proposal to replace compact binary persistence with dictionaries. |
| `python/objects.cpp`, `StorageBindings.hpp`, `ObjectReadBindings.hpp`, `GeometryRegionBindings.hpp`, `ExtensionBindings.hpp`, `SessionBindings.hpp`, `TypedGeometryBindings.hpp` | Reference-bearing helpers and registrations searched/traced. Ordinary exact refs/overlaps/selection lists adapt through native request codec; region output does likewise. Extension exact input duplicates interpretation (N-005). Object/relationship revision dicts mirror selected records; receipt shape issue originates natively (N-004). Geometry-only `a_revision` repair selectors are geometry UUIDs, not incomplete object references. |
| `src/geometry/GeometryExport.cpp`, `geometry/SourceMetadata.cpp`, `formats/FiberJson.cpp`, `AttachedDocument.cpp` | Selected export/source/graph/document flows retain reference or selected revision. Shared relationship selection reaches N-002; no additional consumer-specific owner reconstruction identified here. |
| `src/Extensions.cpp`, `ExtensionHistory.cpp`, `Versions.cpp`, `ObjectCopy.cpp`, `migration/ConvertStore.cpp`, `StoreConversion.cpp` | Physical SQL/schema/migration and provenance boundaries. Binary identity/reference layouts, generation/commit records and copied_from provenance are not automatically competing public ObjectVersions. Existing wire arrays in receipts are N-004; SQL 16/56-byte reference storage remains permitted. No schema redesign proposed. |
| `volume-cartographer/apps/VC3D/SurfacePanelController.cpp`, `SurfaceCatalog.cpp/.hpp`, `segmentation/SurfaceStoreEdit.cpp/.hpp`, `SegmentationModule_Session.cpp`, `SegmentationModule.cpp`, segmentation/growth pin declarations | History row producer/View/Restore traced (N-003). Catalog accepts complete refs; edit repinning uses receipt-selected revision and pinOf (N-001). |
| `VC3D/LineAnnotationController.cpp/.hpp`, `LineAnnotationStoredFiber.hpp`, `FiberCollectionController.cpp/.hpp`, `PointCollectionSync.cpp/.hpp` | Current/exact selected fiber refs retained through reads/export; main nonstandard transport is shared PinnedObject. Direct unsaved aggregate constructors included in N-001. Point collection saved revision map uses ObjectVersion. |
| `VC3D/SpiralStoreInputs.cpp/.hpp`, `SpiralStoreEdits.cpp/.hpp`, `SpiralWorkspace.hpp`, `overlays/StoreSurfaceOverlayController.hpp` | Input/patch/committed/edit/conflict records retain ObjectVersion beside UUID and application data; UUID duplication alone is not a competing representation. No new finding. |
| `volume-cartographer/core/src/StoreSurfaces.cpp`; `VC3D/agent_bridge/AgentBridgeHandlers_stores.cpp`; `vc_object_store/apps/vc_object_export.cpp` | Core pin consumer reaches N-001. Agent export JSON is a legitimate external boundary (N-005 consolidation). CLI parses explicit selectors once and delegates selection through LocalStoreReader; its raw CLI argument variables are not an internal alternate-reference defect. |
| `VC3D/LasagnaServiceManager.cpp`; `core/src/lasagna` reference-related search | Service attachment store_uuid is attachment/workflow scope, not by itself an exact historical object reference. Model/solver state and external Python Lasagna transport are excluded from this native review and require their assigned consumer review. |

Coverage is explicit caller tracing and targeted repository searches, not a claim that every line in every listed file was audited for unrelated correctness. Tests and historic review reports were used only as context/validation targets, not proof. Unlisted native code without reference-bearing paths was not exhaustively inspected.

## Edge-case status and handoff

Local/referenced, history on/off, packed/sharded, current/exact, retained-after-override/deletion and unchanged receipt inputs are validation obligations in the candidate files. They were not executed in this review-only task. Read-only searches confirmed existing complete-reference APIs rather than introducing alternatives. Independent adjudication should separate consolidation requests from correctness regressions; all candidates are present in the frozen baseline.

No build command is required for these Markdown-only review artifacts. No numeric, persistence, runtime or portability behavior changed. Implementation must preserve archive compatibility, binary schema-21 layouts, optional history policy and distinct active-write/attachment guards.

## Subsequent implementation

Final integrated evidence is in [validation](validation.md); the entries below preserve the implementation-stage source account.

After independent acceptance, N-001–N-005 were implemented in the working tree. The baseline review above remains source-only evidence; runtime results must be recorded separately after the integrated suite finishes.

- Pins now retain `ObjectVersion selected`; `reference()` returns that value unchanged. Attachment/write-generation/geometry guards remain separate. Unsaved fiber pins retain their existing explicit zero-version state and `stored()` check. Line annotation stale checks now compare full references. Native queued displacement jobs also retain ObjectVersion until request serialization.
- History now uses `ObjectRevisionKey = variant<archive UUID, ObjectVersion>` only where genuine schema archive identities and full exact references are both possible. The internal array-string encoder was deleted. Legacy array decoding remains only at old receipt/revision input boundaries. Binary SQL fields remain 16/56 bytes. Selected relationship reads retain selected revisions, compare endpoint ObjectVersions, and no longer go through string-valued graph overloads; unused private graph/relationship overloads and their unused shim were deleted. Redundant native endpoint version counters were removed; legacy wire fields derive from the complete reference at serialization.
- Surface history View and Restore use one typed item-to-reference map. Project umbilicus JSON now wraps the standard reference next to points; the Spiral UI summary and producer regression were migrated together.
- New receipts emit standard reference dictionaries; old arrays remain readable only through the receipt decoder. Missing complete references still require reload. Shared native field codecs serve wire, restore commands, draft checkpoints, selection transforms and extension reads. Existing permissive draft/command/selection enclosing-format validation is retained explicitly; the new public scalar input decoder is strict about standard keys, positive integral version and valid canonical UUIDs.

Added regression assertions in `test_object_store_versions` cover standard scalar roundtrip, fractional/zero version and malformed owner rejection, new receipt dictionaries, legacy array receipt decode, missing-reference reload, and invalid native receipt encoding. Existing pin/history/hosted tests were migrated from removed fields to complete reference assertions. Exact build command used: `cmake --build local/vcobj-pr-tests/build --parallel 8`; integrated approved runner remains `bash local/scripts/run_vcobj_pr_tests`. No completed suite result is implied by these pending steps.
