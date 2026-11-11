# Post-closure consumer review

Frozen source: `de3afb06543cef8088491da7b310652851033139`; previous review/fix baseline `70da58935`. This lane changed documentation only. No production changes, dependency installs, conflicting test runners or dataset mutations. Contracts read: `review-contract.md`, `versioned-reference-contract.md`, `reading.md`, and `coherent-store-api-cleanup.md#contract`.

## Result

No new concrete consumer defect established in the bounded inventory below. The latest closure fixes do not reintroduce numeric guards, alternate reference representations or physical-placement handling in the inspected consumer paths. This is a source-review result, not a proof of absence of all bugs and not a new runtime/platform claim. Astra independently reviews this result and any findings from the other lanes.

### Latest implementation delta

- **Outer shell:** `SpiralStoreInputs::objectPins` now enumerates shell and ordinary objects with the application role beside the same complete reference. `manifestPins`, full reads and incremental reads all use it. `readObject` routes Missing/Edited/MetadataChanged shell state to the dedicated slot. `applyFitInputChanges`, `count` and `unchanged` include it. `SpiralWorkspace::_fitRecords` now admits its notifications, and both initial/incremental display paths call `fitSurfaceChanges` with the same shell category/key. Missing results generate the matching removal key. Full and incremental problems include the shell. There is no separate physical shell resolver.
- **Shell endpoint separation:** relationship endpoint-kind construction and incremental incident queries explicitly exclude shell-role pins. That is the application's fitting-selection rule, not an identity exception; native `fittingRelationshipInput` still owns family and eligibility policy. An empty relationship scope remains an explicitly empty optional edge-UUID selection, not unrestricted discovery.
- **Neural provenance:** `_StoredSurfaces.save` now carries `parameters.input.reference = self.loaded_version`; it neither flattens the reference nor keeps the redundant `surface` field. Source UUID lists and selected geometry loading remain unchanged. No old-format reader was added.
- **Relationship extension ingress:** belongs to the independent core/bindings lane. This consumer lane did not duplicate or bypass its shared decoder.

## Inventory method and evidence levels

Started from all paths in `../final-mismatches/consumer-inventory.md` and followed siblings/wrappers. Direct discovery: `rg -l 'vc_object_store|vc::objects|ObjectStoreSession|ObjectRevision|ObjectVersion' volume-cartographer --glob '*.{cpp,hpp,h}' --glob '!**/test/**' --glob '!**/tests/**' --glob '!**/build/**'` (109 files) and `rg -l 'vc_object_store|stored_surfaces|object_store_inputs|surface_store' spiral-fitting lasagna vesuvius/src --glob '*.py' --glob '!**/tests/**' --glob '!**/test_*.py'`. Includes and incidental substrings are explicitly disposed below rather than counted as full store implementations. Indirect callers from the seed remain included even if those keywords do not match.

Disposition: **D** delegates store policy to canonical APIs; **A** legitimate application/boundary adaptation; **H** declaration/header corresponding to the named implementation, no second resolver; **X** irrelevant match or independent source format, excluded with reason. These are per-file/symbol source dispositions. “No candidate” is not a runtime assertion. Storage internals, codecs, bindings and hosted transport are covered by the separate core lane. Geometry/rendering/model numerical algorithms outside their store boundaries are excluded.

## Native GUI consumer ledger

Paths in this table are relative to `volume-cartographer/apps/VC3D/`.

| File | Symbols / disposition |
| --- | --- |
| `CState.cpp` | D/A — attach/storeFor/store routes, event polling and draft updates use ObjectStoreSession; attachment identity is application routing, not selected-object ownership. |
| `CState.hpp` | H — session/routes/draft declarations, shared attachment/reference types. |
| `CWindow.cpp` | D/A — controller/session wiring and project routing; no local schema reader. |
| `CWindow.hpp` | H — controller ownership/session wiring declarations. |
| `CWindowContextMenu.cpp` | D/A — context actions delegate surface/fiber/segmentation controllers. |
| `CommandLineToolRunner.cpp` | D/A — store CLI configuration and exportedObjCopy use native session artifact materialization; no database copying/resolution fallback. Explicit current output lookup remains output selection, not a loaded-input pin. |
| `FiberCollectionController.cpp` | D/A — region queries, selected metadata/geometry and graphReadViewAround retain complete selected identities; batched immutable payloads retain mappings and endpoint evidence separately. |
| `FiberCollectionController.hpp` | H — selectedPin/scene/read scheduling declarations. |
| `FiberMapStaleness.hpp` | A — application dependency counters plus complete umbilicus reference; counters are not object write guards. |
| `LasagnaServiceManager.cpp` | D/A — setObjectStore, service request/reference transport, source-model artifacts via session; path/UUID binding identifies service attachment, not source version. |
| `LasagnaServiceManager.hpp` | H — service/attachment state and request declarations. |
| `LineAnnotationController.cpp` | D/A — selected fiber loads, graph baselines, mapped geometry, structural point IDs, guarded/durable saves and receipt repinning use shared APIs; UI runtime IDs remain distinct from persisted references. Mapping/membership retained for deferred edits. |
| `LineAnnotationController.hpp` | H — loaded pins, mapping/graph state, deferred-edit declarations. |
| `LineAnnotationStoredFiber.hpp` | A — UI fiber/branch/control conversion around selected data; no placement resolver. |
| `LineAnnotationStructuralEdits.cpp` | D/A — merges/splits operate on loaded fiber state; shared geometryCoordinates comparison/checked save path, not a new version format. |
| `MenuActionController.cpp` | D/A — create/attach/import/export/session actions delegate native store/closure APIs and controllers; project resource operations are application-level. |
| `ObjectStoreArtifacts.hpp` | A — immutable JSON transport cache for already-built application documents, UUID validation and shared atomic file writer; not vcobj export/copy. |
| `ObjectStoreRegions.hpp` | A — generic GUI background scheduling and slab request state; native region APIs own spatial semantics/budgets. |
| `OpenDataSampleProject.cpp` | D/A — sample project/store creation and native catalog import; no per-object physical resolver. |
| `OpenDataSampleProject.hpp` | H — sample-project creation/import declarations. |
| `PointCollectionSync.cpp` | D/A — volumeCollections/writeVolumeCollections, selected references and mapping retained through flush; no-op pending-state check does not manufacture a new baseline. |
| `PointCollectionSync.hpp` | H — selected collection and pending UI state. |
| `ProjectResources.cpp` | D/A — plan/import/retire umbilici use common binding, volume selection and checked current-edit APIs; file/volume-name selection is user-boundary configuration. |
| `ProjectResources.hpp` | H — import plan and annotation selection types. |
| `ProjectStoreCounts.hpp` | A — asynchronous aggregate counts keyed by attachment UUID/instance generation; not versioned-object identity/cache. |
| `ProjectWorkspace.cpp` | D/A — store management, counts, prefetch/status and copy operations through session; local/hosted UI location display is legitimate transport configuration. |
| `ProjectWorkspace.hpp` | H — project/store workspace state. |
| `SegmentationCommandHandler.cpp` | D/A — dispatches segmentation/editor operations; no independent vcobj resolution. |
| `SegmentationCommandHandler.hpp` | H — command-handler declarations. |
| `SessionDrafts.cpp` | D/A — useDraftDirectory/checkpointDrafts/resolvePending shared; saved request identity preserved on uncertain outcomes. |
| `SessionDrafts.hpp` | H — draft maintenance state; attachment is request routing, not selected ownership. |
| `SpiralPointCollectionEdit.cpp` | A — detached source-document editing and shared reference decode/encode. Source uint64 UI keys/renumbering are not the narrower vcobj int64 import schema. |
| `SpiralPointCollectionEdit.hpp` | H/A — editable draft retains shared reference and application source/revision state. |
| `SpiralProjectInputs.cpp` | D/A — volume facts/Lasagna catalog metadata plus native volumeUmbilicus selected reference; dataset identities are not object version counters. |
| `SpiralProjectInputs.hpp` | H — project dataset/volume input declarations. |
| `SpiralServiceManager.cpp` | D/A — service attachment binding, positions, receipt/reload protocol and change-feed queries; full-reference object edits remain separate. |
| `SpiralServiceManager.hpp` | H/A — service binding/position/update state, not a replacement object handle. |
| `SpiralStoreEdits.cpp` | D/A — saveBrushPatches/point collection edits guard loaded references; conflict reporting may read current for the user message, not replace the submitted baseline. |
| `SpiralStoreEdits.hpp` | H — brush/pcl save records carry ObjectVersion. |
| `SpiralStoreInputs.cpp` | D/A — manifestReferences, full/incremental role-aware object enumeration, native geometry views and fittingRelationshipInput. Latest shell chain checked above. |
| `SpiralStoreInputs.hpp` | H/A — application pin contains unchanged ObjectVersion plus role; shell included in count/unchanged. |
| `SpiralStoredFits.cpp` | D/A — explicit current/list selection and objectAtGeometry retained metadata selection delegate StoreReader; channel payload reads are immutable. Geometry-only selector is not claimed to be an exact metadata selector. |
| `SpiralStoredFits.hpp` | H — committed fit display/materialization results. |
| `SpiralWorkspace.cpp` | D/A — native full/incremental input capture, role-aware shared display builder, full-reference drafts/updates and app overlay installation; no local placement rules. |
| `SpiralWorkspace.hpp` | H/A — retains shared pins with draft and asynchronous GUI state; no alternate reference tuple. |
| `StoreImportController.cpp` | D/A — prepare/start native ingest; UUID namespaces and attachment/kind routing are explicit input configuration. |
| `StoreImportController.hpp` | H — ingest request/progress/result types. |
| `SurfaceCatalog.cpp` | D/A — open current/exact selected surfaces, retained mapping/pin and shared regions/resident surfaces; history policy delegated to selected owner. |
| `SurfaceCatalog.hpp` | H — selected catalog/opened surface records. |
| `SurfacePanelController.cpp` | D/A — history selections/full references, explicit restore/current edits and loaded edit preparation; native codecs/store APIs own identity. |
| `SurfacePanelController.hpp` | H — selected/opened history and editor declarations. |
| `VolumeRegistrations.cpp` | A — parses user-entered matrix syntax then native VolumeAffine validates invertibility/homogeneous row. |
| `VolumeRegistrations.hpp` | H — matrix boundary parser declaration. |
| `viewer_controls/panels/VolumeRegistrationsPanel.cpp` | D/A — registration UI operates on canonical native records/checked edits, not local SQL. |
| `viewer_controls/panels/VolumeRegistrationsPanel.hpp` | H — registration UI state. |
| `agent_bridge/AgentBridgeHandlers_session.cpp` | D/A — delegates project/session/controller actions; no independent selected-object storage resolver. |
| `agent_bridge/AgentBridgeHandlers_stores.cpp` | D/A — import/export/current discovery through shared session; export selects one full revision in read callback and native exportGeometry. JSON result is an external response, not an internal alternate handle. |
| `agent_bridge/AgentBridgeInternal.hpp` | H/A — agent transport/error helpers, no store schema implementation. |
| `agent_bridge/AgentBridgeServer.hpp` | H — session/job handler declarations and lifetime state. |
| `fibers3d/Fiber3DData.cpp` | D/A — selectedObject(record.reference), shared transformed view, geometry-derived counts and native graph; display packaging is app policy. |
| `fibers3d/Fiber3DData.hpp` | H/A — scene source/data structures, shared references retained where needed. |
| `fibers3d/Fiber3DPanel.cpp` | D/A — store/transform events invalidate/requery the scene; no schema handling. |
| `overlays/InkDetectionOverlayController.cpp` | D/A — catalogReference/fetchPublishedFile parse published catalog metadata, not fresh current object metadata substituted for displayed data. |
| `overlays/InkDetectionOverlayController.hpp` | H — catalog file/display selection state. |
| `overlays/SegmentationOverlayController.cpp` | A — displayed segmentation overlay, no independent vcobj placement/version implementation. |
| `overlays/StoreSurfaceOverlayController.cpp` | D/A — shared region/resident surface paths retain visible references; surfaceChanged clears resident geometry caches for in-place edits. Fetch uses session ensureGeometry. |
| `overlays/StoreSurfaceOverlayController.hpp` | H/A — display runs retain ObjectVersion; resident geometry/cache scheduling state is app-local. |
| `segmentation/SegmentationModule.cpp` | D/A — surface editor and catalog/controller integration; no duplicate physical store query. |
| `segmentation/SegmentationModule.hpp` | H — editor/pinned surface state. |
| `segmentation/SegmentationModule_Session.cpp` | D/A — loaded editor/shared pins, drafts/autosave and growth annotations; grouped interaction state deliberately separate from store mutation semantics. |
| `segmentation/SurfaceCorrections.cpp` | D/A — attachedCollections/replaceAttachedCollections and selected editor guard; standalone source-collection IDs are application metadata. |
| `segmentation/SurfaceCorrections.hpp` | H/A — correction snapshot retains native baseline, growth payload metadata. |
| `segmentation/SurfaceStoreEdit.cpp` | D/A — loaded PinnedObject and membership/mapping baseline, checked/durable save, retained receipt repin. Coordinate utilities use pinned payload plus explicit native write/mapping guards; no ownership inferred from attachment. |
| `segmentation/SurfaceStoreEdit.hpp` | H — shared editor state/locking; pending request UUID retained. |
| `segmentation/growth/SegmentationGrower.cpp` | D/A — copy job carries ObjectVersion, external socket serializes native codec; grower saves through common loaded editor. Geometry algorithm not re-audited. |
| `segmentation/growth/SegmentationGrower.hpp` | H — pending grower state retains before pin/selected surface data. |
| `segmentation/panels/SegmentationLasagnaPanel.cpp` | D/A — fitting request uses controller/session native reference paths; service transport is boundary adaptation. |
| `segmentation/tools/SegmentationEditManager.hpp` | H/A — pending GUI edits grouped before native checked save; no separate store versioning implementation. |

## Native core and CLI consumer ledger

Paths relative to `volume-cartographer/`. These are consumers of `vc_object_store`, not its internal implementation.

| File | Symbols / disposition |
| --- | --- |
| `apps/src/FiberConstraintExtraction.cpp` | D/A — one store read snapshot for status/selection/transforms/payloads; immutable payload batches and whole-source snapshot evidence. Independent constraint SQLite output is not vcobj runtime SQL. |
| `apps/src/FiberConstraintGeometry.hpp` | D/A — selectFiberConstraintGeometry reads current geometry through store within caller's pinned snapshot, not ingestion counts. |
| `apps/src/ObjAlphaCompRefinement.cpp` | D — indirect StoreSurfaces load/create/update/artifact path; algorithm and source CLI not a store resolver. |
| `apps/src/vc_gen_umbilicus.cpp` | D/A — shared umbilicus file/binding/import APIs; explicit CLI selection boundary. |
| `apps/src/vc_grow_seg_from_seed.cpp` | D/A — StoreSurfaces/ToolObjects loaded source and output/artifact workflow; native reader callback selection. |
| `apps/src/vc_grow_seg_from_segments.cpp` | D/A — StoreSurfaces source/target/geometry and output publication, native reader listing; dataset/source-directory mode independent. |
| `apps/src/vc_obj2tifxyz.cpp` | D/A — StoreSurfaces source/derived output, native coordinate provenance through ToolObjects. OBJ/tifxyz source parsing is not schema fallback. |
| `apps/src/vc_obj2tifxyz_legacy.cpp` | D/A — legacy-named OBJ algorithm uses current StoreSurfaces for optional store output; no old vcobj schema support. |
| `apps/src/vc_tifxyz2obj.cpp` | D/A — shared store surface/artifact API; separate tifxyz external-file input remains valid. |
| `apps/src/vc_render_tifxyz.cpp` | D/A — StoreToolOptions/StoreSurfaces selected loading; rendering and output-volume algorithms excluded. |
| `core/include/vc/core/PointCollections.hpp` | H/A — standalone UI/source collections and UUID fields; not persistent vcobj table schema. |
| `core/include/vc/core/StoredCollection.hpp` | H — native attached-collection conversion declarations. |
| `core/include/vc/core/types/ProjectCreation.hpp` | H — project creation uses OpenOptions, no store internals. |
| `core/include/vc/core/types/VolumePkg.hpp` | H/A — project routes/resource declarations around ObjectStoreSession, not per-object physical routing. |
| `core/include/vc/core/util/QuadSurface.hpp` | H/A — native selected surface conversion/channel loader declarations, no duplicate resolution implementation. |
| `core/include/vc/core/util/StoreSurfaces.hpp` | H — ToolObjects/session-backed surface API. |
| `core/include/vc/core/util/StoreToolOptions.hpp` | A — one CLI option adapter shared by tools; native store layer interprets object identity/placement. |
| `core/include/vc/core/util/SurfaceArea.hpp` | A/X — area metadata arithmetic only; no version/read/write policy. |
| `core/include/vc/core/util/SurfaceQuerySet.hpp` | D/A — application surface identity around native resident SurfaceIndex; no vcobj owner/version inference. |
| `core/include/vc/fiber_tracer/FiberJson.hpp` | A/X — independent vc3d_fiber v4 transfer parser uses UUID validator; its source-format version is not vcobj schema21 compatibility. |
| `core/src/FiberView.cpp` | A/X — resident view geometry/selection operations; no persistent store resolver. |
| `core/src/GridStore.cpp` | X — reuse of memory-map utility only; raw grid storage is not a vcobj database. |
| `core/src/GrowSurface.cpp` | A/X — surface geometry/growth uses common validity helpers; no selected object resolver. |
| `core/src/PointCollections.cpp` | A/X — standalone PointCollections source JSON/UI UUIDs, not store schema handling. Narrowing uint64 source IDs to native int64 implicitly would be wrong. |
| `core/src/ProjectCreation.cpp` | D/A — create/open native stores then project document; no backing-database copy bypass. |
| `core/src/ProjectDocument.cpp` | A — project-file resource/location/route schema and UUIDs; not per-object store placement/history. |
| `core/src/QuadSurface.cpp` | A/X — external surface load/save/new UUID using shared UUID API; no old vcobj schema path. |
| `core/src/QuadSurfaceStore.cpp` | D/A — readSurfaceGrid from selected revision, immutable channel reads, shared geometry encoding/coordinate utilities; no consumer SQL. |
| `core/src/StoreSurfaces.cpp` | D/A — load selects once/pinOf; create uses native surface ops; update uses loaded ToolObjects guards including no-op submit. |
| `core/src/StoredCollection.cpp` | D/A — native decodePointMap validates point IDs/XYZ/annotations once; remaining JSON construction is external UI transport. |
| `core/src/fiber_tracer/FiberSideStripIntersections.cpp` | D/A — common geometry intersection helpers; no alternate store placement implementation. |
| `libs/flatboi/flatboi.cpp` | D/A — optional StoreToolOptions/StoreSurfaces import/output/artifact boundaries; flattening/W&B logic excluded. |
| `scripts/benchmark_cleanup.cpp` | X — shared geometry cleanup helper benchmark, not a persisted vcobj consumer; no schema/reference handling to consolidate. |

## Python consumer ledger

| File | Symbols / disposition |
| --- | --- |
| `lasagna/stored_surfaces.py` | D/A — StoreSession/read_surface/StoredSurface carry native selected reference; current/new-output artifact publication intentionally selects output at that point. |
| `lasagna/fit_service.py` | D/A — request decoding delegates StoreSession/native boundary; model artifact retrieval shares selected reference with grid inputs. Service type/path dispatch is not native schema validation. |
| `lasagna/fit2tifxyz.py` | D/A — create_surface, attach_document, selected input provenance, artifacts; shared scalar codec validates saved input refs. |
| `lasagna/tifxyz_io.py` | D/A — explicit stored-surface intent delegates stored_surfaces; external tifxyz path mode is independent source format. |
| `lasagna/approval_inpaint.py` | D/A — selected surface grid/channel handling through common adapter; array shape/inpaint math is domain policy. |
| `lasagna/setup.py` | X — packaging dependency discovery, no runtime store interpretation. |
| `lasagna/scripts/bootstrap_venv.py` | X — bootstrap packaging match; not executed or treated as store consumer. |
| `spiral-fitting/config.py` | A — application input selectors/fit configuration, not reference ownership resolver. |
| `spiral-fitting/fit_session.py` | D/A — coordinates full-reference input captures, request/receipt provenance and materialized solver input state. |
| `spiral-fitting/fit_spiral.py` | D/A — consumes already-selected/materialized object_store_inputs, not direct local database reads. Model/solver policy excluded. |
| `spiral-fitting/grow_track_graph.py` | D/A — retire_surfaces selects references once, checked write; empty input bypass avoids unintended all-selection. Graph/growth algorithm excluded. |
| `spiral-fitting/object_store_inputs.py` | D/A — native manifest identity/fitting-family/position/coordinate validation, SelectedObject and selected_inputs; app selection/minimum-position/content projection is explicit. No numeric guard fallback. |
| `spiral-fitting/runners/run_single.py` | D/A — passes configured input/pin workflow through fit_session; no second native schema interpretation. |
| `spiral-fitting/service_commits.py` | D/A — creates/reuses committed fit outputs through shared surface API; preview schema and flatten provenance are domain metadata, not vcobj generation substitutes. |
| `spiral-fitting/spiral_service.py` | D/A — strict native position decoder plus app binding/receipt/minimum-position semantics; no new local integer/UUID schema parser. |
| `spiral-fitting/find_inconsistent_windings.py` | A/X — reads fit materialization/summary context only; no independent vcobj selection or persistence. |
| `spiral-fitting/pcl_views.py` | A/X — transforms already-materialized PCL/fiber data for losses, no store I/O or placement policy. |
| `spiral-fitting/spiral_runtime.py` | D/A — input provenance/summary/status and fit update scheduling use object_store_inputs; no physical resolver. |
| `vesuvius/src/vesuvius/fiber_spline_generator/cli.py` | D/A — routes generated output to shared writer; CLI source formats independent. |
| `vesuvius/src/vesuvius/fiber_spline_generator/store.py` | D/A — checked absent-object native create/put_geometry; derived store generation is not graph/history cloning. |
| `vesuvius/src/vesuvius/neural_tracing/inference/surface_store.py` | D — common surfaces.read_surface including raw exact selectors and reference; selected grid identity preserved in output provenance. |
| `vesuvius/src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py` | D/A — _StoredSurfaces load/name exact refs, selected-source output reference nested unchanged; current discovery only for unselected source. Neural algorithms excluded. |
| `vesuvius/src/vesuvius/neural_tracing/tracers/shared/annotations/fiber_store.py` | D/A — complete-reference graph payload reads, source-position validation/evidence digest, measured catalog bounds/counts, native checked derived writer. Scalar direct-bound-volume restriction explicit, not registration fallback. |
| `vesuvius/src/vesuvius/neural_tracing/autoreg_mesh/dataset.py` | X — `surface_stored` tensor helper is a substring hit, no vcobj I/O or schema policy. |

## Additional indirect wrappers and omissions

- `volume-cartographer/apps/VC3D/SpiralUpdatedDrafts.hpp::spiralDraftsIncludedByUpdate`: generic template comparing saved values; actual callers pass shared references. Name `currentVersion` is not evidence of numeric-only pinning.
- `volume-cartographer/core/src/SurfaceQuerySet.cpp` / `include/vc/core/util/SurfaceQuerySet.hpp`: shared native resident geometry kernel; application pointer/ptr-coordinate conversion is not storage-location handling.
- `lasagna/fit.py` and `lasagna/fit_data.py`: indirect selected-surface data consumers through the common adapters; inference/model internals excluded from vcobj finalization.
- Source AFV/constraint SQLite extraction workflows and DBM track readers are independent formats; their SQLite/JSON/version terminology does not imply direct access to vcobj tables. No removal of legitimate source-file support is proposed.
- Native headers are inspected for identity-bearing fields/inline behavior, not claimed as separate executable coverage. GUI callback/event-loop interleavings, every numeric geometry kernel, unsupported platform builds and full-volume performance are **not runtime verified** by this lane. Exact/history/local/referenced/sharded/packed regression execution belongs to the approved root runner/core lane; no passing-test count is reused as proof for an unexecuted consumer branch.

All discovered files above have an explicit scoped disposition; no hidden “unchecked means clean” category. This does not convert boundary inspection into an exhaustive line-by-line algorithm review. No new candidate requires comparison/provenance classification against `70da58935`; the latest changed consumers were directly diff-reviewed as recorded above.
