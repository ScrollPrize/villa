# Post-class-sweep consumer and binding review

Frozen HEAD: `d13d33629`; reviewed working production/test patch hash: `90ad71c5ac3c79abcff42509dd0d2c7584d0a43e645180cd13983e23ad01662f` (`git diff --binary -- vc_object_store/src vc_object_store/include vc_object_store/python spiral-fitting lasagna volume-cartographer vesuvius | sha256sum`). This lane changes this report only; no production/test edits, installations, builds or competing test runs.

## Result

No additional confirmed consumer correctness defect established. This is bounded source review, not a proof that every GUI interleaving, external service failure or geometry algorithm is correct. Receipt decoder admission is being reviewed in the core lane; callers correctly using its shared result do not independently validate the receipt schema.

The known receipt-marker, saved Lasagna collection and saved Spiral pins candidates are closed in this patch at their owning boundary. They are not new findings in this pass. Malformed saved containers previously being accepted/reinterpreted is the demonstrated effect; an actual loss of selected source geometry/provenance was not demonstrated by empty malformed inputs.

## Inventory and reproducibility

Contracts read: `review-contract.md`, `versioned-reference-contract.md`, `reading.md`, and the coherent-store contract. Started from `class-sweep/fixes.md`, the per-file/symbol inventory in [post-closure/consumers.md](../post-closure/consumers.md), the nine-format inventory in [post-closure/formats.md](../post-closure/formats.md), and the repository-wide Python boundary dispositions in [class-sweep/python-references.md](../class-sweep/python-references.md). Those are detailed inherited evidence, not a claim that all geometry/model code was re-read line-by-line in this pass.

Repeated discovery:

```sh
rg -l 'vc_object_store|vc::objects|ObjectStoreSession|ObjectRevision|ObjectVersion' volume-cartographer --glob '*.{cpp,hpp,h}' --glob '!**/test/**' --glob '!**/tests/**' --glob '!**/build/**'
rg -l 'vc_object_store|stored_surfaces|object_store_inputs|surface_store' spiral-fitting lasagna vesuvius/src --glob '*.py' --glob '!**/tests/**' --glob '!**/test_*.py'
git grep -l -e vc_object_store -e 'vc::objects' -- '*.py' '*.cpp' '*.hpp' '*.h' ':!vc_object_store' ':!**/test/**' ':!**/tests/**' ':!**/test_*.py' ':!**/build/**'
rg -n 'objects_at_versions|object_reference|relationship_reference|selected_inputs|expected_objects|expected_relationships|input_references' spiral-fitting lasagna vesuvius/src --glob '*.py' --glob '!**/tests/**' --glob '!**/test_*.py'
rg -n 'sqlite3|SELECT.*objects|SELECT.*edges|external:|legacy_source|\.version[[:space:]]*==' volume-cartographer/apps/VC3D volume-cartographer/core/src vc_object_store/python --glob '*.{cpp,hpp,py}' --glob '!**/test/**'
git diff --name-only de3afb065 -- volume-cartographer spiral-fitting lasagna vesuvius vc_object_store/python
```

Discovery returns the same 109 native and 24 Python seed files with individually named dispositions in the inherited ledger, plus explicitly followed indirect wrappers there and in the Python sweep. The source comparison confirms no native VC consumer changed since that detailed ledger. Outside tests, only `spiral-fitting/object_store_inputs.py`, `lasagna/fit2tifxyz.py` and `vc_object_store/python/objects.cpp` changed in these consumer/binding trees; each is delta-reviewed below. Tracked sibling `spiral-fitting/cpp/track_crossings.cpp` includes the shared TrackCrossings geometry library and implements geometric correspondence computations, not vcobj persistence/reference selection. It is excluded from store policy review for that reason, not silently omitted.

## Invariant and sibling closure

| Consumer family / files | Owning APIs, repeated trace and disposition |
| --- | --- |
| All native GUI/CLI/core files individually listed in `post-closure/consumers.md` | Retain their per-file D/A/H/X dispositions; source-equivalent since that review. Re-screened current/selected identity, SQL and numeric-guard sites. No consumer SQLite resolver or `external:`/`legacy_source` alternate reference path surfaced. Header declarations are not counted as independent executable paths. |
| `StoreSurfaces.cpp`, `QuadSurfaceStore.cpp`; corresponding headers and tool callers | `currentObject` selects once, `pinOf`/`noteRead` retain full selected pin, `GeometrySnapshot`/`readSurfaceGrid` read geometry, `checkMappingOperation` retains mapping evidence, `ToolObjects::writable` guards edits. `update` returning the receipt numeric version is a public result projection, not an internal guard or reconstructed reference. |
| `SurfaceCatalog.cpp`, `SurfacePanelController.cpp`, `FiberCollectionController.cpp`, `LineAnnotationController.cpp` and structural edits | Current discovery is explicitly separate from retained exact pins. Shared session/geometry/region APIs own placement; geometry revision IDs identify immutable payloads. Loaded mapping, membership and deferred draft state remain separate guards. Their detailed symbol dispositions are unchanged. |
| `PointCollectionSync.cpp`, `segmentation/SurfaceStoreEdit.cpp`, `SpiralStoreEdits.cpp` | Checked writes use loaded complete references. Attached collection snapshots retain publication/membership and mapping. Receipt repinning calls `receipt.objectReference`, not committing-root plus numbers. Explicit current reads used for discovery/conflict messaging do not overwrite the loaded baseline. Multi-brush-patch UI saves remain separate checked requests, not a newly promised atomic multi-patch API. |
| `SpiralStoreInputs.cpp`, `SpiralWorkspace.cpp`, service managers, grower and overlay consumers | Previously closed shell role handling remains present in full/incremental shared input paths. Fitting relationships delegate shared eligibility/order policy. Native service result numeric counters remain diagnostics/current-output discovery, not hidden exact handles. Event/cache invalidation and selected geometry remain separate responsibilities. |
| `spiral-fitting/object_store_inputs.py::InputPins.from_provenance` | New application map/UUID-array shape admission occurs before normalization; native scalar codec remains the identity owner. `from_manifest` first calls native `manifestReferences`; internal iteration does not need repeated array guards. Full references remain beside application roles. |
| `object_store_inputs.py::submit_link_plan`, `_link_marker_at`, `_refresh` | Marker uses `receipt_object_reference` unchanged in exact lookup; the request-owner/numeric reconstruction is deleted. `_refresh` already uses the same helper. No numeric compatibility reader remains. Invalid receipt decode is not treated as proof that the transaction did not commit. |
| `object_store_inputs.py::changes_after`; `spiral_service.py`, `fit_session.py`, `service_commits.py` and indirect solver/runtime wrappers | Feed adaptation is legitimate: native ChangeRecord exposes UUID/counter with its feed position, not a complete reference discarded by Python. Saved application pins use the decoder above. CommitResult's number is status only; native panel selects output explicitly by UUID. No new alternate internal handle established. |
| `lasagna/fit2tifxyz.py::_stored_sources` and `_export`; `fit_service.py`, `stored_surfaces.py`, `tifxyz_io.py`, `approval_inpaint.py`, indirect `fit.py`/`fit_data.py` | Saved reference array now decodes through native `object_versions_from_json` once; tuple of UUIDs is an intentional output provenance-source projection, never used to reconstruct selected read guards. Input artifact/geometry reads retain the complete selected reference. Service parsing and external tifxyz source dispatch remain application boundaries. |
| Vesuvius `surface_store.py`, `_StoredSurfaces`, `fiber_store.py`, fiber spline store/CLI; sheet-strip build/index/infer and matching siblings | Full selected references remain in graph reads, cache digests, resume evidence and derived-input provenance. New-object absent expectations remain valid. Independent source-path/coordinate jobs and scalar explicit CLI selectors are not legacy vcobj handles. All per-file dispositions from the Python sweep remain applicable. |
| `volume-cartographer/python/vc/fiber_evaluation.py`, MCP store/catalog wrappers and trace-service diagnostics | Native complete references are retained when supplied. UUID routing/current discovery or echoing an external numeric selector is not reconstruction of a discarded exact selection. Independent geometric reference axes/evaluation datasets are excluded. |

## Python native/facade binding inventory

| File | Disposition / shared owner |
| --- | --- |
| `python/objects.cpp` | Public read/write/hosted/session adapters call native query, selected-input, transaction and geometry APIs. New list codec is a thin `objectVersionsFrom` binding, not a second Python schema. Scalar codecs and receipt/reference helpers remain native-owned. Emitted record projections are decoded/native output, not saved protocol ingress. |
| `python/StorageBindings.hpp` | Object/edge conversion, JSON adaptation and transaction context lifecycle; anchors/scale delegate native codecs. Native writes remain validation/atomicity owners; Python context manager commits only successful operation completion. |
| `python/ObjectReadBindings.hpp` | Graph read view lifetime/thread checks and resident-array adaptation. Current graph branch and explicitly selected graph branch are typed native modes, not attachment-owner inference. Payload access remains by retained native geometry revision. No saved JSON is decoded by Python loops. |
| `python/TypedGeometryBindings.hpp` | `bindGeometryFactory` distinguishes explicit UUID current selection from a complete reference decoded with native codec, then `snapshot.selected`; rejects mixing selected ref with a geometry selector. NumPy layout/data ownership adaptation is not a version resolver. |
| `python/GeometryRegionBindings.hpp` | Native region result adaptation retains `run.objectReference` and mapping separately. |
| `python/SessionBindings.hpp` | Attachment/session lifecycle, copy/import delegation and native event transport. Attachment generation is session identity, not a substitute object version. |
| `python/ExtensionBindings.hpp` | Extension value/table/row adaptation; versioned source identity uses native extension/object/relationship codecs. Object-versus-relationship dispatch chooses the typed overload; native API owns exact selection and placement. |
| `python/AfvBindings.hpp` | Independent source format metadata/geometry adapter; not a vcobj physical resolver. |
| `python/vc_object_store/selected.py` | Retains record/revision; reference property delegates `object_reference`. No current re-read. |
| `python/vc_object_store/surfaces.py` | `SurfaceGrid` holds selected record beside data; native `read_surface_grid` owns current/numbered/full selection, transformation and validation. Shape checks for new NumPy grids are language/data adaptation. |
| `python/vc_object_store/locations.py` | Shared session attachment, native exact reads and query execution. Public iterable-to-list conversion supports Python tuples/generators; no concrete saved raw JSON ingress bypass was established. Do not mistake it for the now-fixed saved-checkpoint list boundary. |
| `python/vc_object_store/documents.py` | Explicit current UUID attachment/document API delegates native shared implementation. |
| `python/vc_object_store/derivation.py` | Application metadata copy policy refuses managed payload copies; retains complete supplied source reference. Not a geometry/owner resolver. |
| `python/vc_object_store/__init__.py`, `python/CMakeLists.txt` | Export/build wiring only. |

## Formats and current patch effects

The nine-format per-file ledger in `post-closure/formats.md` remains the scoped source-format inventory. `FiberJson.cpp` now calls `validateSourceTopology` before file decode and endpoint discovery; `PointCollectionInput.cpp` calls it on normalized point attributes. Prepared/batch ingest reaches the same helper from `GeometryIngestBatch::normalizeMetadata`. `SourceRelationships.cpp` alone owns the source-topology container admission. These external fiber source versions are not legacy vcobj database schemas; independent source import remains supported. The extra validation does not infer endpoints/owners or add cross-shard tables. Internal typed loops remain legitimate after admission. No new consumer-specific shape or scalar codec is required.

## Explicit candidate disposition: ObjectCopy provenance

`src/ObjectCopy.cpp:31–33::readObjectCopy` emits `copied_from = {store_uuid, object_uuid, generation, version, geometry_revision}` using the selected source owner. This is documented by `objects/ObjectCopy.hpp` and `docs/writing.md`, and prior native/class-sweep reviews explicitly classified it as external informational provenance. Repository search found its producer, tests and documentation, but no operational reader rebuilding a reference. Therefore this pass does **not** promote it as a correctness regression or wrong-geometry issue. Standardizing it to `{reference: <shared reference>, geometry_revision: ...}` is possible contract uniformity cleanup, but changes an explicitly documented metadata shape and should be described that way. Old conversion fixtures containing historic `object_revision` provenance are source conversion evidence, not a live fallback API.

## Limitations

No new runtime probes or builds were run by this lane. Root's approved runner is the independent execution evidence. Local/referenced/history-on/off/packed/sharded behavior at native store boundaries is delegated to the core review and regressions, not re-proven by consumer inspection. Interactive GUI concurrency, macOS/arm64 builds, every numerical geometry/model kernel, full-volume solver tuning, unrelated source-format algorithms and intentionally informational metadata layouts are outside this source review. The inherited detailed per-file ledger is explicitly retained rather than represented as a fresh line-by-line re-read of unchanged code. No unchecked store-sensitive sibling discovered by the repeated inventory is hidden as a successful runtime result.
