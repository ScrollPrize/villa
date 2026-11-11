# Follow-up: complete handles across consumers and deferred edits

Reviewed source: `5920625ccaa7b72ee8f60a9d659f6b49190df299`. Scope: the complete-reference invariant across native store APIs, C++ GUI/CLI consumers, Python bindings, Lasagna, Spiral and neural-tracing adapters. This is source inspection, not another general geometry/solver audit or runtime verification. No production changes are made by this review.

## Contract clarification

Numeric versions may be an API/wire/storage encoding detail. They are not an alternate handle to pass between application components. Consumers should retain `ObjectVersion`, `RelationshipVersion` or selected records exposing those references; shared transaction APIs translate them into checks. Absence/creation is an explicit expectation, not a fabricated version-zero published reference. A current UUID selector resolves once before it becomes a selected dependency.

Deferred/grouped editing legitimately retains an unpublished draft, attachment, loaded mappings, pending submission and complete loaded reference. These are separate responsibilities, not reasons to replace that reference with a numeric counter or a freshly read head. Resolving pending submissions and repinning from complete receipts remain necessary.

## Findings and dispositions

### F-01: incomplete umbilicus fingerprint — shared identity needed

`LineAnnotationController.cpp:umbilicusFingerprint()` serializes object UUID, generation and numeric version, omitting the owning store. `FiberMapStaleness.hpp: FiberMapDependencies`, `FiberMapWorkspace.cpp:withCachedUmbilicusStatus`, dependency snapshots and in-flight-job guards carry this string. No caller parses it into a reference, but it is still a partial selected-identity comparison.

The map genuinely needs status for a missing/unreadable umbilicus, target-volume/mapping invalidation and application generation counters. Keep those application fields alongside an optional full `ObjectVersion`; use shared reference equality. `_storedUmbilici` is separately invalidated by `invalidateStoredUmbilicus`; this review does not claim that its event invalidation is absent or that a visible stale-result bug was reproduced. A domain cache key may still include volume/scale, which are not object identity.

### F-02: one-element list codecs for scalar references — unnecessary boundary wrapping

`SpiralStoreEdits.cpp:setPointCollectionObject`, its corresponding save-time decoder, and `SpiralWorkspace.cpp:loadFitPointCollections` wrap one reference in a `versions` list and extract its first element. Binding siblings in `python/objects.cpp` (attached collection, receipt helper and object-reference helper) and `python/GeometryRegionBindings.hpp:regionResultToPy` do the same for scalar results.

Qt/nanobind type conversion and JSON persistence are legitimate boundaries. No list is required by these scalar consumers: use existing `objectVersionToJson/fromJson` through shared binding helpers. Actual batch reads, selection records and two-endpoint result conversion remain legitimate list operations. These sites retain complete identity today; this is consolidation/avoidable work, not a demonstrated wrong-version read.

### F-03: numeric versions escape transaction boundaries — broader API migration needed

The current `VersionSelection` type is `map<UUID, optional<int64_t>>`. It is exposed in `TransactionRequest`, `TransactionContext`, `PreparedOperation`, `PreparedOperations::expectedObjects/readDependencies`, session fetch callbacks and some document helpers. `expectObject(request, reference)` already provides the full-handle entrypoint for writes, but requests retain both numeric and complete maps; read dependencies and relationship expectations do not have the equivalent complete-handle route.

Concrete producer/caller closure inspected:

- `linking/StoredGeometry.hpp/.cpp`: `versions_`, `select`, `validate`, `expectations` retain/pass only object UUID and number; `LinkObjects.cpp` exports those into preparation expectations. Snapshot/retry handling is necessary; dropping identity is not.
- `StoreReader.cpp`: the request overload of `expectRelationshipEndpoints` adds complete references. Its `VersionSelection&` overload then discards them. No production caller of that numeric overload was found. `transformDependencies` returns a numeric map, consumed by `VolumeTransforms.cpp`; the enclosing commit guard provides additional protection, but does not require this alternate representation.
- `ObjectStoreSession.cpp:publishArtifacts(attachment, uuid, expectedVersion, ...)`, its Python binding and `lasagna/stored_surfaces.py:StoreSession.publish_artifacts`: current discovery is legitimate; after discovery the selected reference should survive upload/submission. The existing `PinnedObject` overload and `ToolObjects::publishArtifacts` already carry complete identity. Artifact bytes and selected metadata remain separate.
- `ObjectStoreSession.cpp:ensureGeometry/fetchWrite`, `ObjectStoreSession.hpp:FetchWrite`: fetch callbacks expose numeric selections to their next layer. Keep fetch-specific eligibility/transaction ownership but pass full selected handles.
- `SurfaceCorrections.cpp:writeGrowthAnnotation`: the only production caller (`SegmentationModule_Session.cpp:saveGrowthAnnotation`) passes its editor. `SurfaceStoreEdit::submitAttached/guardAttached` supplies the complete loaded pin before submission. The direct optional-editor path is used by a native fixture and lacks this full guard; it is not evidence of a currently unguarded GUI call. Both paths can use the same handle API; grouping does not justify a numeric-only selection. Avoid re-reading the current surface merely to reconstruct the editor's already loaded identity.
- `AttachedDocument.cpp`: collection updates retain loaded references; endpoint expansion adds complete guards, and replacement guards membership plus loaded owner/collection references. Numeric assignments here are not all missing-guard bugs. Preserve loaded overrides, explicit absence and deleted/restoration semantics while centralizing conversion. `Project.cpp:appendRecordWrite` likewise adds the existing record's complete reference. `VolumeUmbilicus.cpp:setVolumeUmbilicusRequest` guards the selecting commit; it can still carry full references to selected existing umbilici.
- `geometry/GeometryIngestBatch.cpp:addIngestOperation`: imported UUID absence and existing fiber-branch endpoint selection share a numeric map. New-object absence is necessary; an already selected existing endpoint can use its complete reference instead of losing owner/lineage.
- `Commands.cpp`, `Versions.cpp` and transaction receipt/wire paths are the owning conversion/checking boundaries, not consumers to duplicate elsewhere. Full reference checks compare all four fields today; numeric-only checks compare only numbers. Some producers additionally guard the whole commit or hold one transaction, so this is an API consistency gap with case-dependent correctness risk, not a claim that every numeric use currently fails.

Recommended shape: one internal selection of complete handles plus explicit absent expectations, with shared API helpers for write/read/relationship dependencies. Keep legacy numeric external input conversion at a documented boundary if required, and keep numeric SQL counters internally. Do not add another independently synchronized full-reference map for every existing numeric map.

### F-04: relationship provenance retains only a number — full relationship identity needed

`spiral-fitting/object_store_inputs.py:StoredRelationship/filter_relationships/provenance` retains relationship UUID and numeric version without its owning store/lineage. Native `SpiralStoreInputs.cpp:consumedIdentity/compareRelationships` compares the same partial fields. Native selected relationships already have the owning record's publication identity, and the shared `RelationshipVersion` type exists.

The fitter deliberately projects only materialized metadata/anchors and may compare content rather than flag every irrelevant metadata edit. Keep that projection/hash beside a full relationship reference. Content identity does not replace publication identity. This is a provenance/consistency gap; no stale-fit reproduction is claimed. Selected endpoint `ObjectVersion`s and relationship geometry UUIDs have distinct meanings and must not be conflated.

## Justified representations, not findings

| Area inspected | Requirement and disposition |
| --- | --- |
| Surface/fiber/point-collection editing; growth jobs; Spiral patches/conflicts; surface history UI | Shared `ObjectVersion`/`PinnedObject` or selected revisions. Draft values, loaded mapping/membership guards, attachment identities and request IDs remain separate state. |
| `SurfaceCatalog`, `SurfaceStoreEdit`, `PointCollectionSync`, `SpiralStoreEdits`, line-annotation saves, `ToolObjects` | Complete selected references/receipt repinning present. Numeric local variables or explicit creation expectations do not by themselves establish another reference format. |
| `StoreReader`, local/hosted readers, selected graphs, surface grids, regions, geometry export, extension exact reads | Public selected reads use complete references or selected records; placement remains in shared resolution. |
| Geometry revisions and lazy channel loaders (`QuadSurfaceStore`, `SpiralStoredFits`) | Immutable geometry UUIDs identify payloads, not historical object metadata. `objectAtGeometry` is the shared geometry-only selection API. Do not replace these with guessed object references. |
| `SpiralServiceManager::CommitResult`, `service_commits.py` and status consumers | UUID/geometry/version fields currently report commit status and trigger refresh; no exact-reference reconstruction was found in those consumers. Do not treat this display/status structure as a selected object handle. |
| `ObjectRecord`/`ObjectRevision`, SQL rows, migrations, wire records | Schema-owned fields and `.reference()` are legitimate construction boundaries. No requirement to erase stored numeric counters, archive UUIDs, or compact binary selection formats. |
| CLI/server locators, persisted old receipts/drafts, selection artifacts | External decoding through shared codecs is legitimate. Legacy compatibility is not permission for internal tuple/string transport. |
| Lasagna reads/config/service, Spiral object pins/project umbilicus, Vesuvius surface/fiber adapters | Complete reference dictionaries retained. Source digests serialize complete identities without reconstructing references. F-03/F-04 record the distinct remaining artifact-write/relationship-provenance exceptions. |
| Store freshness/feed cursors, UI/data generations, content hashes and volume mappings | Different identities with different purposes; must not be mistaken for object-version handles. |

## Review coverage and limits

Repository-wide native/Python searches covered `ObjectVersion`, `ObjectRevision`, `PinnedObject`, `RelationshipVersion`, reference codecs, owner/generation fields, numeric expectation/dependency maps, artifact publication and provenance. Caller tracing covered the named GUI editing/history/cache/job families, CLI/export, hosted/client/server, binding adaptation, drafts/selection/history, extension selection, Lasagna/Spiral and neural-tracing adapters. Findings list concrete residual occurrences rather than a quota. This is a scoped source audit, not a claim that every line of unrelated code was inspected or that keyword search proves correctness.

No tests were run and no dependencies installed: this pass changes review documentation only. Future implementation validation must use the fixed approved runners recorded in [validation.md](validation.md), not ad-hoc test commands. Add targeted equal-counter/different-owner and deferred-submit cases before claiming these remaining paths fixed.
