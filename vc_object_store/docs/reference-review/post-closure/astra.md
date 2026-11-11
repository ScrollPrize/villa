# Independent post-closure review

Frozen source: `de3afb065`; latest delta `70da58935..de3afb065`. Review-only:
no production edits, installations or competing builds. Contracts read:
`review-contract.md`, `versioned-reference-contract.md`, `reading.md`, and the
coherent-store contract. Prior review evidence is not treated as a new runtime
result. Independent storage/transport coverage is complete to the explicitly
bounded ledger below; cross-lane adjudication is recorded separately when ready.

## PC-A01 — metadata re-import resurrects deleted identities

**Medium, pre-existing omission.** `src/Snapshots.cpp::ObjectStore::importJson`
checks `objectByUuid` / `edgeByUuid`, which deliberately hide tombstones, then
calls the ordinary resurrection-capable `upsertObjectImpl` / `addEdgeImpl`.
Consequently importing an earlier metadata export silently restores objects and
relationships deleted since that export. Import has no explicit restore option;
its adjacent cases are idempotent same-content import and rejection of conflicting
content. This differs from the documented rule that ingestion reserves deleted
identities and re-import is not restoration.

Native ownership trace: `internal/RecordResolver.hpp::{objectSource,edgeSource}`
supports `includeDeleted`; the import does not request it. Collection callbacks in
`ObjectCollection.cpp::{installScope,locateSourceSql}` (the installed resolver
lambdas and `locateSourceSql`) stop at the winning tombstone rather than falling
through it. That correct live lookup still returns absent to this import.
`internal/RecordSql.cpp::upsertObjectImpl` can reactivate a local deleted object;
an inherited tombstone likewise does not create an ordinary live before-state
override. `internal/EdgeSql.cpp::addEdgeImpl` explicitly continues a deleted edge
when a caller intentionally inserts it. Those low-level capabilities are not
the bug: the metadata-import admission policy incorrectly chooses them after a
live-only presence check. `geometry/GeometryIngestBatch.cpp::ingestGeometryBatch`
already uses tombstone-aware admission and provides the sibling policy.

Runtime diagnostic against the existing native Python build:

```sh
PYTHONPATH=local/vcobj-pr-tests/build/python /home/hendrik/.venv_las/bin/python -c 'import vc_object_store as v,tempfile,pathlib,json
for history in (False,True):
 d=tempfile.TemporaryDirectory(); s=v.ObjectStore.create(str(pathlib.Path(d.name)/"x.vcobj"),author="probe",history_enabled=history); a=s.upsert_object({"kind":"fiber"}); b=s.upsert_object({"kind":"fiber"}); e=s.add_edge(dict(kind="test_link",a=a,b=b,a_anchor=dict(whole_geometry=True),b_anchor=dict(whole_geometry=True))); snapshot=s.export_json(); s.remove_edge(e); s.import_json(snapshot); print("history",history,"edge resurrected",s.edge(e) is not None); s.remove_edge(e); s.delete_object(a); obj=json.loads(snapshot);obj["edges"]=[];s.import_json(json.dumps(obj));print("history",history,"object resurrected",s.object(a) is not None);s.close()'
```

All four resurrection outputs are `True`. Temporary fixtures auto-clean on exit.
Separate local diagnostic mutations to use a deleted edge UUID for an object,
and a deleted object UUID for an edge, **reject** for both history policies.
The lower-level closure-wide cross-kind checks are therefore not reported broken.
Referenced/sharded/packed effects are source-traced, not reproduced by this lane;
no hosted metadata-import endpoint is claimed. The Python method delegates to
the native importer. Source attribution: identical admission logic exists at
`70da58935`; this is not caused by the latest three fixes.

Minimal fix direction: share tombstone-aware record admission for import,
retaining current idempotence/conflict behavior and operation-wide savepoints;
do not change intentional explicit resurrection APIs. Regressions should cover
object and edge, local/referenced tombstones, both history policies, cross-kind
rejection, and catch-error/commit-outer-transaction atomicity.

## PC-CORE-01 adjudication — selection artifact position coercion

**Accepted, low boundary-validation inconsistency; pre-existing.** The core lane
identified `src/Selections.cpp::decodeSelectionRecord` reading saved
`commit_sequence` with `get<int64_t>()` instead of shared `commitSequenceFrom`.
Independent runtime confirmation:

```sh
PYTHONPATH=local/vcobj-pr-tests/build/python /home/hendrik/.venv_las/bin/python -c 'import vc_object_store as v,json,struct,uuid; u=str(uuid.uuid4()); h={"store":u,"generation":u,"commit_sequence":1.5,"query":{},"transforms":[],"origins":[],"count":0}; b=json.dumps(h).encode(); print(v.decode_selection_record(b"VCSEL\x00\x02\n"+struct.pack("<I",len(b))+b))'
```

The returned position is `commit_sequence: 1`; the shared position codec rejects
the fractional input. Actual demonstrated effect is altered saved provenance/
position, not lost object identity or proven unauthorized/wrong geometry reads.
The native encoder and Python `selectionRecordFromPy` are sibling ingress paths
to cover, including negative native values and malformed Python values. A shared
position-validation route is preferable to another numeric-policy table.
`70da58935` contains the same decoder. This is not a latest-fix regression.

**Additional scalar sibling, same root cause:** source-traced
`internal/RecordJson.cpp::commonFromJson` also converts `commit_sequence`,
`version` and non-null `approved_version` with general integer getters before
range/approval-consistency checks. Include this under PC-CORE-01, not a third
finding. `Snapshots.cpp` and non-projected edge decoding use it; native retained
record reconstruction and wire revision decoding share it as well. For metadata
import specifically, `RecordSql.cpp::restoreCommon` intentionally republishes
local commit/version counters, so accepting malformed source numbers does **not**
establish incorrect destination version ownership or wrong historical geometry.
The bounded issue is permissive source-number validation (including conversion
before comparing approval to source version). Reuse strict shared scalar helpers
where these fields promise numbered values, preserve the existing local-publication
semantics, and add sibling malformed-input tests. This additional branch is
source-traced, not independently runtime-reproduced here.

## Latest-fix delta dispositions

- `python/ExtensionBindings.hpp` and `tests/python/test_extensions.py`: strict
  relationship decoder now matches object decoder at store and transaction
  boundaries. Final test correction accepts the existing missing-field exception;
  validation was not relaxed.
- `SpiralStoreInputs.cpp/.hpp`, `SpiralWorkspace.cpp/.hpp`, and
  `test_spiral_store_inputs.cpp`: common role-bearing enumeration feeds full,
  incremental and notification paths; common read and display builders handle
  shell state/geometry/deletion. Shell role excluded from fitting kinds and
  incident discovery; shared relationship policy rejects it even on a direct
  relationship notification. Final fixture removes the existing shell edge and
  supplies expected geometry revision for a valid geometry write. No new defect
  confirmed. Actual GUI event-loop installation remains source-traced.
- Neural `_StoredSurfaces.save` and `test_surface_store_io.py`: selected reference
  retained unchanged under `input.reference`; original and chained-source checks
  use that shape. No geometry behavior change or compatibility reconstruction.

## File/symbol coverage ledger

Each row records an owning boundary and its concrete sibling paths, not an
assertion that every unrelated line in those files was formally proved.

| Files / symbols | Disposition |
| --- | --- |
| `Snapshots.cpp::exportJson/importJson`; `python/objects.cpp::import_json`; `internal/RecordSql.cpp::upsertObjectImpl`; `internal/EdgeSql.cpp::{addEdgeImpl,requireUnusedObjectUuids}` | PC-A01. Export selects visible scoped rows; import has a whole-operation savepoint. Cross-kind checks remain closure-wide and reject tombstones in diagnostics. |
| `geometry/GeometryIngest.cpp`, `geometry/GeometryIngestBatch.cpp::ingestGeometryBatch`; `imports/SourceIdentityIndex.cpp::{sourcePoint,sourceCollection,refreshSourceIdentityIndex}` | Native import policy and `VisibleObjectRows` handle identity/precedence; source index is storage-owned accelerator, not a consumer proxy. No additional defect established. |
| `InitialImport.cpp::{importInitialCollection,InitialShardImport::appendObjectRefs,importEdges}`; `internal/InitialObjects.cpp`, `internal/InitialEdges.cpp` entry points | Bulk orchestration validates provisioning/ranges, confines producers, collects failures before publication; immutable initial storage specialization is legitimate. Detailed SQL batch micro-optimizations not re-proved. |
| `ObjectCollection.cpp::{locateSourceSql,copyObjectCollection,objectAtVersion,relationshipAtVersion,readGeometryArrayAtVersion}` and resolver callbacks; `internal/ReferenceResolver.hpp`; `internal/RecordResolver.hpp` | Shared placement/identity/tombstone resolution; exact wrappers preserve complete refs. Copy traverses pinned closure. No local SQL proxy introduced by latest delta. |
| `ObjectCopy.cpp::{readObjectCopy,objectCopyOperations,writeObjectCopy}`; `ObjectStoreSession.cpp::{copyObject,importObject,submitCopy}` | Copy pins source metadata/geometry once; extension-bearing stores fail closed; artifact transfer uses selected manifest hashes. Explicit new-copy UUID is application action, not owner reconstruction. |
| `Artifacts.cpp::{parseManifest,resolveArtifact,writeArtifactFile,ArtifactExport,putArtifact,readResolvedArtifact}`; `ObjectStoreSession.cpp::{artifacts,materializeArtifacts,writeArtifact,artifactFile,publishArtifacts,publishArtifactManifest}` | Shared native manifest/resolver, exact selected metadata and hash-verified atomic output; writes retain selected guard. Per-file publication, not a multi-file transaction. External corruption/concurrent filesystem mutation not exhaustively fault-injected. |
| `SourcePaths.cpp::{validateExportDestination,resolveStoragePath}`; `geometry/GeometryExport.cpp::exportGeometry`; `internal/RecordOverrides.cpp::{rebaseExternalLocator,relocateExternalLocators,prepareOverrideReferences}` | Destination protects dependency closure and sidecars; selected geometry export and owner-relative relocation shared. History relocation uses archived-record transformer. No additional concrete issue established. |
| `internal/DatabaseCopy.hpp::copySqliteSnapshot/relocateCollectionMembers`; `PackedCollection.cpp::packObjectCollection`; `PackedStore.cpp::{importDatabase,finalize,compactPackedStore}` | Snapshot copier, relocation, closure validation, artifact verification before publication are shared storage responsibilities. Compression/VFS byte-level implementation not re-proved. |
| `migration/ConvertStore.cpp::{convertStoreImpl,copyToCurrentSchema,copyArtifacts,convertStoreInPlace}` | Explicit external conversion boundary, source snapshot and artifact verification, failed-destination cleanup. Old-schema codecs are intentionally not runtime compatibility branches. All old-schema migrations not independently replayed. |
| `Extensions.cpp::{createExtensionTable,resolvedExtensionTable,scanExtensionRows,appendExtensionRows}`; `ExtensionHistory.cpp::{replaceExtensionRows,scanExtensionRowsAtVersion,scanExtensionSnapshot}`; `ObjectCollection.cpp` extension delegates; `python/ExtensionBindings.hpp` | Native schema/row interpretation, visible selected owner descriptors and shared payload reads; exact APIs do not substitute current heads. No-op write rollback preserves ownership/stamps. Python refs now both use shared codecs. |
| `Selections.cpp::{recordSelection,encodeSelectionRecord,decodeSelectionRecord}`; `python/objects.cpp::{selectionRecordFromPy,encode_selection_record,decode_selection_record}` | PC-CORE-01 for saved position validation. Binary origin dictionary is an external storage encoding, not an internal alternate identity. |
| `StoreReader.cpp::{selectedObject,decodeObjectReference,selectedInputs,currentObjectExpectation}`; `internal/ObjectVersionJson.hpp`; `Wire.cpp` selected-input/reference codecs | Exact references resolved before current lookup; explicit discovery distinct from saved exact selection. Native position codec is strict; selection artifact exception above. |
| `client/HostedStore.cpp::{currentObjectRevision,objectsAtVersions,relationshipsAt,selectedInputs,recordSelection,putArtifact,readArtifact,fetchArtifact}`; `server/Server.cpp` corresponding handlers | Shared wire codecs/server read transactions; complete owner-bearing values transported. Hosted `recordSelection` limit rejection delegates to `selectAllObjects`; no missing limit parity defect. Client-local output namespace legitimate. |
| `HostedSession.cpp::{currentObject,objectsAtVersions,relationshipsAt}` and `HostedStoreReader` adapters | Exact cache keys preserve owner/lineage; graph cache includes observed root freshness. No current-head prerequisite added for exact cache reads. All cache races and prefetch interleavings not stress-tested. |
| `Project.cpp::{sortByAdded,selectedExpectation,copyProject,appendRecordWrite}` and project header | Source-traced owner-aware optional history, shared tombstone expectations and closure copy. Explicit resource create/restore policy is distinct from generic metadata re-import. |
| `StoreConversion.cpp::changeHistoryMode`, `Provisioning.cpp::writeRecordVersion`, `Database.cpp` open/create feature checks; corresponding public headers | Source-traced maintenance entry boundaries: exclusive staged rewrite, current identity retention, borrowed payloads left under dependency retention, schema21 feature gating. Trigger/schema reconstruction internals not exhaustively re-proved. |
| `imports/SourceRelationships.cpp::importSourceRelationships` and header | Source-format relationship conversion inside one native write transaction; shared point identities, geometry/dependency reads and edge writer. Native source-format reciprocal-link policy is legitimate, not a second vcobj resolver. |
| `PackedTrackStore.cpp`, `PackedTrackStore.hpp` | Excluded from vcobj placement policy: independent VCTRK01/DBM-derived track source format, not packed SQLite vcobj storage. Header/count/endian validation inspected; numeric format implementation not re-proved. |
| `objects/imports/ParallelImportStream.hpp`, `ShardQueue.hpp`; `imports/AfvSource.cpp`, `AfvRecords.hpp`, `FiberImportExtension.cpp`, `FiberGraphSolveExtension.cpp` | Source-import/queue/domain-schema boundary screen only. Queue scheduling, AFV source decoding and domain-specific solver extension columns not independently re-proved; they remain explicit limits, not delegated-clean claims. Shared native registration/write policy is covered above. |
| `internal/Schema.cpp/.hpp`, `Sqlite.cpp/.hpp`, `Ownership.cpp/.hpp`, `Publication.hpp`, `Digest.hpp`, `ImmutableImport.hpp`, `CollectionReads.hpp`, `Database.hpp` and remaining declaration-only storage headers | Reached through covered open/copy/publication/resolution callers. Low-level SQL schema, locking/platform, digest and publication helper implementations are not fully re-audited line by line here. No consumer-local policy or new representation is introduced by the reviewed delta. |

## Final cross-lane adjudication

Read the completed core, consumer and root ledgers and their follow-up supplements.
Accept **PC-CORE-01**, **PC-A01**, and **PC-S01** with the evidence and bounded
severity recorded here. The core ledger's earlier `Snapshots.cpp` “no new defect found” disposition
is superseded by the independently reproduced tombstone counterexample here;
its savepoint observation remains correct. Consumer lane reports no new candidate,
and its latest shell/provenance conclusions agree with the independent delta trace.

Delegated file labels do not broaden this lane's evidence beyond the explicit
ledger. In particular, low-level format/numeric/concurrency implementations remain
limited or excluded as stated, rather than becoming “clean” because another lane
delegated them. No additional confirmed latest-fix regression. All three accepted
findings are pre-existing and require a subsequent implementation/regression pass.

## PC-S01 — follow-up adjudication: stale-link scope UUID spelling

**Accepted, low-priority scoped-reconciliation defect; pre-existing.** Independently
traced `linking/LinkCandidates.cpp::commitCandidatesImpl` and the public
`object_links/ObjectLinks.hpp::StaleEdgeScope` contract. The scope promises UUID
membership, with no canonical-spelling-only restriction. `scopedObjects` and
`touching` are built from raw strings, while `selectEdges` interprets UUIDs through
the shared parser (`internal/EdgeSql.cpp` endpoint membership uses
`formatUuid(parseUuid(value))`) and returns canonical endpoint strings.

Concrete source-established counterexample: create one pending edge owned by
detector D between UUIDs containing hexadecimal letters. Call public
`commitCandidates` with no new candidates, replacement enabled, matching edge
kind and both endpoints in `StaleEdgeScope.objectUuids`, uppercased. Selection
finds the edge, but raw scope-set checks skip it; the lowercase equivalent removes
it. Keeping ordinary object scope canonical but uppercasing its nonempty
`touching` endpoint independently causes the same omission. Both public overloads
reach this implementation. Invalid touching strings likewise bypass shared UUID
validation at this final comparison boundary.

This does not corrupt selected references or broaden removal; it silently leaves
obsolete detector-owned pending links in an explicitly reconciled scope. Normal
detector-generated scopes use selected canonical record UUIDs and canonical
`freshUuids`, so direct native callers supplying alternate accepted UUID spellings
are the demonstrated boundary exposure. Same raw-set code exists at `70da58935`;
not a regression from the three closure fixes. No new native harness/build or
runtime reproduction claimed here. Existing linker tests do not exercise this
spelling parity case.

Severity is low because ordinary detector producers supply canonical UUIDs and
the failure retains pending edges rather than removing reviewed data; the public
native spelling-parity contract nevertheless warrants a focused fix.

Minimal repair is to canonicalize/validate scope UUID inputs through the existing
shared UUID API at admission, then use those values consistently. Regress both
object scope and touching spelling, both public overloads, empty scope semantics,
and reviewed/foreign-detector preservation. Do not introduce another UUID parser,
change detector geometry algorithms, or weaken selected-write guards.

Read `formats.md`: its nine format-file dispositions and immediate catalog-fetch/
ingest/export traces agree with the already-inspected shared boundary separation.
No additional format candidate is established. The stale-link finding is a third
independent root cause discovered in the follow-up coverage pass and included in
the final three-finding summary above.

Final follow-up crosscheck: read completed `spatial-linking.md` and the root's
seven additional helper/build dispositions. Agree with PC-S01's low severity and
its source-only evidence; no disagreement with the remaining boundary exclusions.
The final accepted set is **PC-A01 (medium), PC-CORE-01 (low), PC-S01 (low)**,
all pre-existing. The additional per-file dispositions close the initially
unassigned boundary inventory, not numerical/platform/concurrency proof limits.
No further confirmed defect or latest-fix regression from this adjudication.

## Verification boundaries

This lane ran only the disposable Python diagnostics above and cross-kind checks,
against the existing build. The main agent owns the approved regression runs.
No full GUI, macOS/arm64, full-volume workload, performance benchmark, network
fault injection or additional build claimed. “Full review” here means explicit
invariant/sibling coverage across the requested storage boundaries, not proof of
every numerical kernel, SQLite/VFS implementation line or asynchronous schedule.
Source-traced placement/history combinations are not silently counted as runtime
coverage. Earlier three Spiral test failures are neither reclassified nor used
to excuse the new findings.
