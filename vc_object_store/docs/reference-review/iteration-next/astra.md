# Independent next-iteration storage and transport review

Reviewed baseline `b73a31f33` plus the source/test patch whose SHA-256 is
`eb68039f0c6f248274a32885417698098b2b9fb319593273024ebf48c34f1ced`.
Verified that hash with `git diff b73a31f33 --binary -- vc_object_store/src
vc_object_store/include vc_object_store/python vc_object_store/tests | sha256sum`.
No production edits or competing build by this reviewer. Any subsequent patch
changes reopen the affected rows below. The approved runner belongs to the root
lane; its results must not be confused with this lane's diagnostics.

This is an invariant/entry-point review, not a claim to re-prove every numerical
kernel, filesystem race or SQLite implementation. Contracts used: repository
AGENTS, review contract, versioned-reference contract, reading and coherent-store
API contract. The preceding `post-closure/astra.md` ledger and its named storage
sources are the unchanged-source foundation; the current pass reopens fixes,
codec siblings, import/write counterparts and storage/transport boundaries below.
Historical runtime results are not presented as new executions.

## Selection-count scalar sibling (PC-CORE-01)

**Low; pre-existing, same malformed-selection scalar root cause.**
`Selections.cpp::decodeSelectionRecord` obtains `count` with `get<uint64_t>()`.
The public header promises malformed bytes throw. A format-2 header with count
`0.5` or `-0.5` and zero entries is accepted as count zero. This is a strict-input
validation defect, not demonstrated object loss: byte-length equality still
guards the number of actual decoded entries. The native decoder is also the
Python artifact boundary; native producers write integer counts.

Independent diagnostic against the existing Python/native build (no fixtures):

```sh
PYTHONPATH=local/vcobj-pr-tests/build/python /home/hendrik/.venv_las/bin/python -c 'import vc_object_store as v,json,struct; u="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
for count in (0.5,False,True,-0.5,"0"):
 h={"store":u,"generation":u,"commit_sequence":0,"query":{},"transforms":[],"origins":[],"count":count}; b=json.dumps(h).encode()
 try: r=v.decode_selection_record(b"VCSEL\x00\x02\n"+struct.pack("<I",len(b))+b); print(repr(count),"accepted",len(r["objects"]))
 except Exception as e: print(repr(count),"rejected",type(e).__name__)'
```

Outputs: `0.5 accepted 0`, `-0.5 accepted 0`; both booleans and string reject.
Do not incorrectly claim that the actual nlohmann getter coerces booleans here.
Root has elected to fold count validation into PC-CORE-01, so this is not counted
as an additional independent root cause. Final fix/hash review remains pending.

## NX-A01 — legacy relationship-wire compatibility branch

**Low; pre-existing strict-ingress/obsolete-compatibility mismatch.**
`Wire.cpp::relationshipFrom` clears `a_revision` / `b_revision` whenever the first
character is `[`. It does not even require well-formed JSON, and silently turns
an obsolete encoded array or arbitrary array-prefixed string into an empty
archive UUID. The single-reference contract rejects obsolete array reference
forms rather than repairing them at ordinary ingress.

Current producers in `History.cpp::relationship` emit an archive UUID when the
stored endpoint key is an archive, otherwise leave the field empty;
`recordedRelationship` for history-off owners emits complete endpoint references
with empty archive fields. `Wire::relationshipJson` serializes those values and
the complete endpoint references. No current producer needs the legacy branch.

Both `relationshipQueryResultFromJson` and `selectedInputsFromJson` use the same
decoder. The former serves HostedStore relationship queries and HostedSession's
disk cache; the latter serves coherent hosted selected-input capture. Server
handlers call the matching serializers inside read transactions. Thus this is
one shared native boundary, not a Python-only discrepancy. Full `a_reference`
and `b_reference` decode successfully *before* the clearing branch: this finding
does not allege incomplete references accepted, owner inferred, or wrong geometry
selected. Preserve legitimate empty archive fields if removing obsolete support.
Source-established only; no custom native build/network responder was run.

## Boundary ledger and dispositions

Paths below are relative to `vc_object_store/`. A grouped header is an API
declaration associated with the named implementation, not an independent runtime
claim. Unchanged foundation rows retain their previous explicit limitations.

| Files / entry points | Current disposition |
| --- | --- |
| `src/Snapshots.cpp::exportJson/importJson`, `python/objects.cpp::import_json`, `src/internal/RecordResolver.hpp`, `RecordSql.cpp`, `EdgeSql.cpp` | Reopened whole import/export flow. Fixed tombstone admission uses shared winning-source lookup, keeps live idempotence/conflict and whole-operation savepoint. Export's empty endpoint list is an engaged optional empty set, not accidental all-edge selection. No new counterexample. |
| `src/Selections.cpp`, `include/vc_object_store/objects/Selections.hpp`, `python/objects.cpp::selectionRecordFromPy` | Full encode/decode/capture flow reopened. Position validation fixed; count issue above. Binary origin dictionary is a legitimate external encoding, not an internal alternate ref. |
| `src/internal/RecordJson.cpp`, `RecordFields.hpp`, `ObjectVersionJson.hpp`, `src/Wire.cpp` | Fixed common/projected/envelope scalars inspected. Optional commit null retained; absent wire version path retained; explicit null version already rejected before patch. Positive published version independent of retention. Legacy branch above. No new validation table. |
| `src/client/HostedStore.cpp::relationshipsAt/selectedInputs`, `src/server/Server.cpp` matching routes, `src/HostedSession.cpp::relationshipsAt/objectsAtVersions` | Fresh source trace of shared serializers, selected refs, and root-fresh graph cache key. Exact object cache keys carry owner/lineage, unlike graph membership. Wire candidate affects both result routes and cached relationship decoding. Cache scheduling not stress-tested. |
| `src/StoreReader.cpp::selectedObject/decodeObjectReference/selectedInputs`, `src/History.cpp::resolveObjectVersion/resolveObjectRevision/relationship/recordedRelationship`, `src/Versions.cpp` | Reopened exact/current and archive/current-only split. Complete identity determines selected source; history-off is retention policy, not a current-head substitute for retained exact refs elsewhere. No new proxy introduced. |
| `src/Extensions.cpp`, `src/ExtensionHistory.cpp::replaceExtensionRows/scanExtensionRowsAtVersion`, `python/ExtensionBindings.hpp` | Rechecked object/relationship typed-reference codec parity; snapshots selected through full refs. Savepoint rollback preserves inherited ownership on no-op rows. Extension source state is explicit schema metadata; integer check plus nonnegative re-encode rejects overflow into signed negative. No confirmed new bug. |
| `src/Artifacts.cpp::parseManifest/resolveArtifact/ArtifactExport/writeArtifactFile`, `src/internal/Artifacts.hpp`, `include/vc_object_store/objects/Artifacts.hpp` | Manifest length is strict unsigned before extraction, unlike selection count. Scoped hash resolution/export and exact selected manifests remain shared. Hash/chunk/atomic-output foundation retained; no new filesystem fault injection. |
| `src/ObjectCopy.cpp::readObjectCopy/objectCopyOperations/writeObjectCopy`, `src/ObjectStoreSession.cpp` copy/artifact publication, public copy/session headers | Rechecked single selected head supplies metadata/geometry; extension-bearing copy fails closed; local multi-step write wrapped in savepoint. Explicit copy identity/provenance is not internal guard reconstruction. |
| `src/DraftCheckpoints.cpp/.hpp`, `src/FetchQueue.cpp/.hpp`, `src/catalog/CatalogStore.cpp` | Complementary boundary screen: checkpoint base uses shared objectVersionJson/objectVersionFrom; draft counters identify application drafts, not publication identities. Queue schedules current UUID fetch discovery; catalog fetch reads selected record then passes record.reference into checked write. Catalog restore emits retained version.reference. Checkpoint byte arithmetic, arbitrary damaged manifests and worker interleavings not exhaustively verified. |
| `src/ObjectCollection.cpp`, `src/internal/ReferenceResolver.hpp`, `DatabaseCopy.hpp`, `src/PackedCollection.cpp`, `src/PackedStore.cpp` | Shared scope/closure snapshot/relocation path retained. Rechecked packed collection loops over members, rewrites member locators, validates closure before finalization. VFS compression internals not re-proved. |
| `src/SourcePaths.cpp`, `src/internal/RecordOverrides.cpp::rebaseExternalLocator/relocateExternalLocators/prepareOverrideReferences` | Rechecked backing-file/sidecar protection and owner-relative relocation, including archived records and packed physical container. Not all OS symlink races tested. |
| `src/geometry/GeometryIngest.cpp`, `GeometryIngestBatch.cpp`, `src/imports/SourceIdentityIndex.cpp`, `SourceRelationships.cpp` | Reopened ingestion counterpart: tombstones reserved with same shared resolver, batch savepoint, source identities and relationships stay storage-owned. Trusted initial registered-object path is deliberate, not generic re-import. |
| `src/geometry/GeometryExport.cpp::exportGeometry`, `src/formats/GeometryOutput.cpp`, `FiberJson.cpp`, public export header | Exact selection and owning snapshot references retained into native materialization; output staging/publishing separate from selected identity. Current discovery deliberately remains current. Native file numeric encodings not re-proved. |
| `src/InitialImport.cpp`, `src/internal/InitialObjects.cpp`, `InitialEdges.cpp`, `src/StoreConversion.cpp`, `src/migration/ConvertStore.cpp` | Unchanged initial-import/conversion foundation plus reopened current-schema admission callers. Explicit migration may interpret old schema; ordinary Wire compatibility is not justified by that. Queue scheduling and every old-schema conversion not rerun. |
| `src/linking/LinkCandidates.cpp`, public ObjectLinks header, scope tests | Reopened both scope sets/overloads. Canonical UUID API now precedes comparison; deterministic uppercase-only touching regression inspected. Detector numerical kernels unchanged. |
| Remaining storage headers/helpers, schema/SQL/publication/ownership primitives and independent `PackedTrackStore` format | Enumerated in preceding complete native ledger; unchanged declarations or lower-level mechanisms reached by named boundaries. Not asserted newly line-by-line verified. Independent VCTRK format is not vcobj packed placement. |
| Native geometry/spatial/import-format kernels and application Python/C++ consumers outside rows above | Other review lanes own current per-file dispositions. Final reconciliation pending, not inferred clean from previous runs. |

## Runtime and pending adjudication

This lane ran the count diagnostic only in this iteration. The prior fix review
is `../post-closure/fix-astra.md`; it did not claim test execution. Local/referenced,
history on/off, copied/sharded/packed matrices are regression obligations and
source-traced routes, not newly run combinations here. Root's approved suite owns
current runtime outcomes. No GUI runtime, macOS/arm64, large-data performance,
concurrent-failure or filesystem-race proof is claimed.

Independent storage/transport boundary pass complete at the hash above, subject
to the declared carried-forward foundation and exclusions. Subsequent reopens
and cross-lane adjudication follow; no finding-free conclusion is asserted.

## Count-fix reopening and updated freeze

Root folded count validation into PC-CORE-01. Reverified patch SHA-256
`89d698050e0e41d28aa8ec2acb55129048467147b2e40a5f27feb1094fe4daab`.
Only selection implementation/regressions changed: count now derives from actual
entry bytes, the JSON field must be integer and equal that count, and remainder
bytes still reject. This prevents fractional truncation without changing valid
encoding or introducing another format. Native nonempty and Python empty-entry
tests exercise malformed scalar cases. Source review has no blocker; runtime
result remains the approved runner's responsibility.

## NX-C01 — selection header collection shapes

**Accepted, low; pre-existing structural validation omission.** Independent
runtime corroborates the core lane's source candidate: `transforms: null` and
`transforms: {}` are accepted as an empty vector; a JSON object with one complete
reference as its value is accepted as one transform. `origins: null`, `{}` and
an object whose value is a valid `[store,generation]` pair are also accepted.
The decoder loops over nlohmann values without first enforcing the format's
array shape. Per-origin pair checks and per-transform complete-reference checks
are present, but they do not validate the containing collection.

Diagnostic used the same Python import, UUID and VCSEL-v2 byte construction as
above, changing exactly one header field in turn: `(transforms,None)`,
`(transforms,{})`, `(transforms,{named: complete_reference})`, `(origins,None)`,
`(origins,{})`, `(origins,{named:[store,generation]})`. All six decoded without
error; the named transform decoded as one complete transform. No files/builds
were created. The count fix does not touch either iteration and these fixtures
use a valid integer count zero.

The public format specifies arrays and promises rejection of malformed bytes.
This is distinct from scalar conversion: enforce collection shapes at the shared
native artifact boundary, preserving per-entry strict reference validation. It
does not demonstrate loss of full endpoint identity, wrong geometry, or ordinary
producer failure. Native encode uses vectors; Python/native decode share the
affected implementation. The same loops exist at baseline `b73a31f33`.

## NX-P01 — consumer receipt marker reconstruction

**Accepted, low; pre-existing reference-contract duplication.** Read the complete
current consumer ledger and independently inspected
`spiral-fitting/object_store_inputs.py::submit_link_plan/_link_marker_at/_refresh`
and `src/linking/LinkObjects.cpp` marker publication. `submit_link_plan` takes a
number from `receipt.objects`; `_link_marker_at` manufactures a full reference
from that number and the request's owner/generation. The shared
`receipt_object_reference` already supplies the complete selected identity and
is used by `_refresh` in this same file. This is internal post-commit transport,
not an external numeric selector requiring boundary adaptation.

The native marker producer writes metadata with the next commit sequence, so an
ordinary successful LinkObjects run publishes the marker in the committing root.
Therefore the current reconstruction normally names the right marker; no
incorrect-owner runtime failure is claimed. The concrete issue is redundant
number/request interpretation where the contract requires carrying the existing
full receipt reference. Use that helper and pass its result into the exact reader;
do not add fallback reconstruction for incomplete receipts. Source-only evidence.

Consumer inventory/sibling dispositions agree with this lane's current boundary
trace; their explicit GUI/platform/numerical exclusions remain.

## Final cross-lane adjudication

Read completed `core.md` (including the per-file inventory) and `consumers.md`.
Accept **NX-A01, NX-C01 and NX-P01**, all low priority and pre-existing. Count
coercion is folded into the already-fixed PC-CORE-01, not a fourth open finding.
No contradiction found between the independent traces and lane conclusions.
No newly established regression from the three repaired root causes. Findings
are bounded validation/contract issues; none is represented as demonstrated
wrong-owner geometry or data loss.

The aggregate inventory includes the prior complementary format/spatial/helper
ledgers and their retained exclusions, rather than labelling delegated files
clean by default. In particular, a file delegated to this lane is covered only
at the named boundary above or in the explicitly retained prior foundation;
concurrency, numerical internals and platform behavior do not become verified
through delegation. Current source review is complete at the updated hash;
root owns approved runtime completion and final commit mapping. New production
changes must reopen their affected obligations.
