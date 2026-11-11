# Independent post-class-sweep adjudication

## Frozen scope and method

Reviewed `d13d33629` plus the source/test working diff with SHA-256
`90ad71c5ac3c79abcff42509dd0d2c7584d0a43e645180cd13983e23ad01662f`
(`git diff --binary -- vc_object_store/src vc_object_store/include vc_object_store/python spiral-fitting lasagna volume-cartographer vesuvius`).
The new untracked native regression is not included in that diff hash:
`tests/cpp/test_json_boundaries.cpp` independently hashes to
`508749f9df3843001544c0b173fe94161651b8c1612f3db7f7132521f4402cf2`.

Read the review, single-reference and reading contracts and
`class-sweep/fixes.md`; independently inspected the complete production delta
and new native regression, then traced the admission owners and downstream
siblings below. This is a fix-focused invariant review plus candidate
adjudication, not a claim to have re-proved every numerical kernel or every
historical full-inventory entry. No production/test changes, installations or
builds were made by this lane.

## Reopened invariants and dispositions

| Owner / files and symbols | Independent disposition and sibling trace |
| --- | --- |
| `internal/JsonFields.{hpp,cpp}`: `jsonArray`, `jsonArrayField` | Actual array and parent-object admission; absent optional fields use a stable empty array, explicit null rejects. `SharedJson` is the same JSON type, so returning references does not introduce a conversion-temporary lifetime bug. No new representation. |
| `Commands.cpp`: `admitCommandCollections`, `requestHeader`, payload/preparation discovery, decode and raw submission/replay | Root operations and the enumerated nested data arrays, relationships, edits, channels, registrations and ingest items are admitted before payload availability affects the path. Downstream loops consume admitted containers. Existing format-3, expectation and savepoint semantics are not replaced. |
| `Wire.cpp`: list envelopes, conflicts, feeds, geometry arrays, selected inputs, region and proximity results | `unwrapArray` applies only to list-result envelopes; scalar/object envelopes remain separate. Exact result null entries remain allowed. Shared owning decoders cover hosted/cache consumers rather than adding consumer checks. |
| `Wire.cpp::relationshipFrom` | Rejects obsolete archive-array strings; canonical UUIDs and intentionally empty history-off archive selectors remain supported. Full endpoint references are retained. This does not require a live head or substitute attachment identity. |
| `Selections.cpp`, `internal/ObjectVersionJson.hpp`, `python/objects.cpp` | Transforms use the shared reference-list decoder; origins use array admission then existing pair checks. Python list codec delegates both list and scalar interpretation to native code. Existing strict count/scalar logic is preserved. |
| `Snapshots.cpp::importJson`, `internal/Anchors.cpp::dependenciesFromJson` | Snapshot object and edge lists are both admitted before writes; repeated loops share the admitted values and savepoint. Anchor points/index admission is shared by edge/command/history adapters; documented Python optional-null omission is not a regression. Tombstone and placement resolution remain unchanged. |
| `PackedStore.cpp::readDirectory` | Directory arrays are admitted in the common open/inspection/finalization path. Framing, hashes, extents and relocation behavior remain unchanged; no root-only copy or placement shortcut introduced. |
| `ExtensionHistory.cpp::readSnapshot` | Chunks admitted before payload interpretation; restore/exact/current enumeration converge on this owner. Internal repeated iteration is not another missing boundary check. |
| `geometry/GeometryDelta.cpp::loadGeometryManifest`, `GeometryReference.cpp::decodeGeometryRecord`, `GeometryReader.cpp` | Delta chunk admission precedes descriptor use. Record resolution supplies admitted descriptors to forward/reverse reconstruction, including parent traversal. Full manifests continue through `verifiedGeometryManifest`; array-name maps are deliberately not converted to lists. |
| `geometry/SequenceMetadata.cpp`: unpack and metadata payload closure | Both independently serialized-descriptor ingress paths share array extraction. Inline values and arbitrary metadata maps remain valid; no new restriction on arbitrary application JSON. |
| `DraftCheckpoints.cpp::load`, `AttachedDocument.cpp`, `catalog/CatalogStore.cpp` | Pending requests and declared document/catalog collections admitted by their owners. Draft maps, catalog metadata and absent optional collections retain their existing meaning. |
| `imports/SourceRelationships.{hpp,cpp}::validateSourceTopology`, `formats/FiberJson.cpp`, `formats/PointCollectionInput.cpp`, `geometry/GeometryIngestBatch.cpp` | File decode, endpoint discovery and prepared/batch inputs share declared topology admission. `normalizeMetadata` validates before stripping relationship fields from its metadata copy; original topology is retained for relationship insertion. Batch savepoint still rolls back prior entries when a later source rejects. External source-version normalization is legitimate input adaptation, not legacy vcobj-reference repair. |
| `client/HostedStore.cpp` missing-payload response | Shared array admission before upload iteration; retry and unknown-payload guards are retained. No injected malformed HTTP server exercised by this lane. |
| `lasagna/fit2tifxyz.py::_stored_sources` | Saved captured references delegate the complete list to native decoding. No per-element Python identity interpretation or fallback to current objects. |
| `spiral-fitting/object_store_inputs.py::InputPins.from_provenance` | Own application maps and per-kind arrays are checked before normalization; native scalar-reference decoding remains authoritative. Discovery-only absent references remain supported; explicit malformed maps do not become empties. |
| `submit_link_plan`, `_link_marker_at` | Receipt reference is passed unchanged to exact selection; no numeric-owner reconstruction or current-head prerequisite remains. The separate receipt-decoder omission below does not justify restoring consumer reconstruction. |

These changes add admission at serialized boundaries, not guards on every
internal range loop. No new local SQL identity proxy, history-off identity
alternative, automatic command-format selection, or consumer placement
resolver was found in the delta.

## Candidate adjudication

### PCS-01 — Receipt decoder admission (accepted; pre-existing, low)

Source-confirmed in `Versions.cpp::selectionFromJson` and `receiptFromJson`.
The former iterates `.items()` without requiring an object and decodes numeric
values with `get<int64_t>()`; the latter uses that same coercing decode for
`commit_sequence`. Numeric-map keys are retained verbatim, whereas
`object_references` and `TransactionReceipt::objectReference` canonicalize
UUIDs. An uppercase numeric-map key can therefore be missed by canonical
lookup, even when a complete reference exists. Both consistency passes call
that lookup and can skip the mismatched key rather than reject it.

This is an actual serialized receipt boundary: public
`transactionReceiptFromJson` is used by hosted submit/receipt responses,
saved-receipt replay (`Versions.cpp`), and Python `receipt_object_reference`.
It is not merely an unused numeric display projection. Treat map shape,
integer validation and canonical-key handling as manifestations of one
owning receipt-admission gap, not three independent consumer bugs. Fixing
should use shared scalar/UUID handling once in native decoding. Complete
object-reference decoding itself remains strict; no wrong-owner selection,
data corruption or ordinary internally-generated receipt failure is claimed.
These lines are unchanged by the reviewed patch: omission, not regression.

The parent supplied runtime results from approved runner `12181`, using a
real `db.submit` no-op receipt and the public native-backed
`receipt_object_reference` API. History off/on had the same result:

| Mutation | Observed result |
| --- | --- |
| Valid receipt | Same complete reference |
| `objects=[]` or `objects=null` | Accepted, reference absent |
| `relationships=[]` or `relationships=null` | Accepted, same reference |
| Object result version `1.5` for selected version 1 | Accepted, same reference (truncation) |
| Object result version `true` or `0` | Rejected (type error or inconsistent identity) |
| Commit sequence `1.5` or `-1` | Accepted, same reference |
| Commit sequence `true` | Rejected (type error) |
| Uppercase object-result map key with canonical complete-reference key | Accepted, reference absent |

Thus booleans must not be reported as accepted. The demonstrated effect is
malformed receipt acceptance and missed repinning, not invented ownership or
corrupted geometry. This one boundary finding is sufficient; numeric scalar
and map/key manifestations are not separate root-cause counts.

## Verification and limits

This lane ran only read-only source/diff/hash checks. The native regression
was inspected: it covers malformed containers and valid controls, all four
command ingress routes, exact null entries, history on/off relationship
archives, packed directory tampering with recomputed framing/checksum,
snapshot and descriptor boundaries. Direct delta-decoder probes are not a
full tampered-store reconstruction test; malformed HTTP response behavior
and interactive GUI/macOS/arm64 are not established here.

Prior approved results are recorded in `class-sweep/fixes.md` (54 native,
279 store Python, 369 Spiral plus its three known baseline failures, 37
Lasagna, final focused 85 combined). They are historical evidence, not new
executions by this review. The receipt probe results above are explicitly
attributed to the parent's approved runner, not this lane.

## Final cross-lane adjudication

Read `native.md` and `consumers.md` completely after their completion. Agree
with their changed-boundary and sibling dispositions: no regression of the
class-sweep fixes and no additional confirmed consumer defect. The native
ledger's receipt identity construction discussion does not close receipt
schema admission; PCS-01 above is the explicit exception. Consumers should
continue using that native decoder, not add a parallel validator or restore
numeric reconstruction.

The consumer lane's `ObjectCopy.cpp::readObjectCopy` observation is not a new
operational reference bug: documented `copied_from` metadata has no located
reader rebuilding selected handles. Uniform representation would be an
explicit metadata/API cleanup, not evidence of wrong-owner reads. Likewise,
public Python iterable adaptation, native typed loops, independent source
formats and deliberately map-shaped metadata remain exclusions, not missing
array checks.

Final verdict: **one accepted low-severity, pre-existing finding, PCS-01**;
no additional confirmed regression in the frozen fix delta. All lane
adjudications for this pass are complete, with runtime/platform limits above.
Unchanged full-inventory evidence is inherited transparently from the linked
prior ledgers rather than claimed as a fresh line-by-line whole-repository
audit. No production or test files were edited by this review lane.
